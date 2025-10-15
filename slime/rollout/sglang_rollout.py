import asyncio
from copy import deepcopy
import base64
import copy
import io
from argparse import Namespace
from typing import Any, Callable, Union

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.http_utils import get, post
from slime.utils.mask_utils import get_response_lengths
from slime.utils.types import Sample

from slime.utils.http_utils import post
from slime.utils.misc import load_function
from slime.utils.types import Sample, SampleStatus
from .rm_hub import async_rm, batched_async_rm
from slime.utils.types import GenerateState

__all__ = ["create_rollout_fn"]


def _load_and_encode_image(path: str) -> str:
    """Load an image from path, ensure RGB, encode as JPEG base64 string."""
    with Image.open(path) as image:
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def generate_one_sample_vanilla(args, tokenizer, sample: Sample, raw_sampling_params) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample["status"] == SampleStatus.PENDING or sample["status"] == SampleStatus.ABORTED
    ), f"Sample status is {sample['status']}"

    # Process prompt to create text and image payload
    sampling_params = deepcopy(raw_sampling_params)
    image_data = []
    if isinstance(sample["prompt"], str):
        text_prompt = sample["prompt"]
    else:  # Multimodal prompt (list of dicts)
        text_prompt = ""
        # sglang uses a placeholder to insert image features
        image_token = tokenizer.special_tokens_map.get("image_token", "<image>")
        for part in sample["prompt"]:
            if part["type"] == "text":
                text_prompt += part["text"]
            elif part["type"] == "image":
                text_prompt += image_token
                try:
                    img_b64 = await asyncio.to_thread(_load_and_encode_image, part["path"])
                    image_data.append(img_b64)
                except Exception as e:
                    print(f"Error processing image {part['path']}: {e}")
                    sample["status"] = SampleStatus.ABORTED
                    return sample


    if len(sample["response"]) > 0:
        sampling_params["max_new_tokens"] -= len(sample.get("tokens", [])) - len(sample["prompt_ids"])

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0, len existing tokens: {len(sample.get('tokens', []))}, len prompt tokens: {len(sample['prompt_ids'])}"
    if sampling_params["max_new_tokens"] == 0:
        sample["status"] = SampleStatus.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample["response"]) > 0:
        payload["input_ids"] = sample["tokens"]
    else:
        prompt_token_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        payload["input_ids"] = prompt_token_ids
        if not sample["tokens"]:  # Initialize sample.tokens for the first turn
            sample["tokens"] = prompt_token_ids

    output = await post(url, payload)

    # Extract new response tokens

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        assert not args.partial_rollout, "Currently parital rollout is not suppurted when using slime router"
        retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
        retrieve_payload = {"text": sample.prompt + output["text"], "return_logp": True}
        retrieve_output = await post(retrieve_url, retrieve_payload)
        sample["tokens"] = retrieve_output["tokens"]
        sample["response"] += output["text"]
        sample["loss_mask"] = retrieve_output["loss_mask"]
        sample["response_length"] = get_response_lengths([sample.loss_mask])[0]
        sample["loss_mask"] = sample["loss_mask"][-sample.response_length :]
        sample["rollout_log_probs"] = retrieve_output["rollout_logp"][-sample["response_length"] :]
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample["tokens"] = sample.tokens + new_response_tokens
        sample["response_length"] += len(new_response_tokens)
        sample["response"] += output["text"]

        if "rollout_log_probs" not in sample:
            sample["rollout_log_probs"] = []
        sample["rollout_log_probs"] += new_response_log_probs

    if "weight_version" in output["meta_info"]:
        if "weight_version" not in sample:
            sample["weight_version"] = []
        sample["weight_version"].append(output["meta_info"]["weight_version"])

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.set_status(SampleStatus.TRUNCATED)
        case "abort":
            sample.set_status(SampleStatus.ABORTED)
        case "stop":
            sample.set_status(SampleStatus.COMPLETED)

    return sample


async def generate_and_rm(args, sample: Sample, tokenizer, sampling_params) -> Sample:
    # For samples with existing response, check if they're complete
    if sample["status"] == SampleStatus.COMPLETED or sample["status"] == SampleStatus.TRUNCATED:
        assert sample["response"] != "", "Sample response should not be empty if status is completed or truncated"
        if not args.group_rm:
            assert sample.get("reward", None) is not None
        return sample

    # generate
    if GenerateState().is_aborted():
        sample.set_status(SampleStatus.ABORTED)
        return sample

    if args.custom_generate_function_path is not None:
        custom_generate_func = load_function(args.custom_generate_function_path)
        sample = await custom_generate_func(args, tokenizer, sample, sampling_params)
    else:
        sample = await generate_one_sample_vanilla(args, tokenizer, sample, sampling_params)

    if sample["status"] == SampleStatus.ABORTED:
        return sample

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    sample["reward"] = await async_rm(args, sample)

    return sample


async def generate_rollout(args, sample_group, tokenizer, sampling_params) -> list[Sample]:
    gen_state = GenerateState()
    if gen_state.is_aborted():
        return sample_group

    tasks = []
    for idx, sample in enumerate(sample_group):
        current_sampling_params = sampling_params.copy()       
        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            seed = sampling_seed_base + idx
            current_sampling_params["sampling_seed"] = seed
        tasks.append(generate_and_rm(args, sample, tokenizer, current_sampling_params))

    sample_group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will not do the rm here
    if not gen_state.is_aborted() and args.group_rm:
        rewards = await batched_async_rm(args, sample_group)
        for sample, reward in zip(sample_group, rewards):
            sample["reward"] = reward

    sample = sample_group[0][0] if isinstance(sample_group[0], list) else sample_group[0]
    if not gen_state.is_aborted() and sample["index"] == 1:
        print(
            f"First rollout sample: {[sample['prompt'] + sample['response']]}, label: {sample['label']}, reward: {sample['reward']}",
            flush=True,
        )
    return sample_group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")

    # abort all the requests
    for url in response["urls"]:
        print(f"Abort request for {url}", flush=True)
        await post(f"{url}/abort_request", {"abort_all": True})

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        print(f"Collected {count} partial samples into the data buffer", flush=True)

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[list[list[Sample]], list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[list[list[Sample]], list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                print(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                    flush=True,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            dynamic_filter_output = _call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    print(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
        flush=True,
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return RolloutFnTrainOutput(samples=data), aborted_samples


def _call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, name: str, path: str
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        name: str, the name of the dataset
        path: str, the path of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    if name not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
        )

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([str(sample.prompt) + sample.response], sample.reward, flush=True)
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
        }
    }


# TODO remove this temp function
def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> Union[RolloutFnTrainOutput, RolloutFnEvalOutput]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    output, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return output
