import asyncio
from copy import deepcopy

from slime.utils.http_utils import post
from slime.utils.mask_utils import get_response_lengths
from slime.utils.misc import load_function
from slime.utils.types import GenerateState, Sample, SampleStatus

from .rm_hub import async_rm, batched_async_rm

__all__ = ["create_rollout_fn"]


async def generate_one_sample_vanilla(args, tokenizer, sample: Sample, raw_sampling_params) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample["status"] == SampleStatus.PENDING or sample["status"] == SampleStatus.ABORTED
    ), f"Sample status is {sample['status']}"

    # Process prompt to create text and image payload
    sampling_params = deepcopy(raw_sampling_params)
    image_data = sample.get("image_data", [])

    if len(sample["response"]) > 0:
        sampling_params["max_new_tokens"] -= len(sample.get("tokens", [])) - sample["prompt_ids_len"]

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0, len existing tokens: {len(sample.get('tokens', []))}, len prompt tokens: {sample['prompt_ids_len']}"
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
        payload["input_ids"] = sample["tokens"]

    output = await post(url, payload)

    # Extract new response tokens

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        assert not args.partial_rollout, "Currently parital rollout is not suppurted when using slime router"
        retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
        retrieve_payload = {"text": sample["prompt"] + output["text"], "return_logp": True}
        retrieve_output = await post(retrieve_url, retrieve_payload)
        sample["tokens"] = retrieve_output["tokens"]
        sample["response"] += output["text"]
        sample["loss_mask"] = retrieve_output["loss_mask"]
        sample["response_length"] = get_response_lengths([sample["loss_mask"]])[0]
        sample["loss_mask"] = sample["loss_mask"][-sample["response_length"] :]
        sample["rollout_log_probs"] = retrieve_output["rollout_logp"][-sample["response_length"] :]
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample["tokens"] = sample.get("tokens", []) + new_response_tokens
        sample["response_length"] = sample.get("response_length", 0) + len(new_response_tokens)
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
