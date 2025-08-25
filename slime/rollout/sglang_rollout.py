import asyncio
from copy import deepcopy

from slime.utils.http_utils import post
from slime.utils.misc import load_function
from slime.utils.types import Sample, SampleStatus
from .rm_hub import async_rm, batched_async_rm
from slime.utils.types import GenerateState

__all__ = ["create_rollout_fn"]


async def generate_one_sample_vanilla(args, tokenizer, sample: Sample, raw_sampling_params) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample["status"] == SampleStatus.PENDING or sample["status"] == SampleStatus.ABORTED
    ), f"Sample status is {sample['status']}"

    sampling_params = deepcopy(raw_sampling_params)
    if len(sample["response"]) > 0:
        sampling_params["max_new_tokens"] -= len(sample.get("tokens", [])) - len(sample["prompt_ids"])

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0, len existing tokens: {len(sample.get('tokens', []))}, len prompt tokens: {len(sample['prompt_ids'])}"
    if sampling_params["max_new_tokens"] == 0:
        sample["status"] = SampleStatus.TRUNCATED
        return sample

    # Token-based mode: use tokens directly
    if len(sample["response"]) > 0:
        input_token_ids = sample["tokens"]
    else:
        # First turn: initialize with prompt tokens
        prompt_token_ids = sample["prompt_ids"]
        input_token_ids = prompt_token_ids
        # Initialize sample.tokens with prompt for subsequent turns
        if "tokens" not in sample:  # Only set if empty
            sample["tokens"] = prompt_token_ids

    # Prepare payload - shared structure
    payload = {
        "input_ids": input_token_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    output = await post(url, payload, use_http2=args.use_http2)

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens = []

    # Update sample with tokens directly - avoiding re-tokenization
    sample["tokens"] = sample["tokens"] + new_response_tokens
    if "response_length" not in sample:
        sample["response_length"] = 0
    sample["response_length"] += len(new_response_tokens)
    sample["response"] += output["text"]

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

    sample_group = await asyncio.gather(
        *[generate_and_rm(args, sample, tokenizer, sampling_params) for sample in sample_group]
    )

    # for the rm that need the whole group, we will not do the rm here
    if not gen_state.is_aborted() and args.group_rm:
        rewards = await batched_async_rm(args, sample_group)
        for sample, reward in zip(sample_group, rewards):
            sample["reward"] = reward

    if not gen_state.is_aborted() and sample_group[0]["index"] == 1:
        print(
            f"First rollout sample: {[sample_group[0]['prompt'] + sample_group[0]['response']]}, label: {sample_group[0]['label']}, reward: {sample_group[0]['reward']}",
            flush=True,
        )

    return sample_group
