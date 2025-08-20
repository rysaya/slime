from slime.utils.mask_utils import MultiTurnLossMaskGenerator

__all__ = ["generate_rollout"]


async def generate_rollout(args, sample_group, tokenizer, sampling_params, aborted=False):
    global MASK_GENERATOR

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    for sample in sample_group:
        (sample,) = sample
        messages = sample["prompt"]
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages)
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample["tokens"] = token_ids
        sample["response_length"] = response_length
        sample["reward"] = 0
        sample["loss_mask"] = loss_mask[-response_length:]

    return sample_group
