from slime.utils.mask_utils import MultiTurnLossMaskGenerator

__all__ = ["generate_rollout"]


MASK_GENERATOR = None


async def generate_rollout(args, sample_group, tokenizer, sampling_params, aborted=False):
    global MASK_GENERATOR

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    for sample in sample_group:
        messages = sample["raw_messages"]
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages)
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample["tokens"] = token_ids
        sample["response_length"] = response_length
        sample["loss_mask"] = loss_mask[-response_length:]

    if sample_group[0]["index"] == 1:
        print(
            f"First rollout sample: {sample_group[0]['input_ids']}, raw_messages: {sample_group[0]['raw_messages']}, rollout_id: {sample_group[0]['rollout_id']}",
            flush=True,
        )

    return sample_group
