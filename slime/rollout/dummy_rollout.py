def generate_rollout(args, sample_group, tokenizer, sampling_params, aborted=False):
    """
    Dummy rollout function that does nothing except printing the sample
    This is used when the actual rollout logic is not needed.
    """
    if sample_group[0]["index"] == 1:
        print(f"First rollout sample: {sample_group[0]}", flush=True)
    return sample_group
