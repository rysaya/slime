import time


async def sleep(args, sample_group, tokenizer, sampling_params):
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        print(f"rollout sleep for {count} hours")
