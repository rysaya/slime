import asyncio
import logging
from typing import Union
import wandb
import ray
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.rollout.sampling_params import compute_sampling_params, eval_sampling_params
from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.utils.async_utils import run
from slime.utils.ray_utils import Box
from slime.utils.wandb_utils import init_wandb_secondary

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def log_eval_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    print(f"eval {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["eval/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)


def pop_first(args, buffer: list[list[Sample]], num_samples: int = -1) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples) if num_samples > 0 else len(buffer)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


def dummy_convert_func(samples: Union[list[Sample], list[list[Sample]]]):
    return samples


def dummy_log_func(rollout_id, args, data):
    return


def convert_eval_samples_to_metrix(args, samples: Union[list[Sample], list[list[Sample]]]):
    reward_key = args.eval_reward_key
    eval_metrics = {}

    for s in samples:
        if s.data_source not in eval_metrics:
            eval_metrics[s.data_source] = {"rewards": [], "truncated": []}
        eval_metrics[s.data_source]["rewards"].append(s.reward if not reward_key else s.reward[reward_key])
        eval_metrics[s.data_source]["truncated"].append(s.status == Sample.Status.TRUNCATED)

    return eval_metrics


def convert_rl_samples_to_train(args, samples: Union[list[Sample], list[list[Sample]]]):
    """
    Convert inference generated samples to training data.
    """
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": [sample.get_reward_value(args) for sample in samples],
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]
    return train_data


class RolloutControllerBase:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        args,
        tag,
        wandb_run_id,
        datasource_cls,
        rollout_function_path,
        post_process_func=dummy_convert_func,
        log_func=dummy_log_func,
    ):
        self.tag = tag
        # TODO whether make it init args?
        self.sampling_paras = compute_sampling_params(args) if tag == "train" else eval_sampling_params(args)
        self.args = args
        init_wandb_secondary(args, wandb_run_id)
        self.generate_func = load_function(rollout_function_path)
        self.post_process_func = post_process_func
        self.log_func = log_func
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        # instantiate data filters
        self.dynamic_filter = (
            load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
        )
        self.data_source = datasource_cls(self.args)

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        data = run(self.generate_rollout_async(rollout_id))
        data = self.post_process_func(self.args, data)
        self.log_func(rollout_id, self.args, data)
        return Box(ray.put(data))

    async def generate_rollout_async(self, rollout_id: int):
        results = []
        sample_idx = 1
        do_print = True
        pbar = tqdm(total=self.args.rollout_batch_size, desc=f"{self.tag} Rollout generation")

        semaphore = asyncio.Semaphore(self.args.rollout_batch_size)  # 最大并发数为rollout_batch_size
        tasks = []

        # 按顺序提交任务
        async with semaphore:

            def submit_sample(sample, sample_idx=None):
                for s in sample:
                    s.set_rollout_id(rollout_id)
                    if sample_idx is not None:
                        s.set_sample_id(sample_idx)
                task = asyncio.create_task(self.generate_func(self.args, sample, self.tokenizer, self.sampling_paras))
                tasks.append(task)

            for _ in range(self.args.rollout_batch_size):
                sample = self.data_source.get_sample()
                if sample is None:
                    break
                submit_sample(sample, sample_idx=sample_idx)
                sample_idx += 1

            # 按顺序收集结果
            while len(results) < self.args.rollout_batch_size and len(tasks) > 0:
                # wait for the generation to finish
                raw_done_tasks, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                new_done_results = []
                new_task_nums = 0
                for t in raw_done_tasks:
                    group: list[Sample] = t.result()

                    if do_print:
                        print(
                            f"{self.tag}: First rollout sample: {[group[0].prompt + group[0].response]}, label: {group[0].label}, reward: {group[0].reward}",
                            flush=True,
                        )
                        do_print = False

                    assert (
                        len(group) == self.data_source.n_sample_per_prompt
                    ), f"{self.tag}: We expect the generation per group is {self.data_source.n_sample_per_prompt}, but got {len(group)}"
                    # not pass dynamic_filter, add a new task into the queue
                    if self.dynamic_filter is not None and not self.dynamic_filter(self.args, group):
                        new_task_nums += 1
                    else:
                        new_done_results.append(group)

                # submit new generations, TODO: better refactor
                for _ in range(new_task_nums):
                    sample = self.data_source.get_sample()
                    if sample is None:
                        break
                    submit_sample(sample, sample_idx=sample_idx)
                    sample_idx += 1

                results.extend(new_done_results)
                pbar.update(len(new_done_results))

            pbar.close()
        # TODO 先留你不杀
        print(
            f"{self.tag}: Finish rollout: {[results[-1][0].prompt + results[-1][0].response]}, label: {results[-1][0].label}, reward: {results[-1][0].reward}",
            flush=True,
        )

        if len(results) != self.args.rollout_batch_size:
            print(
                f"{self.tag}: Warning! Sample Num less than batch size. Got {len(results)} samples, expected {self.args.rollout_batch_size}"
            )
        results = sorted(results, key=lambda group: group[0].index)

        # flatten the data if it is a list of lists
        if isinstance(results[0], list):
            results = sum(results, [])
        return results

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)


# ray do not support inheritance of remote classes, so we cannot add @ray.remote to RolloutControllerBase class
# But we could make a dummy class to let it remote
@ray.remote
class RolloutController(RolloutControllerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
