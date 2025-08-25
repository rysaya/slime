import asyncio
import logging
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

import ray
import torch

from slime.data import dummy_convert_func
from slime.utils.http_utils import post, get
from slime.utils.types import Sample, GenerateState
from slime.utils.async_utils import run
from slime.utils.ray_utils import Box
from .controller import RolloutControllerBase, dummy_log_func

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class RolloutControllerWithBuffer(RolloutControllerBase):
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
        super().__init__(args, tag, wandb_run_id, datasource_cls, rollout_function_path, post_process_func, log_func)
        self.buffer = []
        self.gen_state = GenerateState()

    def generate(self, rollout_id):
        # TODO 先留你不杀
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            data = run(self.generate_rollout_async(rollout_id))

        # TODO 先留你不杀
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=rollout_id))
            print(f"{self.tag}: Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save([d.to_dict() for d in data], path)
        data = self.post_process_func(self.args, data)
        self.log_func(rollout_id, self.args, data)
        return Box(ray.put(data))

    """
    方法：
    1. 先往生产者里塞一个rollout batch size的数据
    2. 当一个没通过dynamatic filter，就往里面再塞一个
    3. 如果一个pass了dynamatic filter，就有1/2概率往里面再塞一个，防止长尾半天出不来
    4. xxx概率可以不用random，直接看已完成的len(data)判断，比如1/2的话就是看其是奇数还是偶数就行
    """

    async def generate_rollout_async(self, rollout_id: int):
        # TODO add a better over-sampling strategy back
        results = []
        sample_idx = 1
        pbar = tqdm(total=self.rollout_batch_size, desc=f"{self.tag} Rollout generation")

        semaphore = asyncio.Semaphore(self.args.rollout_batch_size)  # 最大并发数为args的rollout_batch_size
        tasks = set([])

        # 按顺序提交任务
        async with semaphore:

            def submit_sample(sample, sample_idx=None):
                for s in sample:
                    s.set_rollout_id(rollout_id)
                    if sample_idx is not None:
                        s.set_index(sample_idx)
                task = asyncio.create_task(
                    self.generate_func(self.args, sample, self.tokenizer, deepcopy(self.sampling_paras))
                )
                tasks.add(task)

            # samples in buffer already have sample_idx, no need to set again
            for sample in self.buffer:
                submit_sample(sample)

            for _ in range(self.rollout_batch_size - len(self.buffer)):
                sample = self.data_source.get_sample()
                if sample is None:
                    break
                submit_sample(sample, sample_idx=sample_idx)
                sample_idx += 1

            # 按顺序收集结果
            while len(results) < self.rollout_batch_size and len(tasks) > 0:
                # wait for the generation to finish
                raw_done_tasks, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                new_done_results = []
                new_task_nums = 0
                for t in raw_done_tasks:
                    group: list[Sample] = t.result()

                    assert (
                        len(group) == self.data_source.n_samples_per_prompt
                    ), f"{self.tag}: We expect the generation per group is {self.data_source.n_samples_per_prompt}, but got {len(group)}"
                    # not pass dynamic_filter, add a new task into the queue
                    if self.dynamic_filter is not None and not self.dynamic_filter(self.args, group):
                        new_task_nums += 1
                    else:
                        new_done_results.append(group)
                        # passed the dynamic_filter, if the total done_results is odds, add a new task into the queue
                        if (len(results) + len(new_done_results)) % 2 == 0:
                            new_task_nums += 1

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

        # there are still some unfinished requests, abort them
        aborted_samples = await self.abort(tasks, rollout_id)

        if len(results) != self.rollout_batch_size:
            print(
                f"{self.tag}: Warning! Sample Num not same as batch size. Got {len(results)} samples, expected {self.rollout_batch_size}"
            )
        results = sorted(results, key=lambda group: group[0]["index"])

        # TODO: 这里暂时先截断rollout_batch_size数量，megatron那边sample分配debug还没成功，等成功再撤
        results = results[: self.rollout_batch_size]

        # flatten the data if it is a list of lists
        if isinstance(results[0], list):
            results = sum(results, [])
        self.buffer_append(aborted_samples)
        # reset the aborted state to prevent effects on the next rollout or eval.
        self.gen_state.reset()
        return results

    async def abort(self, pendings, rollout_id: int):
        aborted_samples = []

        self.gen_state.abort()
        list_workers_resp = await get(
            f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/list_workers",
            use_http2=self.args.use_http2,
        )
        worker_urls = list_workers_resp["urls"]

        # abort all the requests
        for url in worker_urls:
            print(f"{self.tag}: Abort request for {url}", flush=True)
            await post(f"{url}/abort_request", {"abort_all": True}, use_http2=False)

        # make sure all the pending tasks are finished
        count = 0
        while pendings:
            done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)

            # for partial rollout, collect the partial samples into the data buffer
            for task in done:
                group = task.result()
                for sample in group:
                    if sample["response"] and not sample.get_metadata("start_rollout_id"):
                        sample.add_metadata("start_rollout_id", rollout_id)
                aborted_samples.append(group)
                count += len(group)

        print(f"{self.tag}: Collected {count} partial samples into the data buffer", flush=True)

        return aborted_samples

    def buffer_append(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.data_source.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.data_source.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)
