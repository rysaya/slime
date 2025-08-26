import multiprocessing
import random
import time

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.controller import RolloutControllerWithBuffer, RolloutController
from slime.ray.controller import log_eval_data
from slime.utils.http_utils import find_available_port, get_host_info, run_router
from slime.data import EvalDataset, convert_eval_samples_to_metrix
from .utils import Lock, NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from typing import List


def create_rollout_engines(args, pg):
    if args.debug_train_only:
        return []

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.rollout_num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    pg, reordered_bundle_indices = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        rollout_engines.append(
            RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}},
            ).remote(args, rank=i)
        )

    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.rollout_num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]
    for rank, engine in enumerate(rollout_engines):
        if rank % num_engines_per_node != 0:
            continue

        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.rollout_num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_per_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i in range(num_engines):
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    ray.get(init_handles)

    if args.colocate:
        ray.get([engine.release_memory_occupation.remote() for engine in rollout_engines])

    return rollout_engines


def _start_router(args):
    if args.sglang_router_ip is not None:
        return

    from sglang_router.launch_router import RouterArgs

    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    router_args = RouterArgs(
        host=args.sglang_router_ip,
        port=args.sglang_router_port,
        balance_abs_threshold=0,
    )

    if hasattr(router_args, "log_level"):
        router_args.log_level = "warn"

    if hasattr(router_args, "request_timeout_secs"):
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    # If router ip is specified, use the specified launched router
    print(f"SGLang router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


class RolloutManager:
    def __init__(self, args, pg, wandb_run_id, init_gen_engines=True):
        self.args = args
        if init_gen_engines:
            _start_router(self.args)
            self.all_rollout_engines = create_rollout_engines(self.args, pg)
            nodes_per_engine = max(1, self.args.rollout_num_gpus_per_engine // self.args.rollout_num_gpus_per_node)
            # when doing multi-node serving, we will only send request to node-0 for each engine.
            self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
            self.rollout_engine_lock = Lock.options(
                num_cpus=1,
                num_gpus=0,
            ).remote()
        else:
            self.rollout_engines = None
            self.rollout_engine_lock = None

        data_loader_cls = RolloutControllerWithBuffer if args.partial_rollout else RolloutController
        dataset_cls, post_process_func = self._get_train_cls_funcs()

        # TODO make RolloutDataset to configurable cls load
        self.train_data_loader = data_loader_cls.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(
            args,
            "train",
            wandb_run_id,
            dataset_cls,
            args.rollout_function_path,
            post_process_func=post_process_func,
        )
        print(f"import {args.rollout_function_path} as generate_rollout function.")
        if args.eval_files is not None and args.eval_interval > 0 and not args.debug_train_only:
            self.eval_data_loader = RolloutController.options(
                num_cpus=1,
                num_gpus=0,
            ).remote(
                args,
                "eval",
                wandb_run_id,
                EvalDataset,
                args.eval_function_path,
                post_process_func=convert_eval_samples_to_metrix,
                log_func=log_eval_data,
            )
            print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")
        else:
            self.eval_data_loader = None

    def _get_train_cls_funcs(self):
        if self.args.train_type == "sft":
            from slime.data import SFTDataset, convert_sft_samples_to_train

            return SFTDataset, convert_sft_samples_to_train
        elif self.args.train_type == "rm":
            from slime.data import RewardDataset, convert_rm_samples_to_train

            return RewardDataset, convert_rm_samples_to_train
        elif self.args.train_type == "rl":
            from slime.data import RolloutDataset, convert_rl_samples_to_train

            return RolloutDataset, convert_rl_samples_to_train
        else:
            raise ValueError(f"Unknown train type: {self.args.train_type}")

    def async_generate(self, rollout_id):
        return self.train_data_loader.generate.remote(rollout_id)

    def async_eval(self, rollout_id):
        if self.eval_data_loader is None:
            return []
        return self.eval_data_loader.generate.remote(rollout_id)

    def async_offload(self):
        if self.rollout_engines is None:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.rollout_engines]

    def async_onload(self, tags: List[str] = None):
        if self.rollout_engines is None:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines]
