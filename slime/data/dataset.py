import base64
import gzip
import io
import json
import multiprocessing
import os
import random

import pandas as pd
import ray
import torch
import torch.distributed as dist
from PIL import Image
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer

from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.timer import Timer

__all__ = ["Dataset"]


# TODO: don't read the whole file into memory.
def read_file(file_name):
    file_suffix = file_name.rsplit(".", 1)[-1]
    datas = None
    if file_suffix == "json":
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    elif file_suffix == "jsonl":
        with open(file_name, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elif file_suffix == "gz" and file_name.endswith("jsonl.gz"):
        with gzip.open(file_name, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elif file_suffix == "gz" and file_name.endswith("json.gz"):
        with gzip.open(file_name, "rt", encoding="utf-8") as f:
            return json.load(f)
    elif file_suffix in ["csv", "tsv"]:
        sep = "," if file_suffix == "csv" else "\t"
        datas = pd.read_csv(file_name, sep=sep)
    elif file_suffix in ["xlsx", "xls"]:
        datas = pd.read_excel(file_name)
    elif file_suffix == "parquet":
        datas = pd.read_parquet(file_name, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {file_name}.")
    data_dict = datas.to_dict(orient="records")
    return data_dict


def load_and_encode_image(path: str) -> str:
    """Load an image from path, ensure RGB, encode as JPEG base64 string."""
    with Image.open(path) as image:
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for l in total_lengths:
        for i in range(len(batches)):
            if batches[i] + l <= max_tokens_per_gpu:
                batches[i] += l
                break
        else:
            batches.append(l)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    rollout_data = {}

    rank = dist.get_rank()
    if rank == 0:
        data = ray.get(rollout_data_ref.inner)
        dist.broadcast_object_list([data], src=0)
    else:
        data = [None]
        dist.broadcast_object_list(data, src=0)
        data = data[0]

    # save the unprocessed reward for logging
    rollout_data["raw_reward"] = data["raw_reward"]

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths

    if args.balance_data:
        # Group-aware partitioning to keep each group together
        n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
        # Calculate group-level lengths (sum of lengths for each group)
        num_groups = len(total_lengths) // n_samples_per_prompt
        group_lengths = []
        for i in range(num_groups):
            start_idx = i * n_samples_per_prompt
            end_idx = start_idx + n_samples_per_prompt
            group_total_length = sum(total_lengths[start_idx:end_idx])
            group_lengths.append(group_total_length)

        # Get partitions at group level
        group_partitions = get_seqlen_balanced_partitions(group_lengths, dp_size, equal_size=True)

        # Expand group partitions to trajectory level
        parititions = []
        for dp_rank_groups in group_partitions:
            trajectory_indices = []
            for group_idx in dp_rank_groups:
                # Add all trajectories in this group
                start_idx = group_idx * n_samples_per_prompt
                end_idx = start_idx + n_samples_per_prompt
                trajectory_indices.extend(range(start_idx, end_idx))
            parititions.append(trajectory_indices)

    def get_partition(val):
        if args.balance_data:
            return [val[i] for i in parititions[dp_rank]]
        else:
            return val[dp_rank::dp_size]

    for key in [
        "tokens",
        "total_lengths",
        "response_lengths",
        "rewards",
        "truncated",
        "loss_masks",
        "round_number",
        "sample_indices",
        "rollout_log_probs",
        "prompt",
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        rollout_data[key] = val

    return rollout_data


# TODO: 写的很烂
class Dataset:
    def __init__(self, args, path):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.max_epoch = args.num_epoch
        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.seed = args.dataset_seed
        self.data_path_info = self.get_data_path_info(path)
        self.origin_samples = None
        self.samples = None
        self.n_samples_per_prompt = 1

    def process_datas(self, datas):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def chunk_data(self, data, chunk_size):
        """将数据按chunk_size分块"""
        n_chunks = (len(data) + chunk_size - 1) // chunk_size
        chunks = []

        for i in range(n_chunks):
            start = i * chunk_size
            if i == n_chunks - 1:
                end = len(data)
            else:
                end = (i + 1) * chunk_size
            chunks.append(data[start:end])

        return chunks

    def init_dataset(self):
        all_datas = []
        for name, data_path in self.data_path_info.items():
            for data in read_file(data_path):
                data["data_path_info"] = name
                all_datas.append(data)
        all_datas = self.chunk_data(all_datas, 128)
        # use chunksize=1 but chunk datas by hand previously for batch tokenizer
        self.origin_samples = process_map(
            self.process_datas,
            all_datas,
            max_workers=multiprocessing.cpu_count() - 8,
            chunksize=1,
            desc="Processing raw datasets",
        )
        self.origin_samples = [s for sublist in self.origin_samples for s in sublist]
        self.samples = self.origin_samples
        if self.args.shuffle_dataset:
            self.shuffle(self.epoch_id)

    def get_sample(self):
        if self.sample_offset >= len(self.samples):
            self.epoch_id += 1
            if self.max_epoch is not None and self.epoch_id >= self.max_epoch:
                return None
            if self.args.shuffle_dataset:
                self.shuffle(self.epoch_id)
            self.sample_offset = 0
        data = self.samples[self.sample_offset]
        self.sample_offset += 1
        # need to return list for compatibility with rollout dataset
        return [data]

    def get_data_path_info(self, path):
        data_path_info = {}

        def get_single_path(p):
            if ":" in p:
                p_name, p_file = p.split(":", 1)
                data_path_info[p_name] = p_file
            else:
                data_path_info["default"] = p

        if isinstance(path, str):
            get_single_path(path)
        elif isinstance(path, list):
            for p in path:
                get_single_path(p)
        else:
            raise ValueError(f"Unsupported path type: {type(path)}. Expected str or list of str.")
        return data_path_info

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def save(self, rollout_id):
        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_index": self.sample_index,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist.")
            return

        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_index = state_dict.get("sample_index", 0)

        if self.args.shuffle_dataset:
            self.shuffle(self.epoch_id)
