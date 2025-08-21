import random
import os
import torch

from typing import Union
from slime.utils.types import Sample
from transformers import AutoTokenizer
from datasets import Dataset as hf_ds

__all__ = ["Dataset"]


# TODO: don't read the whole file into memory.
def read_file(path):
    if path.endswith(".jsonl") or path.endswith(".json"):
        ds = hf_ds.from_json(path)
    elif path.endswith(".parquet"):
        ds = hf_ds.from_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    for data in ds:
        yield data


def dummy_convert_func(samples: Union[list[Sample], list[list[Sample]]]):
    return samples


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

    def init_dataset(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

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
