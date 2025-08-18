import copy
import os
from pathlib import Path

import torch
from slime.data.data import Dataset
from transformers import AutoTokenizer


# TODO may further refactor data-loading part later
class RolloutDataSet:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.n_sample_per_prompt = self.args.n_samples_per_prompt

        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        # TODO move (during the refactor)
        if (d := args.dump_details) is not None:
            tokenizer.save_pretrained(Path(d) / "tokenizer")

        # TODO "default" key is temporaly solution, should further make data_source key in data like verl did
        self.dataset = Dataset(
            "default",
            args.prompt_data,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            input_key=args.input_key,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            seed=args.rollout_seed,
        )
        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def get_sample(self):
        if self.sample_offset >= len(self.dataset):
            self.epoch_id += 1
            if self.args.num_epoch is not None and self.epoch_id >= self.args.num_epoch:
                return None
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            self.sample_offset == 0
        data = self.dataset.samples[self.sample_offset]
        self.sample_offset += 1
        data_group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(data)
            sample.index = self.sample_index
            self.sample_index += 1
            data_group.append(sample)
        return data_group

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

        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)
