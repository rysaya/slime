import copy
import torch
import json
import numpy as np
from slime.data.dataset import Dataset
from slime.utils.types import Sample, SampleStatus
from slime.data.templates import get_chat_template


def convert_rl_samples_to_train(args, samples: list[Sample]):
    """
    Convert inference generated samples to training data.
    """
    train_data = {
        "tokens": [sample["tokens"] for sample in samples],
        "response_lengths": [sample["response_length"] for sample in samples],
        "rewards": [sample["reward"] for sample in samples],
        "truncated": [1 if sample["status"] == SampleStatus.TRUNCATED else 0 for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.get("loss_mask", None) is None:
            sample["loss_masks"] = [1] * sample["response_length"]
        assert (
            len(sample["loss_masks"]) == sample["response_length"]
        ), f"loss mask length {len(sample['loss_masks'])} != response length {sample['response_length']}"
        loss_masks.append(sample["loss_masks"])
    train_data["loss_masks"] = loss_masks

    rewards = train_data["rewards"]
    # overwriting the raw reward
    if samples[0].get_metadata("raw_reward"):
        train_data["raw_reward"] = [sample.get_metadata("raw_reward") for sample in samples]

    # For rollout buffer
    if samples[0].get_metadata("round_number"):
        train_data["round_number"] = [sample.get_metadata("round_number") for sample in samples]

    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        # group norm
        rewards = torch.tensor([r for r in rewards], dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            # when samples count are not equal in each group
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        rewards = rewards.flatten().tolist()
        train_data["rewards"] = rewards

    return train_data


def convert_eval_samples_to_metrix(args, samples: list[Sample]):
    eval_metrics = {}

    for s in samples:
        if s["data_source"] not in eval_metrics:
            eval_metrics[s["data_source"]] = {"rewards": [], "truncated": []}
        eval_metrics[s["data_source"]]["rewards"].append(s["reward"])
        eval_metrics[s["data_source"]]["truncated"].append(s["status"] == SampleStatus.TRUNCATED)

    return eval_metrics


class RolloutDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.n_samples_per_prompt = self.args.n_samples_per_prompt
        self.init_dataset()

    def process_datas(self, datas):
        all_prompts = []
        for data in datas:
            prompt = data[self.args.input_key]
            if self.args.chat_template:
                chat_template = get_chat_template(self.args.chat_template)
                prompt = chat_template(prompt, self.tokenizer)
            else:
                if self.args.tool_key is not None:
                    tools = data[self.args.tool_key]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    elif isinstance(tools, np.ndarray):
                        tools = tools.tolist()
                    assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                else:
                    tools = None
                prompt = self.tokenizer.apply_chat_template(prompt, tools, tokenize=False, add_generation_prompt=True)
                all_prompts.append(prompt)
        all_prompt_ids = self.tokenizer(all_prompts, add_special_tokens=False)["input_ids"]
        processed_samples = []
        for prompt, prompt_id, data in zip(all_prompts, all_prompt_ids, datas):
            if self.args.rollout_max_prompt_len is not None:
                if len(prompt_id) > self.args.rollout_max_prompt_len:
                    continue
            processed_samples.append(
                Sample(
                    prompt=prompt,
                    response="",
                    prompt_ids=prompt_id,
                    data_source=data.get(self.args.datasource_key, data["data_path_info"]),
                    label=data[self.args.label_key] if self.args.label_key is not None else None,
                    status=SampleStatus.PENDING,
                    metadata=data.get(self.args.metadata_key) or {},
                )
            )

        return processed_samples

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
        data_group = []
        for _ in range(self.n_samples_per_prompt):
            sample = copy.deepcopy(data)
            sample.set_index(self.sample_index)
            self.sample_index += 1
            data_group.append(sample)
        return data_group


class EvalDataset(RolloutDataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.n_samples_per_prompt = self.args.n_samples_per_eval_prompt
        self.max_epoch = None
