import copy
import json

import numpy as np
import torch

from slime.data.dataset import Dataset, load_and_encode_image
from slime.data.templates import get_chat_template
from slime.utils.misc import load_function
from slime.utils.types import Sample, SampleStatus


def convert_rl_samples_to_train(args, samples: list[Sample]):
    """
    Convert inference generated samples to training data.
    """
    train_data = {
        "tokens": [sample["tokens"] for sample in samples],
        "response_lengths": [sample["response_length"] for sample in samples],
        "rewards": [sample["reward"] for sample in samples],
        "raw_reward": [sample["reward"] for sample in samples],
        "truncated": [1 if sample["status"] == SampleStatus.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample["index"] for sample in samples],
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

    for k in samples[0].keys():
        if "reward" in k and k != "reward":
            train_data[k] = [sample[k] for sample in samples]

    if args.custom_reward_post_process_path is not None:
        custom_reward_post_process_func = load_function(args.custom_reward_post_process_path)
        train_data["rewards"] = custom_reward_post_process_func(rewards)
    elif args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
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


class RolloutDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.n_samples_per_prompt = self.args.n_samples_per_prompt
        self.sample_group_index = 0
        self.init_dataset()

    def process_datas(self, datas):
        all_prompts = []
        all_image_datas = []
        for data in datas:
            image_data = []
            if self.args.multimodal_keys:
                prompt_content = []
                if self.args.input_key in data:
                    prompt_content.append({"type": "text", "text": data[self.args.input_key]})
                for media_type, data_key in self.args.multimodal_keys.items():
                    if data_key in data:
                        media_path = data[data_key]
                        prompt_content.append({"type": media_type, "path": media_path})
            else:
                prompt_content = data[self.args.input_key]
            if self.args.chat_template:
                chat_template = get_chat_template(self.args.chat_template)
                prompt = chat_template(prompt_content, self.tokenizer)
                all_prompts.append(prompt)
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
                template_input = (
                    [{"role": "user", "content": prompt_content}] if self.args.multimodal_keys else prompt_content
                )
                prompt = self.tokenizer.apply_chat_template(
                    template_input, tools, tokenize=False, add_generation_prompt=True
                )
                # multimodal prompt
                if not isinstance(prompt, str):
                    text_prompt = ""
                    image_token = self.tokenizer.special_tokens_map.get("image_token", "<image>")
                    failed = False
                    for part in prompt:
                        if part["type"] == "text":
                            text_prompt += part["text"]
                        elif part["type"] == "image":
                            text_prompt += image_token
                            try:
                                img_b64 = load_and_encode_image(part["path"])
                                image_data.append(img_b64)
                            except Exception as e:
                                print(f"Error processing image {part['path']}: {e}")
                                failed = True
                                break
                    if failed:
                        continue
                all_prompts.append(prompt)
                all_image_datas.append(image_data)
        all_prompt_ids = self.tokenizer(all_prompts, add_special_tokens=False)["input_ids"]
        processed_samples = []
        for prompt, prompt_id, image_data, data in zip(all_prompts, all_prompt_ids, all_image_datas, datas):
            if self.args.rollout_max_prompt_len is not None and not self.args.multimodal_keys:
                if len(prompt_id) > self.args.rollout_max_prompt_len:
                    continue
            processed_samples.append(
                Sample(
                    prompt=prompt,
                    response="",
                    tokens=prompt_id,
                    image_data=image_data,
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
        self.sample_group_index += 1
        data_group = []
        for _ in range(self.n_samples_per_prompt):
            sample = copy.deepcopy(data)
            sample.set_index(self.sample_index)
            sample["sample_group_index"] = self.sample_group_index
            self.sample_index += 1
            data_group.append(sample)
        return data_group


class EvalDataset(RolloutDataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.n_samples_per_prompt = self.args.n_samples_per_eval_prompt
        self.max_epoch = None
