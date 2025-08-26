from slime.utils.types import Sample
from slime.data.dataset import Dataset, read_file
from slime.data.templates import get_chat_template


def convert_rm_samples_to_train(args, samples: list[Sample]):
    """
    Convert rm samples to training data.
    output_samples token is: [ prompt_ids + chosen_resp_ids + prompt_ids + rejected_resp_ids]
    response part is: [ chosen_resp_ids + prompt_ids + rejected_resp_ids]
    the sub_samples_idx is specially for data packing cu_seq_len
    loss_mask is [[1] * len(chosen_resp_ids) + [0] * len(prompt_ids) + [1] * len(rejected_resp_ids)]
    """

    train_data = {
        "tokens": [],
        "response_lengths": [],
        "loss_masks": [],
        "sub_samples_idx": [[0, len(sample["chosen_ids"])] for sample in samples],
    }
    for sample in samples:
        cho_len, rej_len, prompt_len = len(sample["chosen_ids"]), len(sample["rejected_ids"]), sample["prompt_len"]
        train_data["tokens"].append(sample["chosen_ids"] + sample["rejected_ids"])
        train_data["response_lengths"].append(cho_len - prompt_len + rej_len)
        train_data["loss_masks"].append([1] * (cho_len - prompt_len) + [0] * prompt_len + [1] * (rej_len - prompt_len))
    return train_data


class RewardDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.init_dataset()

    def init_dataset(self):
        self.origin_samples = []
        """
        messages format is [
            {"role": "user", "content": "Hello"}, 
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "chosen": "I'm fine, thank you! And you?", "rejected": "Boring!"},
        ]
        """
        for name, data_path in self.data_path_info.items():
            for data in read_file(data_path):
                # TODO: this is slow. refactor to multiprocess
                raw_datas = data[self.args.input_key]
                if self.args.chat_template:
                    chat_template = get_chat_template(self.args.chat_template)
                    chosen_ids, rejected_ids, prompt_len = chat_template(raw_datas, self.tokenizer)
                else:
                    if self.args.tool_key is not None:
                        tools = data[self.args.tool_key]
                    else:
                        tools = None
                    # TODO: This is kind of stupid, find a better tokenizer ids generation method
                    cho_mes = raw_datas[:-1] + [{"role": "assistant", "content": raw_datas[-1]["chosen"]}]
                    rej_mes = raw_datas[:-1] + [{"role": "assistant", "content": raw_datas[-1]["rejected"]}]
                    chosen_ids = self.tokenizer.apply_chat_template(cho_mes, tools, tokenize=True, return_tensors="pt")
                    rejected_ids = self.tokenizer.apply_chat_template(
                        rej_mes, tools, tokenize=True, return_tensors="pt"
                    )
                    prompt_len = len(
                        self.tokenizer.apply_chat_template(
                            raw_datas[:-1], tools, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                        )
                    )

                # TODO: 换成total len
                if self.args.rollout_max_prompt_len is not None:
                    if len(chosen_ids) + len(rejected_ids) > self.args.rollout_max_prompt_len:
                        continue

                self.origin_samples.append(
                    Sample(
                        chosen_ids=chosen_ids,
                        rejected_ids=rejected_ids,
                        prompt_len=prompt_len,
                        data_source=data.get(self.args.datasource_key, name),
                        raw_messages=raw_datas,
                        metadata=data.get(self.args.metadata_key) or {},
                    )
                )
        self.samples = self.origin_samples
        if self.args.shuffle_dataset:
            self.shuffle(self.epoch_id)
