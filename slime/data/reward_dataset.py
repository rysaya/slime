from typing import Union
from slime.utils.types import Sample
from slime.data.dataset import Dataset, read_file
from slime.data.templates import get_chat_template
from slime.utils.mask_utils import MultiTurnLossMaskGenerator


def convert_rm_samples_to_train(args, samples: Union[list[Sample], list[list[Sample]]]):
    """
    Convert sft samples to training data.
    """
    train_data = {
        "tokens": [sample["tokens"] for sample in samples],
        "response_lengths": [sample["response_length"] for sample in samples],
        "loss_masks": [sample["loss_mask"] for sample in samples],
        "sub_samples_idx": [sample["sub_samples_idx"] for sample in samples],
    }
    return train_data


class RewardDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.mask_generator = MultiTurnLossMaskGenerator(self.tokenizer, tokenizer_type=args.loss_mask_type)
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
                    chosen, rejected = chat_template(raw_datas, self.tokenizer)
                else:
                    if self.args.tool_key is not None:
                        tools = data[self.args.tool_key]
                    else:
                        tools = None
                    # TODO: This is kind of stupid, find a better tokenizer ids generation method
                    cho_mes = raw_datas[:-1] + [{"role": "assistant", "content": raw_datas[-1]["chosen"]}]
                    rej_mes = raw_datas[:-1] + [{"role": "assistant", "content": raw_datas[-1]["rejected"]}]
                    chosen = self.tokenizer.apply_chat_template(cho_mes, tools, tokenize=False)
                    rejected = self.tokenizer.apply_chat_template(rej_mes, tools, tokenize=False)

                token_ids, loss_mask = self.mask_generator.get_loss_mask(raw_datas)

                cho, rej = self.tokenizer([chosen, rejected])
                chosen_ids, rejected_ids = cho["input_ids"], rej["input_ids"]
                # TODO: 换成total len
                if self.args.rollout_max_prompt_len is not None:
                    if len(chosen_ids) + len(rejected_ids) > self.args.rollout_max_prompt_len:
                        continue

                self.origin_samples.append(
                    Sample(
                        chosen_ids=chosen_ids,
                        rejected_ids=rejected_ids,
                        data_source=data.get(self.args.datasource_key, name),
                        raw_messages=raw_datas,
                        metadata=data.get(self.args.metadata_key) or {},
                    )
                )
        self.samples = self.origin_samples
        if self.args.shuffle_dataset:
            self.shuffle(self.epoch_id)
