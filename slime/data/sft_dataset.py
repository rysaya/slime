from typing import Union
from slime.data.dataset import Dataset, read_file
from slime.utils.types import Sample
from slime.data.templates import get_chat_template


def convert_sft_samples_to_train(args, samples: Union[list[Sample], list[list[Sample]]]):
    """
    Convert sft samples to training data.
    """
    train_data = {
        "tokens": [sample["tokens"] for sample in samples],
        "response_lengths": [sample["response_length"] for sample in samples],
        "loss_masks": [sample["loss_mask"] for sample in samples],
    }
    return train_data


class SFTDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.init_dataset()

    def init_dataset(self):
        self.origin_samples = []
        for name, data_path in self.data_path_info.items():
            for data in read_file(data_path):
                # TODO: this is slow. refactor to multiprocess
                raw_datas = data[self.args.input_key]
                if self.args.chat_template:
                    chat_template = get_chat_template(self.args.chat_template)
                    input_datas = chat_template(raw_datas, self.tokenizer)
                else:
                    if self.args.tool_key is not None:
                        tools = data[self.args.tool_key]
                    else:
                        tools = None
                    input_datas = self.tokenizer.apply_chat_template(raw_datas, tools, tokenize=False)

                # TODO: 换成total len
                input_ids = self.tokenizer(input_datas)["input_ids"]
                if self.args.rollout_max_prompt_len is not None:
                    if len(input_ids) > self.args.rollout_max_prompt_len:
                        continue

                self.origin_samples.append(
                    Sample(
                        input_ids=input_ids,
                        data_source=data.get(self.args.datasource_key, name),
                        raw_messages=raw_datas,
                        metadata=data.get(self.args.metadata_key) or {},
                    )
                )
        self.samples = self.origin_samples
        if self.args.shuffle_dataset:
            self.shuffle(self.epoch_id)
