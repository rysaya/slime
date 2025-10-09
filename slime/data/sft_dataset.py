from slime.data.dataset import Dataset
from slime.utils.types import Sample
from slime.data.templates import get_chat_template
from slime.utils.mask_utils import MultiTurnLossMaskGenerator


def convert_sft_samples_to_train(args, samples: list[Sample]):
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
        self.mask_generator = MultiTurnLossMaskGenerator(self.tokenizer, tokenizer_type=args.loss_mask_type)
        self.init_dataset()

    def process_datas(self, datas):
        processed_samples = []
        for data in datas:
            raw_datas = data[self.args.input_key]
            # TODO: 清理重复无用逻辑
            if self.args.chat_template:
                chat_template = get_chat_template(self.args.chat_template)
                input_datas = chat_template(raw_datas, self.tokenizer)
            else:
                if self.args.tool_key is not None:
                    tools = data[self.args.tool_key]
                else:
                    tools = None
                input_datas = self.tokenizer.apply_chat_template(raw_datas, tools, tokenize=False)

            token_ids, loss_mask = self.mask_generator.get_loss_mask(raw_datas)
            if self.args.rollout_max_prompt_len is not None:
                if len(token_ids) > self.args.rollout_max_prompt_len:
                    continue
            response_length = self.mask_generator.get_response_lengths([loss_mask])[0]

            processed_samples.append(
                Sample(
                    tokens=token_ids,
                    data_source=data.get(self.args.datasource_key, data["data_path_info"]),
                    response_length=response_length,
                    loss_mask=loss_mask[-response_length:],
                    raw_messages=raw_datas,
                    metadata=data.get(self.args.metadata_key) or {},
                )
            )
        return processed_samples
