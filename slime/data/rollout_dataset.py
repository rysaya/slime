import copy
from slime.data.dataset import Dataset, read_file
from slime.utils.types import Sample, SampleStatus


class RolloutDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.n_samples_per_prompt = self.args.n_samples_per_prompt
        self.init_dataset()

    def init_dataset(self):
        self.origin_samples = []
        for name, data_path in self.data_path_info.items():
            for data in read_file(data_path):
                # TODO: this is slow. refactor to multiprocess
                prompt = data[self.args.input_key]
                if self.args.apply_chat_template:
                    if self.args.tool_key is not None:
                        tools = data[self.args.tool_key]
                    else:
                        tools = None
                    prompt = self.tokenizer.apply_chat_template(
                        prompt, tools, tokenize=False, add_generation_prompt=True
                    )

                if self.args.rollout_max_prompt_len is not None:
                    if len(self.tokenizer(prompt)["input_ids"]) > self.args.rollout_max_prompt_len:
                        continue

                self.origin_samples.append(
                    Sample(
                        prompt=prompt,
                        response="",
                        data_source=data.get(self.args.datasource_key, name),
                        label=data[self.args.label_key] if self.args.label_key is not None else None,
                        status=SampleStatus.PENDING,
                        metadata=data.get(self.args.metadata_key) or {},
                    )
                )
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
            self.sample_offset == 0
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
        self.max_epoch = 1
