import copy

from slime.data.data import Dataset
from transformers import AutoTokenizer


class EvalDataset:
    def __init__(self, args):
        self.args = args
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.all_eval_datasets = {}
        self.all_eval_names = []
        self.sample_index = 0
        self.sample_offset = 0
        self.current_dataset_idx = 0
        self.n_sample_per_prompt = self.args.n_samples_per_eval_prompt

        for i in range(0, len(args.eval_prompt_data), 2):
            name, path = args.eval_prompt_data[i : i + 2]

            if name not in self.all_eval_datasets:
                # TODO "name" arg is temporaly solution, should further make data_source key in data like verl did
                self.all_eval_datasets[name] = Dataset(
                    name,
                    path,
                    tokenizer=tokenizer,
                    max_length=args.rollout_max_prompt_len,
                    input_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
                    label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
                    metadata_key=args.metadata_key,
                    tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
                    apply_chat_template=args.apply_chat_template,
                )
                self.all_eval_names.append(name)

    def get_sample(self):
        dataset = self.all_eval_datasets[self.all_eval_names[self.current_dataset_idx]]
        while self.sample_offset >= len(dataset):
            self.current_dataset_idx += 1
            if self.current_dataset_idx >= len(self.all_eval_names):
                self._reset_index()
                return None
            dataset = self.all_eval_datasets[self.all_eval_names[self.current_dataset_idx]]
            self.sample_offset == 0
        data = dataset.samples[self.sample_offset]
        self.sample_offset += 1
        data_group = []
        for _ in range(self.n_sample_per_prompt):
            sample = copy.deepcopy(data)
            sample.index = self.sample_index
            self.sample_index += 1
            data_group.append(sample)
        return data_group

    def _reset_index(self):
        self.sample_offset = 0
        self.sample_index = 0
        self.current_dataset_idx = 0

    # currently no save and load for EvalDataSet
    def save(self, rollout_id):
        pass

    def load(self, rollout_id=None):
        pass
