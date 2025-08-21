from typing import Union
from slime.utils.types import Sample
from slime.data.dataset import Dataset


def convert_rm_samples_to_train(args, samples: Union[list[Sample], list[list[Sample]]]):
    pass


class RewardDataset(Dataset):
    def __init__(self, args, path):
        super().__init__(args, path)
