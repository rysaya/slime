from collections import UserDict
from dataclasses import dataclass
from enum import Enum

import torch


class SampleStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


class Sample(UserDict):
    def __getitem__(self, key):
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in Sample data., Available keys: {list(self.data.keys())}")
        return self.data[key]

    def to_dict(self):
        if "status" in self:
            value = self.data.copy()
            value["status"] = self.status.value
            return value
        return self.data

    @staticmethod
    def from_dict(data: dict):
        if "status" in data:
            data["status"] = SampleStatus(data["status"])
        return Sample(**data)

    def set_rollout_id(self, rollout_id):
        self["rollout_id"] = rollout_id

    def set_index(self, sample_id):
        self["index"] = sample_id

    def set_status(self, status: SampleStatus):
        self["status"] = status

    def get_metadata(self, key, default=None):
        if "metadata" not in self:
            self["metadata"] = {}
        return self["metadata"].get(key, default)

    def add_metadata(self, key, value):
        if self.get("metadata", None) is None:
            self["metadata"] = {}
        self["metadata"][key] = value


@dataclass
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int
