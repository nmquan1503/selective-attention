from dataclasses import dataclass
import torch

class InferenceState:
    def __init__(
        self,
        lengths: torch.Tensor
    ):
        self.step = 0
        self.lengths = lengths  # (batch_size,)

    def update(self):
        self.step += 1
        self.lengths += 1