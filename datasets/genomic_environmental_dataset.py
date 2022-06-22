import os
import pandas as pd
import numpy as np
from typing import List
from torch.utils.data import Dataset
import torch

from datasets.utils import drop_unnamed, haploidization, normalization
from utils.logging import get_logger

LOG = get_logger(__name__)


def load_conditions(conditions: List[str], data_dir, norm=False):
    if len(conditions) == 1:
        c = drop_unnamed(pd.read_csv(os.path.join(data_dir,
                                                  conditions[0]))).to_numpy()
        if norm:
            c = normalization(c)
    else:
        c = []
        for file in conditions:
            c.append(
                drop_unnamed(pd.read_csv(os.path.join(data_dir,
                                                      file))).to_numpy())
        c = np.concatenate(c, axis=1)
    return c


def load_data(inputs: str,
              conditions: List[str],
              data_dir,
              qtls=False,
              c_norm=False):
    x = drop_unnamed(pd.read_csv(os.path.join(data_dir, inputs))).to_numpy()
    c = load_conditions(conditions, data_dir, c_norm)
    if qtls:
        m2 = drop_unnamed(pd.read_csv(os.path.join(
            data_dir, "mutationm2.csv"))).to_numpy().squeeze()
        m3 = drop_unnamed(pd.read_csv(os.path.join(
            data_dir, "mutationm3.csv"))).to_numpy().squeeze()
        idx = np.concatenate((m2, m3))
        x = x.T[idx].T
    assert len(x) == len(c)
    return x, c


class GenomicEnvironmentalDataset(Dataset):

    def __init__(self, x, c, do_haploidization: bool = True):
        super().__init__()
        if do_haploidization:
            x = haploidization(x)
        self.x = x
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        c = self.c[idx]
        return (torch.tensor(x, dtype=torch.float),
                torch.tensor(c, dtype=torch.float))
