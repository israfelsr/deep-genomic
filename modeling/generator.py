import os
import numpy as np
import pandas as pd
from typing import List

import torch
from sklearn.linear_model import LinearRegression

from modeling.genomic_model import GenomicModel
from datasets.genomic_environmental_dataset import load_conditions
from datasets.utils import drop_unnamed, filter_by
from utils.logging import get_logger

LOG = get_logger(__name__)


def compute_genomic_offset(x_reconstructed, x_generated):
    n_loci = x_reconstructed.shape[1]
    return np.sum(np.square(x_generated - x_reconstructed), axis=1) / n_loci


def compute_fitness_offset(fitness_current, fitness_future):
    fitness_offset = fitness_future - fitness_current
    return fitness_offset


def get_c_future_files(c_current: List[str], data_dir):
    c_future = []
    for c in c_current:
        c_new = c.replace("current", "future")
        if os.path.exists(os.path.join(data_dir, c)):
            c_future.append(c_new)
        else:
            c_future.append(c)
    return c_future


class Generator:

    def __init__(self, model: GenomicModel, c_current_files: List[str],
                 data_dir: str):
        self.model = model
        self.c_current = torch.tensor(load_conditions(c_current_files,
                                                      data_dir),
                                      dtype=torch.float)
        c_future_files = get_c_future_files(c_current_files, data_dir)
        self.c_future = torch.tensor(load_conditions(c_future_files, data_dir),
                                     dtype=torch.float)
        self.fitness_current = drop_unnamed(
            pd.read_csv(os.path.join(data_dir,
                                     "fitness_current.csv"))).to_numpy()
        self.fitness_future = drop_unnamed(
            pd.read_csv(os.path.join(data_dir,
                                     "fitness_future.csv"))).to_numpy()
        self.population = drop_unnamed(
            pd.read_csv(os.path.join(data_dir,
                                     "pop.csv"))).to_numpy().squeeze()

    def encode(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encoder(torch.tensor(x), self.c_current)
            var = torch.exp(logvar)
        return np.asarray(mu), np.asarray(var), np.asarray(self.c_current)

    def generate_from_conditions(self):
        gen_current = np.asarray(self.model.generate(self.c_current))
        gen_future = np.asarray(self.model.generate(self.c_future))
        gen_current = filter_by(gen_current, self.population)
        gen_future = filter_by(gen_future, self.population)
        return gen_current, gen_future

    def compute_r2(self):
        x_reconstructed, x_generated = self.generate_from_conditions()
        genomic_offset = compute_genomic_offset(x_reconstructed, x_generated)
        fitness_offset = compute_fitness_offset(self.fitness_current,
                                                self.fitness_future)
        fitness_offset = filter_by(fitness_offset, self.population)
        linear_model = LinearRegression().fit(genomic_offset.reshape(-1, 1),
                                              fitness_offset)
        r2 = linear_model.score(genomic_offset.reshape(-1, 1), fitness_offset)
        LOG.info(f"R2 cvae: {r2}")
        predicted_fitness = linear_model.predict(genomic_offset.reshape(-1, 1))
        return r2, genomic_offset, fitness_offset, predicted_fitness
