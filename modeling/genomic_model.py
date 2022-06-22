import dataclasses
from dataclasses import dataclass
from typing import Dict, List
import os
import json
import enum

import torch
import torch.nn as nn

from utils.logging import get_logger

# TODO: Eventually we would like to create several architecures for each part
LOG = get_logger(__name__)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Models(str, enum.Enum):
    SIMPLE_VAE = 'simple_vae'
    PRIOR_VAE = 'prior_vae'

    def __str__(self):
        return self.value


@dataclass
class GenomicModelConfig:
    num_classes: int
    x_dim: int
    encoder_dims: List
    decoder_dims: List
    c_dim: int
    z_dim: int
    conditional: bool
    c_embedded: int
    model: Models
    use_context: bool

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)


class GenomicModel(nn.Module):

    def __init__(self, config: GenomicModelConfig):
        super().__init__()
        self.config = config

    def save(self, output_dir):
        weights = self.state_dict()
        torch.save(weights, os.path.join(output_dir, "pytorch_model.bin"))
        # save config
        with open(os.path.join(output_dir, "config.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(self.config.as_dict(),
                      f,
                      sort_keys=True,
                      separators=(",", ": "),
                      ensure_ascii=False,
                      indent=4)

    @classmethod
    def from_pretrained(cls, model_path: str):
        LOG.info("Loading model from {model_path}")
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"{config_path} cannot be found. Please check and try again")
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"{weights_path} cannot be found. Please check and try again")
        with open(config_path, 'r', encoding="utf-8") as f:
            config_json = json.load(f)
            config = GenomicModelConfig.from_dict(config_json)
        LOG.info(f"Resolved config from {config_path}")
        model = cls(config=config)
        model.load_state_dict(state_dict=torch.load(
            weights_path, map_location=torch.device('cpu')))
        LOG.info(f"Loaded weights from {weights_path}")
        return model
