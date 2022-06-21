from typing import Tuple, Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

from modeling.genomic_model import GenomicModel
from modeling.metrics import GenomicGenerationMetrics, CRITERION
from utils.logging import get_logger

LOG = get_logger(__name__)


class GenomicGeneratorTrainer:

    def __init__(
        self,
        training_args,
        accelerator: Accelerator,
        model: GenomicModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None)):

        self.training_args = training_args
        self.accelerator = accelerator
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = CRITERION[training_args.criterion]

        self.optimizer, self.lr_scheduler = optimizers
        self.training_steps = 0
        self.metrics = GenomicGenerationMetrics(prefix="train_")
        self.val_metrics = GenomicGenerationMetrics(prefix="val_")
        self.wandb = has_wandb and self.training_args.use_wandb

    def create_optimizer(self):
        # TODO: Add a more fancy parameter selection
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "lr": self.training_args.learning_rate
        }
        self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)

    def create_lr_scheduler(self):
        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer,
                                       max_lr=self.training_args.learning_rate,
                                       epochs=self.training_args.num_epochs,
                                       steps_per_epoch=len(
                                           self.train_dataloader))

    def log(self, metrics: Dict):
        if self.wandb:
            wandb.log(metrics)
        else:
            for k, v in metrics.items():
                self.accelerator.print(f"{k}: {v}")

    def train(self):
        if self.optimizer is None:
            self.create_optimizer()
        if self.lr_scheduler is None:
            self.create_lr_scheduler()
        epochs_trained = 0
        LOG.info("Starting training")