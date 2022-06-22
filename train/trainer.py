from typing import Tuple, Dict
from tqdm import tqdm

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
        epochs_trained = 1
        LOG.info("Starting training")

        (self.model, self.optimizer,
         self.train_dataloader) = self.accelerator.prepare(
             self.model, self.optimizer, self.train_dataloader)
        self.metrics.call_prepare(self.accelerator)

        for epoch in range(epochs_trained, self.training_args.num_epochs + 1):
            self.model.train()
            for batch_i, (inputs, conditions) in enumerate(
                    tqdm(self.train_dataloader, desc=f"Epoch {epoch} batch"),
                    1):
                self.optimizer.zero_grad()
                outputs = self.model(inputs, conditions)
                loss = self.criterion(outputs)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.metrics.update(outputs, self.criterion)
                self.training_steps += 1
            evaluation_metrics = {}
            if (self.training_args.do_eval
                    and (epoch % self.training_args.eval_every == 0
                         or epoch == self.training_args.num_epochs)):
                evaluation_metrics = self.evaluate()
            self.log({
                "epoch": epoch,
                "step": self.training_steps,
                **self.metrics.compute_and_reset(),
                **evaluation_metrics,
            })

    def evaluate(self):
        LOG.info("Starting evaluation")
        self.model.to(self.accelerator.device)
        (self.model, self.val_dataloader) = self.accelerator.prepare(
            self.model, self.val_dataloader)
        self.val_metrics.call_prepare(self.accelerator)

        self.model.eval()
        for (inputs, conditions) in tqdm(self.val_dataloader,
                                         desc="Evaluating batch:"):

            with torch.no_grad():
                outputs = self.model(inputs, conditions)
            self.val_metrics.update(outputs, self.criterion)
        return self.val_metrics.compute_and_reset()