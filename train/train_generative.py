# Common imports
import os
import argparse
import random
import numpy as np

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Local imports
from datasets.genomic_environmental_dataset import (load_data,
                                                    GenomicEnvironmentalDataset
                                                    )
from datasets.utils import split_dataset
from modeling.genomic_model import (GenomicModelConfig, Models, init_weights)
from modeling.metrics import CRITERION
from modeling.models import build_model
from train.trainer import GenomicGeneratorTrainer
from utils.logging import get_logger

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

LOG = get_logger(__name__)


def set_seed(seed):
    """Sets a seed in all available libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Genomic Generator Training")
    # Directories
    parser.add_argument("--data_dir",
                        type=str,
                        required=True,
                        help="Path to the data directory")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Path to the output directory")
    parser.add_argument("--condition_files",
                        nargs="*",
                        type=str,
                        default=["var_current.csv", "pop.csv"])
    # Model Parameters
    parser.add_argument("--num_classes",
                        type=int,
                        required=True,
                        help="Possible number of classes in locus")
    parser.add_argument("--z_dim",
                        required=True,
                        type=int,
                        help="Latent space size")
    parser.add_argument("--c_embedded",
                        type=int,
                        default=None,
                        help="Conditional embedded size")
    parser.add_argument("--encoder_dims",
                        nargs="*",
                        type=int,
                        default=[512, 256])
    parser.add_argument("--decoder_dims",
                        nargs="*",
                        type=int,
                        default=[256, 512])
    parser.add_argument("--is_conditional",
                        action="store_true",
                        help="If passed will create a conditioned model")
    parser.add_argument("--model",
                        required=True,
                        type=Models,
                        help="Model to run")
    # Training Parameters
    parser.add_argument("--do_haploidization",
                        action="store_true",
                        help="If passed, will do haploidization")
    parser.add_argument("--batch_size",
                        type=int,
                        default=100,
                        help="Train batch size")
    parser.add_argument("--criterion",
                        type=str,
                        default="",
                        required=True,
                        choices=list(CRITERION.keys()),
                        help="Criterion to use.")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=2,
                        help="Number of epoch to train")
    # Runtime
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--use_wandb",
                        action="store_true",
                        help="If passed, will log to wandb")
    parser.add_argument("--wandb_run_name",
                        type=str,
                        help="Set name of run and output folder")

    args = parser.parse_args()

    # Runtime
    set_seed(args.seed)
    if args.output_dir:
        output_folder = os.path.join(args.output_dir, args.wandb_run_name)
        os.makedirs(output_folder, exist_ok=True)

    if args.use_wandb:
        if has_wandb:
            wandb.init(project="genomic-generation",
                       name=args.wandb_run_name,
                       config=args)

    if args.condition_files:
        use_conditions = True
        if not args.is_conditional:
            LOG.warning(
                "Conditions were passed but the model is not conditioned")
        LOG.info(f'Using conditions from {args.condition_files}')
    else:
        use_conditions = False
        assert not args.is_conditional
        LOG.warning('No condition has been passed')

    LOG.info(f'Loading genomic data from: {args.data_dir}')
    x, c = load_data("genome.csv", args.condition_files, args.data_dir,
                     use_conditions)

    x_dim = x.shape[1]
    c_dim = c.shape[1] if use_conditions else None

    if args.c_embedded is None:
        args.c_embedded = c_dim

    x_val, x_train, c_val, c_train = split_dataset(x, c, use_conditions)

    train_dataset = GenomicEnvironmentalDataset(x_train, c_train,
                                                args.do_haploidization)
    val_dataset = GenomicEnvironmentalDataset(x_val, c_val,
                                              args.do_haploidization)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

    config = GenomicModelConfig(num_classes=args.num_classes,
                                x_dim=x_dim,
                                conditional=args.is_conditional,
                                c_dim=c_dim,
                                z_dim=args.z_dim,
                                c_embedded=args.c_embedded,
                                encoder_dims=args.encoder_dims,
                                decoder_dims=args.decoder_dims,
                                model=args.model)

    model = build_model(config)
    model.apply(init_weights)

    accelerator = Accelerator()

    trainer = GenomicGeneratorTrainer(training_args=args,
                                      accelerator=accelerator,
                                      model=model,
                                      train_dataloader=train_dataloader,
                                      val_dataloader=val_dataloader)

    trainer.train()


if __name__ == '__main__':
    main()
