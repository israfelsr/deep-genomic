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
from modeling.generator import Generator
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
    parser.add_argument("--c_norm",
                        action="store_true",
                        help="If passed, will normalize the conditions")
    # Model Parameters
    parser.add_argument("--num_classes",
                        type=int,
                        required=True,
                        default=1,
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
    parser.add_argument("--eval_every",
                        type=int,
                        default=5,
                        help="Epochs interval to eval.")
    # Runtime
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="If passed, will evaluate while training")
    parser.add_argument("--do_encode",
                        action="store_true",
                        help="If passed, will encode the genome")
    parser.add_argument("--compute_r2",
                        action="store_true",
                        help="If passed, will compute the offset")
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
        LOG.info(f"Experiment will be saved in {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

    if args.use_wandb:
        if has_wandb:
            wandb.init(project="deep-genomic",
                       name=args.wandb_run_name,
                       config=args)

    if args.condition_files:
        if not args.is_conditional:
            LOG.warning(
                "Conditions were passed but the model is not conditioned")
        else:
            LOG.info(f'Using conditions from {args.condition_files}')
    else:
        raise ValueError("No conditions have been passed")

    LOG.info(f'Loading genomic data from: {args.data_dir}')
    x, c = load_data("genome.csv", args.condition_files, args.data_dir,
                     args.c_norm)

    x_dim = x.shape[1]
    c_dim = c.shape[1]
    LOG.info(f"Working with x_dim = {x_dim}")

    use_context = True if args.c_embedded else False
    if args.c_embedded is None:
        args.c_embedded = c_dim
    if use_context:
        assert args.is_conditional

    x_val, x_train, c_val, c_train = split_dataset(x, c)
    LOG.info(f"Train dataset size: {len(x_train)}")
    LOG.info(f"Validation dataset size: {len(x_val)}")
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
                                model=args.model,
                                use_context=use_context)

    model = build_model(config)
    model.apply(init_weights)

    accelerator = Accelerator()

    trainer = GenomicGeneratorTrainer(training_args=args,
                                      accelerator=accelerator,
                                      model=model,
                                      train_dataloader=train_dataloader,
                                      val_dataloader=val_dataloader)

    trainer.train()

    if args.use_wandb:
        model.save(wandb.run.dir)
        LOG.info(f"Model Saved into wandb")

    if args.output_dir:
        model.save(output_folder)
        LOG.info(f"Model Saved into {output_folder}")

    generator = Generator(model, args.condition_files, args.data_dir)
    if args.do_encode:
        mu, var = generator.encode(x, use_context)
        if has_wandb and args.use_wandb:
            LOG.info(f'Saving latent space in wandb')
            columns = []
            columns += [f"mu_{i}" for i in range(args.z_dim)]
            columns += [f"var_{i}" for i in range(args.z_dim)]
            columns += [f"c_{i}" for i in range(c_dim)]
            data = np.concatenate((mu, var, c), axis=1)
            table = wandb.Table(columns=columns, data=data)
            wandb.log({
                "latent_space": table,
            })
        elif args.output_dir:
            LOG.info(f'Saving latent space variable in {output_folder}')
            np.save(os.path.join(output_folder, "mu"), mu)
            np.save(os.path.join(output_folder, "var"), var)
            np.save(os.path.join(output_folder, "c"), c)
        else:
            LOG.warning('The encode was done but no data has been saved')

    if args.compute_r2:
        r2, offset, fitness_offset, predicted_fitness, rec, gen = generator.compute_r2(
        )
        r2q, offsetq, fitness_offsetq, predicted_fitnessq, _, _ = generator.compute_r2(
            qtls=True)
        if args.output_dir:
            generator.offset_data.to_csv(args.output_dir)
        if has_wandb and args.use_wandb:
            generator.offset_data.to_csv(wandb.run.dir)
            #    data = np.concatenate((offset, fitness_offset, predicted_fitness),
            #                         axis=1)
            #    table = wandb.Table(
            #        columns=["offset", "fitness_offset", "predicted_fitness"],
            #        data=data)
            #    dataq = np.concatenate(
            #        (offsetq, fitness_offsetq, predicted_fitnessq), axis=1)
            #    tableq = wandb.Table(columns=[
            #        "offset_qts", "fitness_offset_qts", "predicted_fitness_qts"
            #    ],
            #                         data=dataq)
            wandb.log({
                "r2": r2,
                "r2_qtls": r2q,
                #        "genomic_offset": table,
                #        "qtls_offset": tableq,
            })
        #    np.save(os.path.join(wandb.run.dir, "rec"), rec)
        #    np.save(os.path.join(wandb.run.dir, "gen"), gen)
        else:
            LOG.warning('The R2 value was computed no data has been saved')


if __name__ == '__main__':
    main()
