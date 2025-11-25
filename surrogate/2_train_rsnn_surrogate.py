#!/usr/bin/env python3
"""
train_rsnn.py

Example usage:

python train_rsnn.py --data ./data/dataset.ds --max-epochs 10 --batch-size 2048

Use --help to see all options.
"""

import argparse
import os
import logging
import torch
from torch.utils.data import TensorDataset

import snntorch as snn
import numpy as np
import random
import ast
import re
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from typing import Callable, Union, Optional

from utils.RSNN import SpikeSynth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_optimizer_class(name):
    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    return optimizer_map.get(name.lower(), torch.optim.AdamW)  # default fallback


def parse_optimizer_kwargs(kwargs_str):
    """Parse comma-separated key=value pairs into a dict, safely handling tuples and scientific notation."""
    kwargs = {}
    if not kwargs_str:
        return kwargs

    # Split on commas that are NOT inside parentheses
    parts = re.split(r',(?![^(]*\))', kwargs_str)

    for kv in parts:
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        key, value = key.strip(), value.strip()

        # Try to safely evaluate Python literals (numbers, tuples, lists, bools)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback: handle booleans or keep as string
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            else:
                value = value

        kwargs[key] = value

    return kwargs


def parse_scheduler_args(scheduler_name, scheduler_kwargs_str):
    scheduler_map = {
        "none": None,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_class = scheduler_map.get(scheduler_name.lower(), None)

    kwargs = {}
    if scheduler_kwargs_str:
        for kv in scheduler_kwargs_str.split(","):
            if "=" in kv:
                key, value = kv.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    # safely evaluate Python literals (numbers, tuples, bool, None)
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                kwargs[key] = value

    return scheduler_class, kwargs

def main(args):
    logger.info("Loading dataset from %s", args.data)
    data = torch.load(args.data)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_valid, Y_valid = data["X_valid"], data["Y_valid"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, Y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # Logging directory
    logging_directory = os.path.abspath(args.logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory

    # Surrogate gradient
    if args.surrogate == "atan":
        surrogate = snn.surrogate.atan()
    elif args.surrogate == "fast_sigmoid":
        surrogate = snn.surrogate.fast_sigmoid()
    else:
        logger.warning("Unknown surrogate '%s', falling back to atan()", args.surrogate)
        surrogate = snn.surrogate.atan()

    # Scheduler and optimizer
    scheduler_class, scheduler_kwargs = parse_scheduler_args(args.scheduler_class, args.scheduler_kwargs)
    optimizer_class = get_optimizer_class(args.optimizer_class)
    optimizer_kwargs = parse_optimizer_kwargs(args.optimizer_kwargs)
    SRC_config = parse_optimizer_kwargs(args.src_config)

    loss_kwargs = parse_optimizer_kwargs(args.loss_kwargs)

    # Checkpoint callback
    os.makedirs(args.checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f"{args.experiment_name}-run{{run_idx}}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor=args.monitor,
        mode=args.monitor_mode,
    )

    accelerator = "gpu" if torch.cuda.is_available() and args.use_gpu_if_available else "cpu"
    logger.info("Using accelerator=%s", accelerator)

    # Run loop for multiple seeds
    for run_idx in range(args.num_runs):
        seed = args.base_seed + run_idx
        logger.info("=== Starting run %d/%d with seed %d ===", run_idx + 1, args.num_runs, seed)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Update experiment name for this run
        run_name = f"{args.experiment_name}_run{run_idx+1}"

        # WandB logger
        wandb_logger = None
        if not args.no_wandb:
            wandb_logger = WandbLogger(log_model=True, project=args.project_name, name=run_name, save_dir=logging_directory)

        callbacks = [checkpoint_callback]
        if args.early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor=args.monitor,
                mode=args.monitor_mode,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_delta,
                verbose=True,
            )
            callbacks.append(early_stopping_callback)


        # Instantiate model
        model = SpikeSynth(
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            beta=args.beta,
            lr=args.lr,
            num_hidden=args.num_hidden,
            batch_size=args.batch_size,
            num_hidden_layers=args.num_hidden_layers,
            dropout=0, #args.dropout,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            max_epochs=args.max_epochs,
            surrogate_gradient=surrogate,
            temporal_skip=args.temporal_skip,
            layer_skip=args.layer_skip,
            use_bntt=args.use_bntt,
            neuron_type=args.neuron_type,
            bntt_time_steps=args.bntt_time_steps,
            log_every_n_steps=args.log_every_n_steps,
            use_layernorm=args.use_layernorm,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            SRC_config=SRC_config,
            loss_fn=args.loss_fn,
            loss_kwargs=loss_kwargs,
        )

        # Optional torch.compile
        if args.torch_compile:
            try:
                logger.info("Attempting torch.compile(model)")
                model = torch.compile(model)
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

        # Trainer
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            logger=wandb_logger if not args.no_wandb else None,
            callbacks=callbacks,
            log_every_n_steps=args.log_every_n_steps,
        )

        # Train
        trainer.fit(model)

        logger.info("Starting test for run %d using best checkpoint", run_idx + 1)
        trainer.test(
            model=model,  # optional, will load checkpoint automatically if ckpt_path="best"
            ckpt_path="best"
        )

        # Finalize WandB logger
        if wandb_logger and not args.no_wandb:
            try:
                wandb_logger.finalize("success")
                wandb.finish()
            except Exception as e:
                logger.warning("wandb_logger.finalize() failed: %s", e)

        logger.info("Run %d finished. Best checkpoint: %s", run_idx + 1, checkpoint_callback.best_model_path or "N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpikeSynth RSNN (converted from notebook)")

    # General and Monitoring
    parser.add_argument("--data", type=str, default="./data/dataset.ds", help="Path to dataset (torch file).")
    parser.add_argument("--experiment-name", type=str, default="test", help="WandB experiment/run name.")
    parser.add_argument("--project-name", type=str, default="Spike-Synth-Surrogate", help="WandB project name.")
    parser.add_argument("--logging-directory", type=str, default=".temp", help="Local directory where logs/wandb files are stored.")
    parser.add_argument("--checkpoint-path", type=str, default="models/SRNN", help="Directory to save checkpoints.")
    parser.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor for checkpointing.")
    parser.add_argument("--monitor-mode", dest="monitor_mode", choices=["min", "max"], default="min", help="Monitor mode for checkpointing/early stopping.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Logging frequency (trainer.log_every_n_steps)")

    # Multi-run / seeding
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent runs with different seeds for statistics.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for reproducibility. Each run will increment from this.")

    # Model hyperparameters
    parser.add_argument("--num-hidden", type=int, default=256, help="Number of hidden units.")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("--beta", type=float, default=0.9, help="Beta decay Value. Has no impact when SRC is used as neuron type, then instead configure src-config.")
    parser.add_argument("--temporal-skip", type=int, default=-1, help="Temporal skip value (or None).")
    parser.add_argument("--layer-skip", type=int, default=2, help="Layer skip value.")
    parser.add_argument("--surrogate", type=str, default="atan", help="Surrogate gradient to use (e.g. 'atan').")
    # parser.add_argument("--dropout", type=float, default=0, help="Introduces a dropout layer for each LIF layer.")
    parser.add_argument("--torch-compile", action="store_true", help="Attempt torch.compile(model) before training.")
    parser.add_argument("--use-gpu-if-available", action="store_true", help="Use GPU if available (default: off).")
    parser.add_argument("--use-bntt", type=str2bool, default=False, help="Whether to use Batchnorm or not. Batchnorm is applied for LeakyParallel after the Spike, which doesnt work well and for Leaky and RLeak neuron directly to the input (WX) and for SLSTM to the input.")
    parser.add_argument("--bntt-time-steps", type=int, default=100, help="Batchnorm needs to know the sequence length beforehand.")
    parser.add_argument("--use-layernorm", type=str2bool, default=False, help="Whether to use Layernorm or not.")
    parser.add_argument("--neuron-type", type=str, choices=["LeakyParallel", "Leaky", "RLeaky", "SLSTM", "SRC"], default="LeakyParallel", help="Which type of spiking neuron to use for the spiking layers.")
    parser.add_argument("--src-config",type=str,default="",help=(
        "SRC configuration as key=value pairs separated by commas, e.g. "
        "'alpha=0.9,rho=3.0,r=6.0,rs=8.0,bh_init=-6.0,bh_max=-4.0'. "
        "These will be passed directly to the SRC constructor."),
        )

    # Training hyperparameters
    parser.add_argument("--max-epochs", type=int, default=10, help="Max number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size (passed to SpikeSynth).")
    parser.add_argument("--scheduler-class", type=str, default="cosine", choices=["none", "cosine", "exponential", "step", "plateau"], help="Learning rate scheduler type. Options: none, cosine, exponential, step, plateau.")
    parser.add_argument("--scheduler-kwargs", type=str, default="", help=(
        "Extra scheduler arguments as key=value pairs separated by commas, e.g. "
        "'gamma=0.95,T_max=50,step_size=10'. Ignored if scheduler=none."
    ),)
    parser.add_argument("--optimizer-class",type=str,default="AdamW",choices=["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"],help="Optimizer to use. Options: Adam, AdamW, SGD, RMSprop, Adagrad.")
    parser.add_argument("--optimizer-kwargs",type=str,default="",help="Extra optimizer arguments as key=value pairs, e.g. 'betas=(0.9,0.999),eps=1e-8,weight_decay=0.01'.",)
    parser.add_argument("--early-stopping", type=str2bool, default=True, help="Enable early stopping.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Number of epochs with no improvement after which training stops.")
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4, help="Minimum change in monitored quantity to qualify as improvement.")
    parser.add_argument("--loss-fn", type=str, default="mse", help="Loss function to use; e.g. 'mse','mae','huber','logcosh','pearson'.")
    parser.add_argument("--loss-kwargs", type=str, default="", help="Optional loss kwargs as key=value pairs separated by commas, e.g. 'delta=0.5' for Huber.")

    args = parser.parse_args()

    main(args)