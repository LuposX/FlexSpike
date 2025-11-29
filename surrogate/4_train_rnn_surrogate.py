#!/usr/bin/env python3
"""
train_rnn.py

Example usage:

python train_rnn.py --data ./data/dataset.ds --max-epochs 10 --batch-size 2048

Use --help to see all options.
"""

import argparse
import os
import logging
import torch
import numpy as np
import random
import ast
import re

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

from typing import Dict, Any, Optional

from utils.RNN import RNNLightning

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


def get_optimizer_class(name: str):
    if name is None:
        return torch.optim.AdamW
    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    return optimizer_map.get(name.lower(), torch.optim.AdamW)  # default fallback


def parse_optimizer_kwargs(kwargs_str: str) -> Dict[str, Any]:
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
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            else:
                # keep string as-is
                value = value

        kwargs[key] = value

    return kwargs


def parse_scheduler_args(scheduler_name: str, scheduler_kwargs_str: str):
    scheduler_map = {
        "none": None,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_class = None
    if scheduler_name:
        scheduler_class = scheduler_map.get(scheduler_name.lower(), None)

    kwargs = {}
    if scheduler_kwargs_str:
        parts = re.split(r',(?![^(]*\))', scheduler_kwargs_str)
        for kv in parts:
            if "=" in kv:
                key, value = kv.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # keep as string
                    pass
                kwargs[key] = value

    return scheduler_class, kwargs


def main(args):
    logger.info("Loading dataset from %s", args.data)
    data = torch.load(args.data)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_valid, Y_valid = data["X_valid"], data["Y_valid"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    # infer num_inputs & seq_len from data if not provided
    if getattr(args, "num_inputs", None) is None or args.num_inputs <= 0:
        # expect shape (N, seq_len, num_inputs) or (N, seq_len)
        if X_train.ndim == 3:
            inferred_num_inputs = X_train.shape[2]
        elif X_train.ndim == 2:
            inferred_num_inputs = 1
            X_train = X_train.unsqueeze(-1)
            X_valid = X_valid.unsqueeze(-1)
            X_test = X_test.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected X_train shape: {X_train.shape}")
        args.num_inputs = inferred_num_inputs
        logger.info("Inferred num_inputs=%d from data", args.num_inputs)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, Y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # Logging directory
    logging_directory = os.path.abspath(args.logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory

    # Scheduler and optimizer parsing
    scheduler_class, scheduler_kwargs = parse_scheduler_args(args.scheduler_class, args.scheduler_kwargs)
    optimizer_class = get_optimizer_class(args.optimizer_class)
    optimizer_kwargs = parse_optimizer_kwargs(args.optimizer_kwargs)
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

        # Update experiment/run name
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
        # RNNLightning uses self.hparams, so we pass the same kw args here
        model = RNNLightning(
            num_hidden_layers=args.num_hidden_layers,
            num_hidden=args.num_hidden,
            rnn_type=args.rnn_type,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            layer_skip=args.layer_skip,
            use_layernorm=args.use_layernorm,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            log_every_n_steps=args.log_every_n_steps,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            num_inputs=args.num_inputs,
            loss_fn=args.loss_fn,
            loss_kwargs=loss_kwargs,
            num_outputs=(args.num_outputs if args.num_outputs > 0 else None),
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
            model=model,
            ckpt_path="best"
        )

        # Finalize WandB logger
        if wandb_logger and not args.no_wandb:
            try:
                wandb_logger.finalize("success")
            except Exception as e:
                logger.warning("wandb_logger.finalize() failed: %s", e)

        logger.info("Run %d finished. Best checkpoint: %s", run_idx + 1, checkpoint_callback.best_model_path or "N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNLightning model")
    default_max_epochs = 10

    # Data / logging
    parser.add_argument("--data", type=str, default="./data/dataset.ds", help="Path to dataset (torch file).")
    parser.add_argument("--experiment-name", type=str, default="rnn_test", help="WandB experiment/run name.")
    parser.add_argument("--project-name", type=str, default="RNN-Lightning", help="WandB project name.")
    parser.add_argument("--logging-directory", type=str, default=".temp", help="Local directory where logs/wandb files are stored.")
    parser.add_argument("--checkpoint-path", type=str, default="models/RNN", help="Directory to save checkpoints.")
    parser.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor for checkpointing.")
    parser.add_argument("--monitor-mode", dest="monitor_mode", choices=["min", "max"], default="min", help="Monitor mode for checkpointing/early stopping.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Logging frequency (trainer.log_every_n_steps)")

    # Multi-run / seeding
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent runs with different seeds for statistics.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for reproducibility. Each run will increment from this.")

    # Model hyperparameters (map to RNNLightning)
    parser.add_argument("--num-hidden", type=int, default=64, help="Number of hidden units.")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru", "rnn"], help="RNN type: lstm, gru, rnn.")
    parser.add_argument("--num-inputs", dest="num_inputs", type=int, default=0, help="Input feature dimension (inferred from data if 0).")
    parser.add_argument("--num-outputs", type=int, default=0, help="Output dimension (defaults to num_inputs if 0).")
    parser.add_argument("--layer-skip", type=int, default=0, help="Layer skip (residual) distance.")
    parser.add_argument("--use-layernorm", type=str2bool, default=False, help="Whether to use LayerNorm between RNN layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between layers (0=no dropout).")

    # Training hyperparameters
    parser.add_argument("--max-epochs", type=int, default=default_max_epochs, help="Max number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (passed to model).")
    parser.add_argument("--scheduler-class", type=str, default="cosine", choices=["none", "cosine", "exponential", "step", "plateau"], help="Learning rate scheduler type.")
    parser.add_argument("--scheduler-kwargs", type=str, default=f"T_max={default_max_epochs}", help="Scheduler kwargs as key=value pairs separated by commas.")
    parser.add_argument("--optimizer-class", type=str, default="adamw", choices=["adam", "adamw", "sgd", "rmsprop", "adagrad"], help="Optimizer to use.")
    parser.add_argument("--optimizer-kwargs", type=str, default="", help="Extra optimizer args as key=value pairs, e.g. 'betas=(0.9,0.999),eps=1e-8,weight_decay=0.01'.")
    parser.add_argument("--early-stopping", type=str2bool, default=True, help="Enable early stopping.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4, help="Min delta for early stopping.")
    parser.add_argument("--loss-fn", type=str, default="mse", help="Loss function to use; e.g. 'mse','mae','huber','logcosh'.")
    parser.add_argument("--loss-kwargs", type=str, default="", help="Optional loss kwargs as key=value pairs separated by commas.")

    # Misc
    parser.add_argument("--torch-compile", action="store_true", help="Attempt torch.compile(model) before training.")
    parser.add_argument("--use-gpu-if-available", action="store_true", help="Use GPU if available (default: off).")

    args = parser.parse_args()

    main(args)