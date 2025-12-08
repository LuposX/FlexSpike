#!/usr/bin/env python3
"""
train_gpt.py

Run from terminal, e.g.:

python train_gpt.py --data ./data/dataset.ds --batch-size 2048 --max-epochs 100

Use --help to see all options.
"""

import argparse
import os
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

# adjust this import to match your package layout
from utils.MyTransformer_lP import GPTLightning, GPT

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


def main(args):
    logger.info("Loading dataset from %s", args.data)
    data = torch.load(args.data)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_valid = data["X_valid"]
    Y_valid = data["Y_valid"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    train_ds = TensorDataset(X_train, Y_train)
    valid_ds = TensorDataset(X_valid, Y_valid)
    test_ds = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Build model config
    model_config = GPT.get_default_config()
    model_config.model_type = args.model_type
    # keep block_size from training data shape
    model_config.block_size = X_train.shape[1]
    model = GPTLightning(model_config, lr=args.lr, max_epochs=args.max_epochs)

    # logging directory setup
    script_dir = os.getcwd()
    logging_directory = os.path.join(script_dir, args.logging_directory)
    logging_directory = os.path.abspath(logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory

    summary = ModelSummary(model, max_depth=1)
    num_params = summary.total_parameters
    trainable_params = summary.trainable_parameters
    logger.info("Model has %d trainable parameters (total %d)", trainable_params, num_params)

    # WandB logger
    wandb_logger = None
    if not args.no_wandb:
        logger.info("Initializing WandbLogger (project=%s name=%s) at %s", args.project_name, args.experiment_name, logging_directory)
        wandb_logger = WandbLogger(
            log_model=True,
            project=args.project_name,
            name=args.experiment_name,
            save_dir=logging_directory,
        )
        try:
            wandb_logger.experiment.summary["trainable_parameters"] = trainable_params
            wandb_logger.experiment.summary["total_parameters"] = num_params
        except Exception as e:
            logger.warning("wandb_logger.experiment.summary() failed: %s", e)
        # log gradients and model topology (safe-guard)
        try:
            wandb_logger.watch(model)
        except Exception as e:
            logger.warning("wandb_logger.watch() failed: %s", e)
        try:
            # log code (only python/ipynb)
            wandb_logger.experiment.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        except Exception as e:
            logger.warning("wandb_logger.experiment.log_code() failed: %s", e)
    else:
        logger.info("WandB disabled (--no-wandb).")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,
        filename=f"{args.experiment_name}-{{epoch:02d}}-{{{args.monitor}:.2f}}",
        save_last=True,
        dirpath=args.checkpoint_dir if args.checkpoint_dir else None,
    )

    early_stop_callback = EarlyStopping(monitor=args.monitor, patience=args.patience, mode=args.monitor_mode, verbose=True)

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() and args.use_gpu_if_available else "cpu"
    logger.info("Using accelerator=%s (torch.cuda.is_available()=%s, use_gpu_if_available=%s)", accelerator, torch.cuda.is_available(), args.use_gpu_if_available)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger if not args.no_wandb else None,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=logging_directory,
    )


    # Optionally compile model (commented by default)
    if args.torch_compile:
        try:
            logger.info("Attempting torch.compile(model)")
            model = torch.compile(model)
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

    # Train
    logger.info("Starting training for up to %d epochs", args.max_epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # finalize wandb
    if wandb_logger and not args.no_wandb:
        try:
            wandb_logger.finalize("success")
        except Exception as e:
            logger.warning("wandb_logger.finalize() failed: %s", e)

    # Test if requested
    if args.test_after_training:
        logger.info("Running trainer.test()")
        trainer.test(model, dataloaders=test_loader)

    logger.info("Best checkpoint saved at: %s", checkpoint_callback.best_model_path or "N/A")
    print("Best checkpoint saved at:", checkpoint_callback.best_model_path or "N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT Surrogate.")

    # Data and dataloader
    parser.add_argument("--data", type=str, default="./data/dataset.ds", help="Path to dataset (torch .pt/.ds) file.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for all dataloaders.")
    parser.add_argument("--num-workers", type=int, default=4, help="num_workers for DataLoader.")
    parser.add_argument("--pin-memory", type=str2bool, default=False, help="Whether to use pin_memory in DataLoader (true/false).")

    # Logging / wandb
    parser.add_argument("--experiment-name", type=str, default=None, help="WandB experiment/run name.")
    parser.add_argument("--project-name", type=str, default="Spike-Synth-Surrogate", help="WandB project name.")
    parser.add_argument("--logging-directory", type=str, default=".temp", help="Local directory where logs/wandb files are stored.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging (useful for local quick debugging).")

    # Checkpointing and callbacks
    parser.add_argument("--checkpoint-dir", type=str, default="models/GPT", help="Directory to save checkpoints (ModelCheckpoint.dirpath).")
    parser.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor for checkpointing/early stopping.")
    parser.add_argument("--monitor-mode", dest="monitor_mode", choices=["min", "max"], default="min", help="Monitor mode for checkpointing/early stopping.")
    parser.add_argument("--save-top-k", type=int, default=1, help="How many top checkpoints to keep.")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience (in validation checks).")

    # Model / training hyperparameters
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max number of training epochs.")
    parser.add_argument("--model-type", type=str, default="pico-test", help="model_config.model_type to set.")
    parser.add_argument("--torch-compile", dest="torch_compile", action="store_true", help="Attempt torch.compile(model) before training.")
    parser.add_argument("--use-gpu-if-available", dest="use_gpu_if_available", action="store_true", help="Use GPU if available (default: off).")
    
    # Misc
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Trainer.log_every_n_steps")
    parser.add_argument("--test-after-training", dest="test_after_training", action="store_true", help="Run trainer.test() after training.")

    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = args.model_type

    main(args)