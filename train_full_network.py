#!/usr/bin/env python3
"""
train_snn.py

Usage examples:
  python train_snn.py --epochs 50 --device gpu --experiment myrun --project Spike-Synth-Full \
      --checkpoint-dir models/SRNN --surrogate-ckpt surrogate/models/SRNN/testspike_model-epoch=09-val_loss=0.09-v3.ckpt \
      --hidden 128 64

See --help for all options.
"""

import pprint
import os
import time
import logging
import torch
import snntorch as snn
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from utils.configuration import load_args
from utils import FormulateArgs, MakeFolder, SetSeed
from utils.Loader import GetDataLoader
from utils.logger import GetMessageLogger
import utils.training as training
import utils.PrintedSpikingNN_lP_New as pSNN

from surrogate.utils.spiking_architecture import SpikingNetwork
from surrogate.utils.non_spiking_architecture import NonSpikingNetwork
from surrogate.utils.MyTransformer_lP import GPTLightning, GPT

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train a Printed Spiking Network (from notebook->script)")

    # dataset / run basics
    p.add_argument("--dataset", type=int, default=0, help="Dataset index. See utils.Loader.py")
    p.add_argument("--seed", type=int, default=43, help="SEED value")
    p.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs (overrides config EPOCH)")
    p.add_argument("--timelimit", type=float, default=0.1, help="TIMELIMITATION value")

    # optimizer / lr
    p.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    p.add_argument("--lr-min", type=float, default=5e-2, help="Minimum learning rate")

    # model topology
    p.add_argument("--hidden", type=int, nargs="*", default=None,
                   help="Hidden layer sizes, e.g. --hidden 128 64 (if omitted, will use config/defaults)")
    p.add_argument("--surrogate-class", type=str, choices=["baseline-gpt", "spiking", "non-spiking"], default="spiking",
                   help="Which type of surrogate you want to use.")

    # logging / checkpoint
    p.add_argument("--experiment", type=str, default="test", help="WandB experiment/run name")
    p.add_argument("--project", type=str, default="Spike-Synth-Full", help="WandB project name")
    p.add_argument("--log-dir", type=str, default=".temp", help="Directory for wandb/local logs")
    p.add_argument("--checkpoint-dir", type=str, help="Where to save checkpoints")
    p.add_argument("--surrogate-ckpt", type=str, default="surrogate/models/Spiking/LeakyParallel/RSNN_wLMSE-runrun_idx=0-epoch=78-val_loss=0.07.ckpt.ckpt", help="Surrogate model checkpoint path for SpikeSynth")

    # runtime flags
    p.add_argument("--progressive", action="store_true", help="Set PROGRESSIVE flag")
    p.add_argument("--fast-dev-run", action="store_true", help="Run lightning in fast_dev_run mode (debug)")

    return p.parse_args()


def main():
    logger.info("Starting train_snn.py")
    
    args_cli = parse_args()
    if args_cli.checkpoint_dir is None:
        args_cli.checkpoint_dir = f"models/FullNetwork/{args_cli.surrogate_class}"
    
    logger.info("Parsed CLI args: %s", args_cli)

    # Build overrides dict for configuration.load_args
    overrides = {
        "DATASET": args_cli.dataset,
        "SEED": args_cli.seed,
        "projectname": "pLR-SNN",
        "DEVICE": args_cli.device,
        "PROGRESSIVE": args_cli.progressive,
        "EPOCH": args_cli.epochs,
        "TIMELIMITATION": args_cli.timelimit,
        "LR_MIN": args_cli.lr_min,
        "LR": args_cli.lr,
    }
    logger.debug("Configuration overrides: %s", overrides)

    # Load base args using the project's configuration loader
    try:
        args = load_args(overrides=overrides)
        logger.info("Loaded configuration via load_args")
    except Exception as e:
        logger.exception("Failed to load configuration with overrides=%s: %s", overrides, e)
        raise

    # If user provided CLI hidden sizes, override args.hidden
    if args_cli.hidden:
        args.hidden = list(args_cli.hidden)
        logger.info("Overrode hidden topology from CLI: %s", args.hidden)
    else:
        logger.info("Using hidden topology from config: %s", getattr(args, "hidden", None))

    # Finalize args (same as notebook)
    try:
        args = FormulateArgs(args)
        logger.info("Formulated args successfully")
    except Exception as e:
        logger.exception("FormulateArgs failed: %s", e)
        raise

    # Set seed for reproducibility
    try:
        SetSeed(args.SEED)
        logger.info("Set random seed to %s", args.SEED)
    except Exception:
        # If SetSeed is not available or fails, ignore but warn
        logger.warning("SetSeed failed or is unavailable; continuing without explicit seed set.")

    # Create data loaders
    try:
        train_loader, datainfo = GetDataLoader(args, 'train')
        valid_loader, _ = GetDataLoader(args, 'valid')
        test_loader, _ = GetDataLoader(args, 'test')
        logger.info("Data loaders created successfully")
    except Exception as e:
        logger.exception("Failed creating data loaders: %s", e)
        raise

    logger.info("Data information:\n%s", pprint.pformat(datainfo))

    # prepare logging directory
    script_dir = os.getcwd()
    logging_directory = os.path.join(script_dir, args_cli.log_dir)
    logging_directory = os.path.abspath(logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory
    logger.info("Logging directory is set to %s (WANDB_DIR)", logging_directory)

    if args_cli.surrogate_class == "spiking":
        surrogate_class = SpikingNetwork
    elif args_cli.surrogate_class == "baseline-gpt":       
        surrogate_class = GPTLightning
    else:
        surrogate_class = NonSpikingNetwork

    # instantiate the model wrapper
    surrogate_ckpt = args_cli.surrogate_ckpt
    topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
    logger.info("Instantiating PrintedSpikingNetwork with topology: %s", topology)
    try:
        psnn = pSNN.LightningPrintedSpikingNetwork(
            topology=topology,
            args=args,
            model_class=surrogate_class,
            ckpt_path=surrogate_ckpt,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            surrogate_gradient=snn.surrogate.atan()
        )
        logger.info("PrintedSpikingNetwork instantiated (surrogate_ckpt=%s, surrogate_ckass=%s)", surrogate_ckpt, args_cli.surrogate_class)
    except Exception as e:
        logger.exception("Failed to instantiate PrintedSpikingNetwork: %s", e)
        raise

    # WandB logger
    logger.info("Setting up WandB logger (project=%s, run=%s)", args_cli.project, args_cli.experiment)
    wandb_logger = WandbLogger(
        log_model=True,
        project=args_cli.project,
        name=args_cli.experiment,
        save_dir=logging_directory,
    )

    # optional: log code and watch model if wandb available
    try:
        wandb_logger.watch(psnn)
        wandb_logger.experiment.log_code(".", include_fn=lambda path: path.endswith('.py') or path.endswith('.ipynb'))
        logger.info("WandB watch and log_code succeeded")
    except Exception as e:
        logger.warning("wandb watch/log_code failed: %s", e)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args_cli.checkpoint_dir,
        filename=f"{args_cli.experiment}-{args_cli.surrogate_class}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    logger.info("ModelCheckpoint configured: dir=%s filename=%s", args_cli.checkpoint_dir, checkpoint_callback.filename)

    accelerator = "cpu"
    if args_cli.device == "gpu" and torch.cuda.is_available():
        accelerator = "gpu"
    logger.info("Using accelerator: %s", accelerator)

    # instantiate trainer
    trainer = Trainer(
        fast_dev_run=args_cli.fast_dev_run,
        max_epochs=args_cli.epochs,
        logger=wandb_logger,
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
    )
    logger.info("PyTorch Lightning Trainer instantiated (fast_dev_run=%s, max_epochs=%d)", args_cli.fast_dev_run, args_cli.epochs)

    # Train
    try:
        logger.info("Starting training for %d epochs", args_cli.epochs)
        trainer.fit(psnn)
        logger.info("Training finished successfully")
    except Exception as e:
        logger.exception("Training failed with exception: %s", e)
        raise

    # --- Run test on best checkpoint ---
    logger.info("Starting test using best checkpoint")
    try:
        trainer.test(
            model=psnn,        # optional, will load checkpoint automatically if ckpt_path="best"
            ckpt_path="best"
        )
        logger.info("Test finished. Best checkpoint: %s", checkpoint_callback.best_model_path or "N/A")
    except Exception as e:
        logger.exception("Testing failed: %s", e)
        raise

    # --- Finalize wandb logger ---
    try:
        if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment:
            wandb_logger.experiment.finish()
            logger.info("WandB experiment finished")
    except Exception as e:
        logger.warning("wandb_logger.finalize() failed: %s", e)

    logger.info("train_snn.py completed")


if __name__ == '__main__':
    main()