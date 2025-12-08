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
import torch
import snntorch as snn

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



def parse_args():
    p = argparse.ArgumentParser(description="Train a Printed Spiking Network (from notebook->script)")

    # dataset / run basics
    p.add_argument("--dataset", type=int, default=0, help="Dataset index (matches notebook DATASET)")
    p.add_argument("--seed", type=int, default=0, help="SEED value")
    p.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs (overrides config EPOCH)")
    p.add_argument("--timelimit", type=float, default=0.1, help="TIMELIMITATION value")

    # optimizer / lr
    p.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    p.add_argument("--lr-min", type=float, default=5e-2, help="Minimum learning rate")

    # model topology
    p.add_argument("--hidden", type=int, nargs="*", default=None,
                   help="Hidden layer sizes, e.g. --hidden 128 64 (if omitted, will use config/defaults)")

    # logging / checkpoint
    p.add_argument("--experiment", type=str, default="test", help="WandB experiment/run name")
    p.add_argument("--project", type=str, default="Spike-Synth-Full", help="WandB project name")
    p.add_argument("--log-dir", type=str, default=".temp", help="Directory for wandb/local logs")
    p.add_argument("--checkpoint-dir", type=str, default="models/SRNN", help="Where to save checkpoints")
    p.add_argument("--surrogate-ckpt", type=str, default=None, help="Optional surrogate model checkpoint path for SpikeSynth")

    # runtime flags
    p.add_argument("--progressive", action="store_true", help="Set PROGRESSIVE flag")
    p.add_argument("--fast-dev-run", action="store_true", help="Run lightning in fast_dev_run mode (debug)")

    return p.parse_args()


def main():
    args_cli = parse_args()

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

    # Load base args using the project's configuration loader
    args = load_args(overrides=overrides)

    # If user provided CLI hidden sizes, override args.hidden
    if args_cli.hidden:
        args.hidden = list(args_cli.hidden)

    # Finalize args (same as notebook)
    args = FormulateArgs(args)

    # Set seed for reproducibility
    try:
        SetSeed(args.SEED)
    except Exception:
        # If SetSeed is not available or fails, ignore but warn
        print("Warning: SetSeed failed or is unavailable; continuing without explicit seed set.")

    # Create data loaders
    train_loader, datainfo = GetDataLoader(args, 'train')
    valid_loader, _ = GetDataLoader(args, 'valid')
    test_loader, _ = GetDataLoader(args, 'test')

    pprint.pprint(datainfo)

    # prepare logging directory
    script_dir = os.getcwd()
    logging_directory = os.path.join(script_dir, args_cli.log_dir)
    logging_directory = os.path.abspath(logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory

    # instantiate the model wrapper
    surrogate_ckpt = args_cli.surrogate_ckpt

    psnn = pSNN.LightningPrintedSpikingNetwork(
        topology=[datainfo['N_feature']] + args.hidden + [datainfo['N_class']],
        args=args,
        model_class=SpikingNetwork,
        ckpt_path=surrogate_ckpt,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        surrogate_gradient=snn.surrogate.atan()
    )

    # WandB logger
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
    except Exception as e:
        print(f"Warning: wandb watch/log_code failed: {e}")

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args_cli.checkpoint_dir,
        filename=args_cli.experiment + str("-pLRSNN-{epoch:02d}-{val_loss:.2f}"),
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    accelerator = "cpu"
    devices = None
    if args_cli.device == "gpu" and torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1

    # instantiate trainer
    trainer = Trainer(
        fast_dev_run=args_cli.fast_dev_run,
        max_epochs=args_cli.epochs,
        logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
    )

    # Train
    try:
        trainer.fit(psnn)
    except Exception as e:
        print(f"Training failed with exception: {e}")
        raise
    finally:
        # ensure wandb run finishes cleanly (if present)
        try:
            if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment:
                wandb_logger.experiment.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()