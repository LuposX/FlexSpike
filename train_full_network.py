#!/usr/bin/env python3
"""
train_snn.py

Usage examples:
  python train_snn.py --epochs 50 --device gpu --experiment myrun --project Spike-Synth-Full \
      --checkpoint-dir models/SRNN --surrogate-ckpt surrogate/models/SRNN/testspike_model-epoch=09-val_loss=0.09-v3.ckpt \
      --hidden 128 64

  # multiple datasets:
  python train_snn.py --datasets "0,5,8,12" --experiment multi --project Spike-Synth-Full

See --help for all options.
"""

import pprint
import os
import time
import logging
import torch
import snntorch as snn
import argparse
from typing import List

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


def parse_dataset_list(s: str) -> List[int]:
    """
    Parse a dataset list string like "0,5,8,12" or "0 5 8 12" or mixed.
    Returns list of ints.
    """
    if s is None:
        return []
    parts = []
    for token in s.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        # support simple ranges "2-5"
        if "-" in token:
            start_end = token.split("-")
            if len(start_end) == 2:
                try:
                    start = int(start_end[0])
                    end = int(start_end[1])
                    if start <= end:
                        parts.extend(list(range(start, end + 1)))
                    else:
                        parts.extend(list(range(start, end - 1, -1)))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid dataset range token: '{token}'")
            else:
                raise argparse.ArgumentTypeError(f"Invalid dataset range token: '{token}'")
        else:
            try:
                parts.append(int(token))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid dataset token: '{token}'")
    return parts


def parse_args():
    p = argparse.ArgumentParser(description="Train a Printed Spiking Network (from notebook->script)")

    # dataset / run basics
    p.add_argument("--dataset", type=int, default=0, help="Dataset index. See utils.Loader.py (used if --datasets not provided)")
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma- or space-separated dataset indices (e.g. '0,5,8,12'). Overrides --dataset if provided. Ranges like 2-4 are supported.")
    p.add_argument("--seed", type=int, default=42, help="SEED value")
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
    p.add_argument("--stop-on-error", action="store_true", help="Stop on first dataset error (default: continue to next dataset)")

    return p.parse_args()


def main():
    logger.info("Starting train_snn.py")

    args_cli = parse_args()

    # parse dataset list (CLI)
    if args_cli.datasets:
        try:
            dataset_list = parse_dataset_list(args_cli.datasets)
            if not dataset_list:
                raise ValueError("Parsed --datasets is empty.")
        except Exception as e:
            logger.exception("Failed parsing --datasets '%s': %s", args_cli.datasets, e)
            raise
    else:
        dataset_list = [args_cli.dataset]

    # default checkpoint dir if not provided
    if args_cli.checkpoint_dir is None:
        args_cli.checkpoint_dir = f"models/FullNetwork/{args_cli.surrogate_class}"
    base_checkpoint_dir = args_cli.checkpoint_dir

    logger.info("Will run datasets in sequence: %s", dataset_list)

    # We'll process each dataset sequentially
    for dset in dataset_list:
        logger.info("=== Starting run for dataset %s ===", dset)
        # Build overrides dict for configuration.load_args
        overrides = {
            "DATASET": dset,
            "SEED": args_cli.seed,
            "projectname": "pLR-SNN",
            "DEVICE": args_cli.device,
            "PROGRESSIVE": args_cli.progressive,
            "EPOCH": args_cli.epochs,
            "TIMELIMITATION": args_cli.timelimit,
            "LR_MIN": args_cli.lr_min,
            "LR": args_cli.lr,
        }
        logger.debug("Configuration overrides for dataset %s: %s", dset, overrides)

        # Load base args using the project's configuration loader
        try:
            args = load_args(overrides=overrides)
            logger.info("Loaded configuration via load_args (dataset=%s)", dset)
        except Exception as e:
            logger.exception("Failed to load configuration for dataset %s with overrides=%s: %s", dset, overrides, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                continue

        # If user provided CLI hidden sizes, override args.hidden
        if args_cli.hidden:
            args.hidden = list(args_cli.hidden)
            logger.info("Overrode hidden topology from CLI: %s", args.hidden)
        else:
            logger.info("Using hidden topology from config: %s", getattr(args, "hidden", None))

        # Finalize args (same as notebook)
        try:
            args = FormulateArgs(args)
            logger.info("Formulated args successfully (dataset=%s)", dset)
        except Exception as e:
            logger.exception("FormulateArgs failed for dataset %s: %s", dset, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                continue

        # Set seed for reproducibility
        try:
            SetSeed(args.SEED)
            logger.info("Set random seed to %s (dataset=%s)", args.SEED, dset)
        except Exception:
            logger.warning("SetSeed failed or is unavailable; continuing without explicit seed set.")

        # Create data loaders
        try:
            train_loader, datainfo = GetDataLoader(args, 'train')
            valid_loader, _ = GetDataLoader(args, 'valid')
            test_loader, _ = GetDataLoader(args, 'test')
            logger.info("Data loaders created successfully (dataset=%s)", dset)
        except Exception as e:
            logger.exception("Failed creating data loaders for dataset %s: %s", dset, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                continue

        logger.info("Data information (dataset=%s):\n%s", dset, pprint.pformat(datainfo))

        # prepare logging directory (make separate directories per dataset to avoid clobber)
        script_dir = os.getcwd()
        logging_directory = os.path.join(script_dir, args_cli.log_dir, f"dataset_{dset}")
        logging_directory = os.path.abspath(logging_directory)
        os.makedirs(logging_directory, exist_ok=True)
        os.environ["WANDB_DIR"] = logging_directory
        logger.info("Logging directory is set to %s (WANDB_DIR) (dataset=%s)", logging_directory, dset)

        # select surrogate class
        if args_cli.surrogate_class == "spiking":
            surrogate_class = SpikingNetwork
        elif args_cli.surrogate_class == "baseline-gpt":
            surrogate_class = GPTLightning
        else:
            surrogate_class = NonSpikingNetwork

        # instantiate the model wrapper
        surrogate_ckpt = args_cli.surrogate_ckpt
        hidden_list = args.hidden if getattr(args, "hidden", None) else []
        topology = [datainfo['N_feature']] + hidden_list + [datainfo['N_class']]
        logger.info("Instantiating PrintedSpikingNetwork with topology: %s (dataset=%s)", topology, dset)
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
            logger.info("PrintedSpikingNetwork instantiated (surrogate_ckpt=%s, surrogate_class=%s, dataset=%s)",
                        surrogate_ckpt, args_cli.surrogate_class, dset)
        except Exception as e:
            logger.exception("Failed to instantiate PrintedSpikingNetwork for dataset %s: %s", dset, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                continue

        # WandB logger: include dataset id in run name to keep separate runs
        run_name = f"{args_cli.experiment}-ds{dset}"
        logger.info("Setting up WandB logger (project=%s, run=%s) (dataset=%s)", args_cli.project, run_name, dset)
        wandb_logger = WandbLogger(
            log_model=True,
            project=args_cli.project,
            name=run_name,
            save_dir=logging_directory,
        )

        # optional: log code and watch model if wandb available
        try:
            wandb_logger.watch(psnn)
            wandb_logger.experiment.log_code(".", include_fn=lambda path: path.endswith('.py') or path.endswith('.ipynb'))
            logger.info("WandB watch and log_code succeeded (dataset=%s)", dset)
        except Exception as e:
            logger.warning("wandb watch/log_code failed for dataset %s: %s", dset, e)

        # checkpoint callback: put checkpoints into a per-dataset subdir
        dataset_checkpoint_dir = os.path.join(base_checkpoint_dir, f"dataset_{dset}")
        os.makedirs(dataset_checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=dataset_checkpoint_dir,
            filename=f"{args_cli.experiment}-{args_cli.surrogate_class}-ds{dset}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        logger.info("ModelCheckpoint configured: dir=%s filename=%s (dataset=%s)", dataset_checkpoint_dir, checkpoint_callback.filename, dset)

        accelerator = "cpu"
        if args_cli.device == "gpu" and torch.cuda.is_available():
            accelerator = "gpu"
        logger.info("Using accelerator: %s (dataset=%s)", accelerator, dset)

        # instantiate trainer
        trainer = Trainer(
            fast_dev_run=args_cli.fast_dev_run,
            max_epochs=args_cli.epochs,
            logger=wandb_logger,
            accelerator=accelerator,
            callbacks=[checkpoint_callback],
        )
        logger.info("PyTorch Lightning Trainer instantiated (fast_dev_run=%s, max_epochs=%d) (dataset=%s)",
                    args_cli.fast_dev_run, args_cli.epochs, dset)

        # Train
        try:
            logger.info("Starting training for %d epochs (dataset=%s)", args_cli.epochs, dset)
            trainer.fit(psnn)
            logger.info("Training finished successfully (dataset=%s)", dset)
        except Exception as e:
            logger.exception("Training failed for dataset %s with exception: %s", dset, e)
            # finalize wandb experiment before continuing or exiting
            try:
                if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment:
                    wandb_logger.experiment.finish()
                    logger.info("WandB experiment finished (dataset=%s)", dset)
            except Exception as e2:
                logger.warning("wandb_logger.finalize() failed after training error for dataset %s: %s", dset, e2)

            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                # free GPU cache if available
                if accelerator == "gpu":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue

        # --- Run test on best checkpoint ---
        logger.info("Starting test using best checkpoint (dataset=%s)", dset)
        try:
            trainer.test(
                model=psnn,
                ckpt_path="best"
            )
            logger.info("Test finished. Best checkpoint: %s (dataset=%s)", checkpoint_callback.best_model_path or "N/A", dset)
        except Exception as e:
            logger.exception("Testing failed for dataset %s: %s", dset, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")

        # --- Finalize wandb logger ---
        try:
            if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment:
                wandb_logger.experiment.finish()
                logger.info("WandB experiment finished (dataset=%s)", dset)
        except Exception as e:
            logger.warning("wandb_logger.finalize() failed for dataset %s: %s", dset, e)

        # free GPU cache before next run (if applicable)
        if accelerator == "gpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("=== Completed run for dataset %s ===", dset)

    logger.info("train_snn.py completed for all datasets")


if __name__ == '__main__':
    main()
