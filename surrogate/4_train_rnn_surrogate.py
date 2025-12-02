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
import sys
import copy
import yaml
import wandb

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

    # Parse some defaults (these may be overridden per-run if not sweeping)
    scheduler_class, scheduler_kwargs = parse_scheduler_args(args.scheduler_class, args.scheduler_kwargs)
    optimizer_class = get_optimizer_class(args.optimizer_class)
    # optimizer_kwargs may be a string (from CLI) -> parse now; if sweep supplies dict, we'll use that instead
    optimizer_kwargs_cli = parse_optimizer_kwargs(args.optimizer_kwargs)
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

    # Helper that runs a single run (seed)
    def run_single_run(run_idx: int, config_override: Optional[Dict[str, Any]] = None):
        # Make a shallow copy of args so per-run overrides don't leak
        local_args = copy.deepcopy(vars(args))

        # Convert config_override into a plain dict (if given)
        config_dict = None
        if config_override:
            # If it's already a dict, use it directly
            if isinstance(config_override, dict):
                config_dict = config_override
            else:
                # Try to convert (this will succeed when wandb.agent has provided wandb.config)
                try:
                    config_dict = dict(config_override)
                except Exception as e:
                    # Fallback: log and continue with no overrides
                    logger.warning(
                        "Could not convert config_override to dict: %s. Ignoring sweep overrides for this run.",
                        e,
                    )
                    config_dict = None

        # If we got a config dict, apply it to local args.
        # NOTE: keys in wandb.config should match your CLI arg names (e.g. 'lr', 'batch_size').
        if config_dict:
            # Normalize simple string booleans like "true"/"false"
            for k, v in list(config_dict.items()):
                if isinstance(v, str) and v.lower() in ("true", "false"):
                    config_dict[k] = v.lower() == "true"
            local_args.update(config_dict)

        # Recompute classes/kwargs now that we may have overrides in local_args
        local_scheduler_class, local_scheduler_kwargs = parse_scheduler_args(
            local_args.get("scheduler_class"), local_args.get("scheduler_kwargs")
        )
        local_optimizer_class = get_optimizer_class(local_args.get("optimizer_class"))
        # optimizer kwargs: if sweep provided a dict/object, accept that; if string, parse it
        local_optimizer_kwargs = local_args.get("optimizer_kwargs", "")
        if isinstance(local_optimizer_kwargs, str):
            local_optimizer_kwargs = parse_optimizer_kwargs(local_optimizer_kwargs)
        # If it was provided via CLI originally and parsed earlier, prefer parsed CLI unless overridden
        if not local_optimizer_kwargs and optimizer_kwargs_cli:
            local_optimizer_kwargs = optimizer_kwargs_cli

        local_loss_kwargs = local_args.get("loss_kwargs", "")
        if isinstance(local_loss_kwargs, str):
            local_loss_kwargs = parse_optimizer_kwargs(local_loss_kwargs)
        if not local_loss_kwargs and loss_kwargs:
            local_loss_kwargs = loss_kwargs

        # seeds
        seed = int(local_args.get("base_seed", 42)) + run_idx
        logger.info(
            "=== Starting run %d/%d with seed %d ===", run_idx + 1, int(local_args.get("num_runs", 1)), seed
        )

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Update experiment/run name
        run_name = f"{local_args.get('experiment_name')}_run{run_idx+1}"

        # WandB logger (do NOT call wandb.init() manually; let WandbLogger manage runs)
        wandb_logger = None
        if not local_args.get("no_wandb"):
            wandb_logger = WandbLogger(
                log_model=True,
                project=local_args.get("project_name"),
                name=run_name,
                save_dir=logging_directory,
            )

        callbacks = [checkpoint_callback]
        if local_args.get("early_stopping"):
            early_stopping_callback = EarlyStopping(
                monitor=local_args.get("monitor"),
                mode=local_args.get("monitor_mode"),
                patience=int(local_args.get("early_stopping_patience")),
                min_delta=float(local_args.get("early_stopping_delta")),
                verbose=True,
            )
            callbacks.append(early_stopping_callback)

        # Instantiate model with local args (these values may have been overridden by sweep)
        model = RNNLightning(
            num_hidden_layers=int(local_args.get("num_hidden_layers")),
            num_hidden=int(local_args.get("num_hidden")),
            rnn_type=local_args.get("rnn_type"),
            optimizer_class=local_optimizer_class,
            optimizer_kwargs=local_optimizer_kwargs,
            lr=float(local_args.get("lr")),
            batch_size=int(local_args.get("batch_size")),
            dropout=float(local_args.get("dropout")),
            max_epochs=int(local_args.get("max_epochs")),
            layer_skip=int(local_args.get("layer_skip")),
            use_layernorm=bool(local_args.get("use_layernorm")),
            scheduler_class=local_scheduler_class,
            scheduler_kwargs=local_scheduler_kwargs,
            log_every_n_steps=int(local_args.get("log_every_n_steps")),
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            num_inputs=int(local_args.get("num_inputs")),
            loss_fn=local_args.get("loss_fn"),
            loss_kwargs=local_loss_kwargs,
            num_outputs=(int(local_args.get("num_outputs")) if int(local_args.get("num_outputs")) > 0 else None),
        )

        # Optional torch.compile
        if local_args.get("torch_compile"):
            try:
                logger.info("Attempting torch.compile(model)")
                compiled = torch.compile(model)
                model = compiled
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

        # Trainer
        trainer = Trainer(
            max_epochs=int(local_args.get("max_epochs")),
            accelerator=accelerator,
            logger=wandb_logger if not local_args.get("no_wandb") else None,
            callbacks=callbacks,
            log_every_n_steps=int(local_args.get("log_every_n_steps")),
        )

        # Train
        trainer.fit(model)

        logger.info("Starting test for run %d using best checkpoint", run_idx + 1)
        trainer.test(model=model, ckpt_path="best")

        # Finalize WandB logger
        if wandb_logger and not local_args.get("no_wandb"):
            try:
                wandb_logger.finalize("success")
            except Exception as e:
                logger.warning("wandb_logger.finalize() failed: %s", e)

        logger.info("Run %d finished. Best checkpoint: %s", run_idx + 1, checkpoint_callback.best_model_path or "N/A")

    # If sweep is enabled, set up the sweep and use wandb.agent to run trials
    if args.wandb_sweep_enable:
        # Validate YAML path
        if not args.wandb_sweep_yaml:
            logger.error("WandB sweep is enabled but no --wandb-sweep-yaml path was provided.")
            sys.exit(2)
        sweep_yaml_path = os.path.abspath(args.wandb_sweep_yaml)
        if not os.path.exists(sweep_yaml_path):
            logger.error("WandB sweep yaml file not found: %s", sweep_yaml_path)
            sys.exit(2)

        # load yaml
        try:
            with open(sweep_yaml_path, "r") as f:
                sweep_config = yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load sweep yaml: %s", e)
            sys.exit(2)

        # create the sweep on wandb
        try:
            sweep_id = wandb.sweep(sweep_config, project=args.project_name)
            logger.info("Created/returned sweep id: %s", sweep_id)
            # print some helpful info
            print(f"Create sweep with ID: {sweep_id}")
            if "project" in sweep_config:
                print(f"Sweep project in yaml: {sweep_config.get('project')}")
            print(f"Sweep URL: https://wandb.ai/{args.project_name}/sweeps/{sweep_id}")
        except Exception as e:
            logger.error("wandb.sweep() failed: %s", e)
            sys.exit(2)

        # Define the function that wandb.agent will call for each trial
        def _agent_run():
            # Let the wandb.agent / WandbLogger lifecycle create/manage wandb runs.
            # wandb.config is populated by the agent; we can access it directly here.
            cfg = wandb.config
            num_runs_local = int(args.num_runs)
            for run_idx in range(num_runs_local):
                run_single_run(run_idx, config_override=cfg)

        # Start agent. pass count if user provided positive integer
        count = int(args.wandb_sweep_count) if args.wandb_sweep_count and args.wandb_sweep_count > 0 else None
        logger.info("Starting wandb.agent for sweep_id=%s (count=%s)", sweep_id, str(count))
        try:
            wandb.agent(sweep_id, function=_agent_run, count=count)
        except Exception as e:
            logger.error("wandb.agent failed: %s", e)
            sys.exit(2)
    else:
        # Not sweeping -> normal runs
        for run_idx in range(args.num_runs):
            run_single_run(run_idx, config_override=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNLightning model")
    default_max_epochs = 10

    # Data / logging
    parser.add_argument("--data", type=str, default="./data/v3_dataset.ds", help="Path to dataset (torch file).")
    parser.add_argument("--experiment-name", type=str, default="rnn_test", help="WandB experiment/run name.")
    parser.add_argument("--project-name", type=str, default="Non-Spiking", help="WandB project name.")
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

    # WandB sweep related
    parser.add_argument("--wandb-sweep-enable", action="store_true", help="Enable WandB hyperparameter sweep mode. When enabled, many CLI hyperparameters are ignored and should be controlled via the sweep yaml.")
    parser.add_argument("--wandb-sweep-yaml", type=str, default="sweep_rnn.yaml", help="Path to WandB sweep yaml file (required when --wandb-sweep-enable).")
    parser.add_argument("--wandb-sweep-count", type=int, default=0, help="Optional: max number of runs for wandb.agent (0 means no explicit limit).")

    args = parser.parse_args()

    # If sweep mode is enabled, check for any CLI-provided hyperparameters that would be meaningless
    if args.wandb_sweep_enable:
        # list of argument dest names that are usually controlled by sweep and should not be set on CLI
        suspect_args = [
            "lr",
            "batch_size",
            "num_hidden",
            "num_hidden_layers",
            "dropout",
            "optimizer_class",
            "optimizer_kwargs",
            "scheduler_class",
            "scheduler_kwargs",
            "loss_fn",
            "loss_kwargs",
        ]
        conflicting = []
        for dest in suspect_args:
            cli_val = getattr(args, dest, None)
            default_val = parser.get_default(dest)
            # Special handling for empty string vs None
            if isinstance(default_val, str):
                is_conflict = (cli_val != default_val and cli_val != "" and cli_val is not None)
            else:
                is_conflict = (cli_val != default_val)
            if is_conflict:
                conflicting.append((dest, default_val, cli_val))

        if conflicting:
            msg_lines = [
                "Incompatible CLI arguments detected while WandB sweep mode is enabled.",
                "When using --wandb-sweep-enable you should let the sweep YAML control hyperparameters.",
                "Please remove the following CLI flags or disable sweep mode:"
            ]
            for dest, default_val, cli_val in conflicting:
                msg_lines.append(f"  --{dest.replace('_', '-')} (default: {default_val!r})  -- you provided: {cli_val!r}")
            logger.error("\n".join(msg_lines))
            parser.print_help()
            sys.exit(2)

    main(args)