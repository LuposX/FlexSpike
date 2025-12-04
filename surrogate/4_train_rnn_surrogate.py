#!/usr/bin/env python3
"""
train_rnn_fixed.py

Example usage:
python train_rnn_fixed.py --data ./data/dataset.ds --max-epochs 10 --batch-size 2048
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

# Ensure this import path matches your project structure
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
        logger.debug("No optimizer name provided, defaulting to AdamW")
        return torch.optim.AdamW
    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    cls = optimizer_map.get(name.lower())
    if cls is None:
        logger.warning("Unknown optimizer '%s' specified, falling back to AdamW", name)
        return torch.optim.AdamW
    logger.info("Using optimizer: %s", name.lower())
    return cls


def parse_optimizer_kwargs(kwargs_str: str) -> Dict[str, Any]:
    kwargs = {}
    if not kwargs_str:
        logger.debug("No optimizer kwargs string provided")
        return kwargs

    logger.debug("Parsing optimizer kwargs string: %s", kwargs_str)
    parts = re.split(r',(?![^(]*\))', kwargs_str)

    for kv in parts:
        if "=" not in kv:
            logger.debug("Skipping non key=value part: %s", kv)
            continue
        key, value = kv.split("=", 1)
        key, value = key.strip(), value.strip()

        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            else:
                # leave it as string
                value = value

        kwargs[key] = value
        logger.debug("Parsed optimizer kw: %s = %r", key, value)

    logger.info("Final parsed optimizer kwargs: %s", kwargs)
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
        if scheduler_class is None:
            logger.warning("Unknown scheduler '%s' specified, will not use a scheduler", scheduler_name)
        else:
            logger.info("Using scheduler: %s", scheduler_name)

    kwargs = {}
    if scheduler_kwargs_str:
        logger.debug("Parsing scheduler kwargs string: %s", scheduler_kwargs_str)
        parts = re.split(r',(?![^(]*\))', scheduler_kwargs_str)
        for kv in parts:
            if "=" in kv:
                key, value = kv.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                kwargs[key] = value
                logger.debug("Parsed scheduler kw: %s = %r", key, value)

    logger.info("Final scheduler kwargs: %s", kwargs)
    return scheduler_class, kwargs


def main(args):
    logger.info("Loading dataset from %s", args.data)
    data = torch.load(args.data)
    logger.info("Loaded data keys: %s", list(data.keys()))
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_valid, Y_valid = data["X_valid"], data["Y_valid"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    logger.info("Shapes: X_train=%s, Y_train=%s, X_valid=%s, Y_valid=%s, X_test=%s, Y_test=%s",
                getattr(X_train, 'shape', None), getattr(Y_train, 'shape', None),
                getattr(X_valid, 'shape', None), getattr(Y_valid, 'shape', None),
                getattr(X_test, 'shape', None), getattr(Y_test, 'shape', None))

    # infer num_inputs
    if getattr(args, "num_inputs", None) is None or args.num_inputs <= 0:
        if X_train.ndim == 3:
            inferred_num_inputs = X_train.shape[2]
        elif X_train.ndim == 2:
            inferred_num_inputs = 1
            X_train = X_train.unsqueeze(-1)
            X_valid = X_valid.unsqueeze(-1)
            X_test = X_test.unsqueeze(-1)
            logger.info("Expanded 2D inputs to 3D by unsqueezing last dim")
        else:
            raise ValueError(f"Unexpected X_train shape: {X_train.shape}")
        args.num_inputs = inferred_num_inputs
        logger.info("Inferred num_inputs=%d from data", args.num_inputs)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, Y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    logger.info("Created datasets: train=%d, valid=%d, test=%d samples",
                len(train_dataset), len(valid_dataset), len(test_dataset))

    # Logging directory
    logging_directory = os.path.abspath(args.logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory
    logger.info("Set WANDB_DIR=%s and ensured directory exists", logging_directory)

    scheduler_class, scheduler_kwargs = parse_scheduler_args(args.scheduler_class, args.scheduler_kwargs)
    optimizer_class = get_optimizer_class(args.optimizer_class)
    optimizer_kwargs_cli = parse_optimizer_kwargs(args.optimizer_kwargs)
    loss_kwargs = parse_optimizer_kwargs(args.loss_kwargs)

    logger.info("Final configuration summary: optimizer=%s, optimizer_kwargs=%s, scheduler=%s, scheduler_kwargs=%s, loss_kwargs=%s",
                getattr(optimizer_class, "__name__", str(optimizer_class)), optimizer_kwargs_cli,
                getattr(scheduler_class, "__name__", str(scheduler_class)), scheduler_kwargs, loss_kwargs)

    os.makedirs(args.checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f"{args.experiment_name}-run{{run_idx}}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor=args.monitor,
        mode=args.monitor_mode,
    )
    logger.info("Checkpointing to %s using pattern %s", args.checkpoint_path, checkpoint_callback.filename)

    accelerator = "gpu" if torch.cuda.is_available() and args.use_gpu_if_available else "cpu"
    logger.info("Using accelerator=%s (cuda available=%s, use_gpu_if_available=%s)", accelerator, torch.cuda.is_available(), args.use_gpu_if_available)

    # --- UPDATED FUNCTION SIGNATURE ---
    def run_single_run(run_idx: int, config_override: Optional[Dict[str, Any]] = None, active_run=None):
        """
        Runs a single training run.
        :param active_run: If provided, this is the active wandb.run object from the sweep agent.
        """
        logger.info("Preparing run %d (active_run present=%s)", run_idx + 1, active_run is not None)
        local_args = copy.deepcopy(vars(args))

        config_dict = None
        if config_override:
            logger.info("Received config_override of type %s", type(config_override))
            if isinstance(config_override, dict):
                config_dict = config_override
            else:
                try:
                    config_dict = dict(config_override)
                except Exception:
                    try:
                        if hasattr(config_override, "as_dict"):
                            config_dict = config_override.as_dict()
                        elif hasattr(config_override, "items"):
                            config_dict = {k: v for k, v in config_override.items()}
                    except Exception as e:
                        logger.warning("Could not convert config_override to dict: %s. Ignoring.", e)

        if config_dict:
            # Normalize booleans
            for k, v in list(config_dict.items()):
                if isinstance(v, str) and v.lower() in ("true", "false"):
                    config_dict[k] = v.lower() == "true"

            logger.info("Applying Sweep Overrides: %s", config_dict)
            local_args.update(config_dict)

        # Recompute classes/kwargs
        local_scheduler_class, local_scheduler_kwargs = parse_scheduler_args(
            local_args.get("scheduler_class"), local_args.get("scheduler_kwargs")
        )
        local_optimizer_class = get_optimizer_class(local_args.get("optimizer_class"))

        local_optimizer_kwargs = local_args.get("optimizer_kwargs", "")
        if isinstance(local_optimizer_kwargs, str):
            local_optimizer_kwargs = parse_optimizer_kwargs(local_optimizer_kwargs)
        if not local_optimizer_kwargs and optimizer_kwargs_cli:
            local_optimizer_kwargs = optimizer_kwargs_cli

        local_loss_kwargs = local_args.get("loss_kwargs", "")
        if isinstance(local_loss_kwargs, str):
            local_loss_kwargs = parse_optimizer_kwargs(local_loss_kwargs)
        if not local_loss_kwargs and loss_kwargs:
            local_loss_kwargs = loss_kwargs

        # Set Seeds
        seed = int(local_args.get("base_seed", 42)) + run_idx
        logger.info("=== Starting run %d (seed %d) ===", run_idx + 1, seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("Random seeds set and cudnn deterministic=%s benchmark=%s",
                     torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

        run_name = f"{local_args.get('experiment_name')}_run{run_idx+1}"

        # --- FIX: WANDB LOGGER SETUP ---
        wandb_logger = None
        if not local_args.get("no_wandb"):
            if active_run:
                # SWEEP MODE: Pass the existing run object.
                # Do NOT pass project/name, they are already set by the sweep controller.
                logger.info("Initializing WandB logger in SWEEP/agent mode for run %d", run_idx + 1)
                wandb_logger = WandbLogger(
                    experiment=active_run, 
                    log_model=True,
                    save_dir=logging_directory
                )
            else:
                logger.info("Initializing WandB logger (project=%s, name=%s)", local_args.get('project_name'), run_name)
                # NORMAL MODE: Create a new run.
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
            logger.info("EarlyStopping enabled (patience=%s, delta=%s)", local_args.get("early_stopping_patience"), local_args.get("early_stopping_delta"))

        logger.info("Instantiating model with num_hidden=%s, num_hidden_layers=%s, rnn_type=%s, lr=%s",
                    local_args.get("num_hidden"), local_args.get("num_hidden_layers"), local_args.get("rnn_type"), local_args.get("lr"))
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

        # Log model parameter counts
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Model instantiated: total_params=%d, trainable_params=%d", total_params, trainable)
        except Exception as e:
            logger.debug("Could not compute parameter counts: %s", e)

        if local_args.get("torch_compile"):
            try:
                logger.info("Attempting torch.compile on the model")
                model = torch.compile(model)
                logger.info("torch.compile succeeded")
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

        trainer = Trainer(
            max_epochs=int(local_args.get("max_epochs")),
            accelerator=accelerator,
            logger=wandb_logger if not local_args.get("no_wandb") else None,
            callbacks=callbacks,
            log_every_n_steps=int(local_args.get("log_every_n_steps")),
        )

        logger.info("Trainer created: max_epochs=%s, accelerator=%s, log_every_n_steps=%s, callbacks=%d",
                    trainer.max_epochs if hasattr(trainer, 'max_epochs') else local_args.get('max_epochs'),
                    accelerator, local_args.get('log_every_n_steps'), len(callbacks))

        logger.info("Starting training (run %d)", run_idx + 1)
        try:
            trainer.fit(model)
            logger.info("Training finished for run %d", run_idx + 1)
        except Exception as e:
            logger.error("Trainer.fit failed for run %d: %s", run_idx + 1, e)
            raise

        logger.info("Starting test for run %d using best checkpoint", run_idx + 1)
        try:
            test_results = trainer.test(model=model, ckpt_path="best")
            logger.info("Test results for run %d: %s", run_idx + 1, test_results)
        except Exception as e:
            logger.error("Trainer.test failed for run %d: %s", run_idx + 1, e)

        if wandb_logger and not local_args.get("no_wandb"):
            try:
                # Only finalize if we created it ourselves, but calling it safe mostly harmless
                if not active_run:
                    logger.info("Finalizing wandb logger for run %d", run_idx + 1)
                    wandb_logger.finalize("success")
            except Exception as e:
                logger.warning("wandb_logger.finalize() failed: %s", e)

        logger.info("Run %d finished.", run_idx + 1)

    # --- END run_single_run ---

    if args.wandb_sweep_enable:
        logger.info("WandB sweep mode enabled")
        if not args.wandb_sweep_yaml:
            logger.error("WandB sweep is enabled but no --wandb-sweep-yaml provided.")
            sys.exit(2)
        sweep_yaml_path = os.path.abspath(args.wandb_sweep_yaml)
        logger.info("Looking for sweep yaml at %s", sweep_yaml_path)
        if not os.path.exists(sweep_yaml_path):
            logger.error("YAML not found: %s", sweep_yaml_path)
            sys.exit(2)

        try:
            with open(sweep_yaml_path, "r") as f:
                sweep_config = yaml.safe_load(f)
            logger.info("Loaded sweep yaml with keys: %s", list(sweep_config.keys()) if isinstance(sweep_config, dict) else type(sweep_config))
        except Exception as e:
            logger.error("Failed to load yaml: %s", e)
            sys.exit(2)

        try:
            sweep_id = wandb.sweep(sweep_config, project=args.project_name)
            logger.info("Sweep ID: %s", sweep_id)
            print(f"Sweep URL: https://wandb.ai/{args.project_name}/sweeps/{sweep_id}")
        except Exception as e:
            logger.error("wandb.sweep() failed: %s", e)
            sys.exit(2)

        # --- FIX: AGENT RUN LOGIC ---
        def _agent_run():
            # Initialize the run via WandB (this gets the params)
            logger.info("Agent run initializing wandb.init()")
            run = wandb.init()
            try:
                # Grab the config
                cfg = wandb.config
                logger.info("Agent obtained cfg: %s", dict(cfg))
                # Pass the config AND the active run object to the trainer
                # We do NOT loop here. The agent calls this function repeatedly.
                run_single_run(run_idx=0, config_override=cfg, active_run=run)
            except Exception as e:
                logger.error("Agent run failed: %s", e)
                raise e
            finally:
                # Close the wandb run so the agent can start the next one
                logger.info("Agent finishing wandb.run")
                wandb.finish()

        count = int(args.wandb_sweep_count) if args.wandb_sweep_count and args.wandb_sweep_count > 0 else None
        logger.info("Starting agent... (count=%s)", count)
        try:
            wandb.agent(sweep_id, function=_agent_run, count=count)
        except Exception as e:
            logger.error("wandb.agent failed: %s", e)
            sys.exit(2)

    else:
        # Standard execution (no sweep)
        logger.info("Starting standard execution: num_runs=%d", args.num_runs)
        for run_idx in range(args.num_runs):
            run_single_run(run_idx, config_override=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNLightning model")
    default_max_epochs = 10

    parser.add_argument("--data", type=str, default="./data/v3_dataset.ds")
    parser.add_argument("--experiment-name", type=str, default="rnn_test")
    parser.add_argument("--project-name", type=str, default="Non-Spiking")
    parser.add_argument("--logging-directory", type=str, default=".temp")
    parser.add_argument("--checkpoint-path", type=str, default="models/RNN")
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--monitor-mode", dest="monitor_mode", choices=["min", "max"], default="min")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)

    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument("--num-hidden", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--num-inputs", dest="num_inputs", type=int, default=0)
    parser.add_argument("--num-outputs", type=int, default=0)
    parser.add_argument("--layer-skip", type=int, default=0)
    parser.add_argument("--use-layernorm", type=str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--max-epochs", type=int, default=default_max_epochs)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scheduler-class", type=str, default="cosine")
    parser.add_argument("--scheduler-kwargs", type=str, default=f"T_max={default_max_epochs}")
    parser.add_argument("--optimizer-class", type=str, default="adamw")
    parser.add_argument("--optimizer-kwargs", type=str, default="")
    parser.add_argument("--early-stopping", type=str2bool, default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4)
    parser.add_argument("--loss-fn", type=str, default="mse")
    parser.add_argument("--loss-kwargs", type=str, default="")

    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--use-gpu-if-available", action="store_true")

    parser.add_argument("--wandb-sweep-enable", action="store_true")
    parser.add_argument("--wandb-sweep-yaml", type=str, default="sweep_rnn.yaml")
    parser.add_argument("--wandb-sweep-count", type=int, default=0)

    args = parser.parse_args()

    # Conflict checking for sweep mode
    if args.wandb_sweep_enable:
        suspect_args = [
            "lr", "batch_size", "num_hidden", "num_hidden_layers", "dropout",
            "optimizer_class", "optimizer_kwargs", "scheduler_class", "scheduler_kwargs",
            "loss_fn", "loss_kwargs",
        ]
        conflicting = []
        for dest in suspect_args:
            cli_val = getattr(args, dest, None)
            default_val = parser.get_default(dest)
            if isinstance(default_val, str):
                is_conflict = (cli_val != default_val and cli_val != "" and cli_val is not None)
            else:
                is_conflict = (cli_val != default_val)
            if is_conflict:
                conflicting.append((dest, default_val, cli_val))

        if conflicting:
            msg_lines = [
                "Incompatible CLI arguments detected while WandB sweep mode is enabled.",
                "Please remove the following CLI flags or disable sweep mode:"
            ]
            for dest, default_val, cli_val in conflicting:
                msg_lines.append(f"  --{dest.replace('_', '-')} (default: {default_val!r})  -- provided: {cli_val!r}")
            logger.error("\n".join(msg_lines))
            sys.exit(2)

    main(args)