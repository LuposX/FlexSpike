#!/usr/bin/env python3
"""
Creates a wandb sweep from a YAML config and runs a wandb.agent
which executes the training function in-process.
This basically does a hyperparameetr search over possbile valeus as defiend in the sweep.yaml.

Usage:
    python 3_hyperparameter_search_rsnn.py --sweep-config sweep_rsnn.yaml --project test
"""

import argparse
import yaml
import inspect
import os
import sys

import wandb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import snntorch as snn

from utils.RSNN import SpikeSynth

import ast
from collections.abc import Mapping

# put this near the top of the file (with other imports)
SRC_ALLOWED_KEYS = {
    "alpha",
    "rho",
    "r",
    "rs",
    "z",
    "zhyp_s",
    "zdep_s",
    "bh_init",
    "bh_max",
    "detach_rec",
    "relu_bypass",
}

def _coerce_value(v):
    """Try to coerce a wandb-provided value (possibly string) into int/float/bool if meaningful."""
    # Already a sensible python type
    if isinstance(v, (int, float, bool)):
        return v
    if v is None:
        return None
    # Strings: try ast.literal_eval first (safe)
    if isinstance(v, str):
        s = v.strip()
        # Common boolean representations:
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            # literal_eval will convert numbers, tuples, lists, True/False, None
            return ast.literal_eval(s)
        except Exception:
            # fallback: try float then int
            try:
                if "." in s:
                    return float(s)
                return int(s)
            except Exception:
                return s  # give up, return original string
    # For other container types (e.g., wandb AttrDict), return as-is
    return v

def _to_plain_dict(m):
    """Convert mappings or wandb AttrDict-like objects to plain dict safely."""
    if isinstance(m, Mapping):
        try:
            return dict(m)
        except Exception:
            # fallback iterate
            return {k: m[k] for k in m}
    return m

def build_src_config_from_wandb(config):
    """
    Build an SRC_config dict from wandb.config.

    Accepts:
      - nested config.SRC_config (dict / AttrDict)
      - flattened keys like config.SRC_alpha, config.SRC_rho, ...
    Returns:
      dict with whitelisted keys only.
    """
    src = {}

    # 1) If there's a nested SRC_config use it (convert to plain dict/AttrDict)
    if hasattr(config, "SRC_config") and getattr(config, "SRC_config") is not None:
        nested = getattr(config, "SRC_config")
        nested = _to_plain_dict(nested)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k in SRC_ALLOWED_KEYS:
                    src[k] = _coerce_value(v)

    # 2) Also check for flattened keys like SRC_alpha
    for k in SRC_ALLOWED_KEYS:
        flatname = f"SRC_{k}"
        if hasattr(config, flatname):
            # Flattened takes precedence (overrides nested) if present
            src[k] = _coerce_value(getattr(config, flatname))

    # 3) If nothing found, return empty dict
    return src



def build_surrogate(surr_name, config):
    """Dynamically build a surrogate gradient callable from its name and sweep config."""
    if not hasattr(snn.surrogate, surr_name):
        raise ValueError(f"surrogate '{surr_name}' not found in snn.surrogate")
    surr_fn = getattr(snn.surrogate, surr_name)
    sig = inspect.signature(surr_fn)

    # Gather kwargs only for parameters the surrogate accepts and that exist in config
    kwargs = {}
    for pname in sig.parameters:
        if hasattr(config, pname):
            kwargs[pname] = getattr(config, pname)

    print(f"[build_surrogate] Building surrogate '{surr_name}' with kwargs: {kwargs}")
    return surr_fn(**kwargs)


def built_optimizer(config):
     # --- Build optimizer parameters ---
        optimizer_class = getattr(torch.optim, config.optimizer_class)

        # Extract Adam-specific parameters if present
        beta1 = getattr(config, "beta1", None)
        beta2 = getattr(config, "beta2", None)

        # also support nested 'adam_betas' dict if wandb flattens it differently
        if hasattr(config, "adam_betas"):
            betas_cfg = getattr(config, "adam_betas")
            if isinstance(betas_cfg, dict):
                beta1 = betas_cfg.get("beta1", beta1)
                beta2 = betas_cfg.get("beta2", beta2)

        # Fallbacks to AdamW defaults if not found
        if beta1 is None:
            beta1 = 0.9
        if beta2 is None:
            beta2 = 0.999

        eps = getattr(config, "eps", 1e-8)

        optimizer_kwargs = {
            "lr": config.lr,
            "betas": (beta1, beta2),
            "eps": eps,
        }
        return optimizer_class, optimizer_kwargs


def parse_scheduler_args(scheduler_name, scheduler_kwargs_str):
    scheduler_map = {
        "none": None,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_class = scheduler_map.get(scheduler_name.lower(), None)

    # Parse kwargs string into dict
    kwargs = {}
    if scheduler_kwargs_str:
        for kv in scheduler_kwargs_str.split(","):
            if "=" in kv:
                key, value = kv.split("=")
                # try to cast to int or float when possible
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                kwargs[key.strip()] = value

    return scheduler_class, kwargs

def parse_loss_kwargs(kwargs_str):
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



def training_run():
    """
    The function to be passed to wandb.agent. It will be called repeatedly by wandb.agent.
    Each call should create a wandb run (wandb.init()) and perform a single training run.
    """
    # Start a wandb run; wandb.agent will set the sweep config values into wandb.config
    run = wandb.init()
    try:
        config = wandb.config

        # Load dataset - same path as your original script
        data_path = config.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        data = torch.load(data_path)
        X_train, Y_train = data['X_train'], data['Y_train']
        X_valid, Y_valid = data['X_valid'], data['Y_valid']
        X_test, Y_test = data['X_test'], data['Y_test']

        train_dataset = TensorDataset(X_train, Y_train)
        valid_dataset = TensorDataset(X_valid, Y_valid)
        test_dataset  = TensorDataset(X_test, Y_test)

        # Build surrogate gradient
        spike_grad = build_surrogate(config.surrogate_gradient, config)

        scheduler_class, scheduler_kwargs = parse_scheduler_args(config.scheduler_class, config.scheduler_kwargs)

        optimizer_class, optimizer_kwargs = built_optimizer(config)

        loss_kwargs = parse_loss_kwargs(config.loss_kwargs)

        # Build SRC_config (handles nested and flattened forms)
        SRC_config = build_src_config_from_wandb(config)
        
        # If user selected SRC but no SRC_config set, that's ok: SRC will use defaults.
        # Optionally warn if neuron_type is SRC but src_config empty (informational)
        if config.get("neuron_type", None) == "SRC" and not SRC_config:
            # optional: keep this log to notice missing explicit SRC params
            print("[sweep] NOTE: running SRC neuron with default SRC params (SRC_config empty).")


        # Build model instance (mirror your original kwargs)
        model = SpikeSynth(
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            beta=config.beta,
            lr=config.lr,
            num_hidden=config.num_hidden,
            batch_size=config.batch_size,
            dropout=config.dropout,
            surrogate_gradient=spike_grad,
            num_hidden_layers=config.num_hidden_layers,
            use_bntt=config.use_bntt,
            bntt_time_steps=config.bntt_time_steps,
            neuron_type=config.neuron_type,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            max_epochs=config.epochs,
            temporal_skip=config.temporal_skip,
            layer_skip=config.layer_skip,
            log_every_n_steps=2,
            use_layernorm=config.use_layernorm,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            SRC_config=SRC_config,
            loss_fn=config.loss_fn,
            loss_kwargs=loss_kwargs
        )

        # Setup WandbLogger for PyTorch Lightning
        wandb_logger = WandbLogger(
            log_model=False,
            project=wandb.run.project or "Spike-Synth-rsnn",
            name=f"sweep_run_{wandb.run.id}"
        )

        # Define checkpoint callback
        checkpoint_dir = os.path.join(
            os.getenv("WANDB_DIR", "."),
            "checkpoints",
            wandb.run.project,
            wandb.run.sweep_id or "manual",
            wandb.run.id
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-val_loss{val_loss:.2f}",
            save_top_k=1,              # keep best 3 models
            monitor="val_loss",        # assumes your model logs 'val_loss'
            mode="min",
            save_last=True,
        )
        
        # Choose accelerator
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(
            max_epochs=config.epochs,
            logger=wandb_logger,
            accelerator=accelerator,
            callbacks=[checkpoint_callback]
        )

        # Fit
        trainer.fit(model)

    finally:
        # Ensure the run is closed
        wandb.finish()


def create_and_run_sweep(sweep_config_path: str, project: str, logging_directory : str, entity: str = None):
    # Set and Create logging directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging_directory = os.path.join(script_dir, logging_directory)
    logging_directory = os.path.abspath(logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory
    
    # Load sweep YAML
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Create sweep (returns sweep id)
    print(f"[sweep] Creating sweep for project='{project}', entity='{entity or wandb.Api().default_entity}'")
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    print(f"[sweep] Created sweep id: {sweep_id}")

    # Determine run cap (count) for agent
    count = None
    # common keys used: 'run_cap' in your YAML; also check nested
    if isinstance(sweep_config, dict):
        if "run_cap" in sweep_config:
            count = sweep_config["run_cap"]
        elif "run" in sweep_config and isinstance(sweep_config["run"], dict):
            count = sweep_config["run"].get("run_cap", None)

    print(f"[sweep] Starting wandb.agent with count={count} (None means run until stopped).")
    # Run the agent programmatically, using training_run as the function to execute for each
    wandb.agent(project + "/" + sweep_id if "/" not in str(sweep_id) else sweep_id,
                function=training_run,
                count=count)


def main():
    parser = argparse.ArgumentParser(description="Create and run a wandb sweep (combined).")
    parser.add_argument("--sweep-config", "-c", default="sweep_hyperparameter_rsnn.yaml",
                        help="Path to sweep YAML config (default: sweep_hyperparameter_rsnn.yaml)")
    parser.add_argument("--project", "-p", default="test", help="W&B project name (default: test)")
    parser.add_argument("--entity", "-e", default=None, help="W&B entity (user or team), optional")
    parser.add_argument("--logging-directory", "-l", default=".temp", help="Local path of the directory we log to.")
    args = parser.parse_args()

    # Make sure no stray run is left open
    try:
        wandb.finish()
    except Exception:
        pass

    create_and_run_sweep(args.sweep_config, args.project, args.logging_directory, args.entity)


if __name__ == "__main__":
    main()
