#!/usr/bin/env python3
"""
train_snn_runner.py

Single-file tool that provides two behaviours:

1) Training mode (default): runs the training loop over the parsed --datasets list in-process,
   and attempts a best-effort in-process cleanup between dataset runs.

2) Spawn-sequential mode (--spawn-sequential): expands the --datasets string into individual
   specs and runs THIS SCRIPT once per dataset sequentially (each dataset in a fresh process).
   The spawned child processes are started one after another; the wrapper waits for each to
   finish before starting the next. Child invocations are passed the same CLI args but with
   --no-spawn added to prevent nested spawning.

Usage examples:
  # default: run datasets in-process (with in-process cleanup between runs)
  python train_snn_runner.py --datasets "temporized:0-2, temporal:4" --epochs 50 --device gpu

  # wrapper mode: run each dataset in its own child process sequentially
  python train_snn_runner.py --spawn-sequential --datasets "temporized:0-2, temporal:4" --epochs 50 --device gpu

Notes:
 - If you use --spawn-sequential the wrapper will call the same script for each dataset but add --no-spawn
   to the child's argv to prevent recursive spawning.
 - The in-process cleanup (_free_memory) is still present and will run as a best-effort even for single-dataset
   runs. When running with the wrapper mode, each child will typically have only one dataset to process.
"""

import argparse
import pprint
import os
import sys
import subprocess
import time
import logging
import gc

import torch
import snntorch as snn  # keep as in original script; if unavailable the script will error early

from typing import List, Dict

# lightning and wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb  # used in cleanup fallback

# project-specific imports (these must be resolvable in your environment)
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


def parse_dataset_list(s: str) -> List[Dict]:
    """Parse dataset list string into [{'task': <str>|None, 'index': int}, ...]."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []

    result = []
    valid_tasks = {
        'normal': 'normal',
        'split': 'split',
        'temporized': 'temporized',
        'temporal': 'temporal',
        # aliases
        'temp': 'temporal',
        't': 'temporal',
        'tp': 'temporal',
        'tz': 'temporized',
    }

    raw_tokens = []
    for part in s.split(","):
        for token in part.split():
            token = token.strip()
            if token:
                raw_tokens.append(token)

    last_task = None
    for token in raw_tokens:
        if ":" in token:
            task_part, idx_part = token.split(":", 1)
            task_key = task_part.strip().lower()
            if task_key not in valid_tasks:
                raise argparse.ArgumentTypeError(f"Unknown dataset task '{task_part}' in token '{token}'. Valid: {', '.join(sorted(valid_tasks.keys()))}")
            task = valid_tasks[task_key]
            last_task = task
        else:
            task = last_task
            idx_part = token

        idx_part = idx_part.strip()
        if not idx_part:
            raise argparse.ArgumentTypeError(f"Missing index in token '{token}'")

        if "-" in idx_part:
            parts = idx_part.split("-")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"Invalid range token: '{idx_part}'")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid integer in range token: '{idx_part}'")
            if start <= end:
                rng = list(range(start, end + 1))
            else:
                rng = list(range(start, end - 1, -1))
            for i in rng:
                result.append({"task": task, "index": i})
        else:
            try:
                i = int(idx_part)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid dataset index token: '{idx_part}'")
            result.append({"task": task, "index": i})

    return result


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated floats into list[float]."""
    if s is None or s.strip() == "":
        return []
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float list: {s}. Error: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Printed Spiking Network (script + wrapper combined)")

    # Controls whether to spawn children per-dataset
    p.add_argument("--spawn-sequential", action="store_true",
                   help="If set, expand --datasets and run each dataset in its own subprocess sequentially (this script will spawn children).")
    p.add_argument("--no-spawn", action="store_true", help=argparse.SUPPRESS)  # internal: prevents recursive spawning

    # dataset / run basics
    p.add_argument("--dataset", type=int, default=0, help="Dataset index (used if --datasets not provided)")
    p.add_argument("--datasets", type=str, default="temporized:0-4, temporal:0,1,2,4,7,8,9,10,11,12",
                   help="Comma-/space-separated dataset specs (see README).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--timelimit", type=float, default=10.0)
    p.add_argument("--compute-roc", type=bool, default=True)
    p.add_argument("--batch-size", type=int, default=None, help="Default batch size for train/valid/test (overridden by mode-specific flags).")

    # new MC + fault args
    p.add_argument("--mc-samples", type=int, default=4)
    p.add_argument("--eval-mc-samples", type=int, default=None)
    p.add_argument("--use-interpolation", action="store_true")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--test-fault-modes", type=str, default="none,single")

    # static params
    p.add_argument("--num-static-param", type=str, default="4,0")
    p.add_argument("--min-static-main", type=str, default=torch.tensor([0.0, 0.1, 0.15, 0.5]))
    p.add_argument("--max-static-main", type=str, default=torch.tensor([1.0, 1.0, 1.0, 1.0]))
    p.add_argument("--min-static-faulty", type=str, default=None)
    p.add_argument("--max-static-faulty", type=str, default=None)

    # optimizer / lr
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr-min", type=float, default=1e-6)

    # model topology
    p.add_argument("--hidden", type=int, nargs="*", default=None)
    p.add_argument("--surrogate-class", type=str, choices=["baseline-gpt", "spiking", "non-spiking"], default="spiking")

    # logging / checkpoint
    p.add_argument("--experiment", type=str, default="test")
    p.add_argument("--project", type=str, default="Spike-Synth-Full")
    p.add_argument("--log-dir", type=str, default=".temp")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--surrogate-ckpt", type=str, default="surrogate/models/Spiking/LeakyParallel/RSNN_wLMSE-runrun_idx=0-epoch=78-val_loss=0.07.ckpt.ckpt")

    # runtime flags
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p.add_argument("--test-fault-levels", type=str, default=None)
    p.add_argument("--faulty-surrogates", type=str, default="")
    p.add_argument("--faulty-static-values", type=parse_float_list, default=[])

    p.add_argument("--num-runs", type=int, default=1, help="Number of repeated independent training runs per dataset (increments seed each run).")
    p.add_argument("--train-with-faults", action="store_true", help="Enable fault-aware training (faults active during training).")
    
    return p.parse_args()


# ------------------------------
# In-process memory cleanup helper
# ------------------------------
def _free_memory(psnn=None, trainer=None, wandb_logger=None, train_loader=None, valid_loader=None, test_loader=None, checkpoint_callback=None, accelerator="cpu"):
    logger.info("Attempting in-process memory cleanup...")
    # finish wandb
    try:
        if wandb_logger is not None and getattr(wandb_logger, "experiment", None):
            try:
                wandb_logger.experiment.finish()
            except Exception:
                try:
                    wandb.finish()
                except Exception:
                    pass
    except Exception:
        pass

    # delete major references
    for name, obj in (("psnn", psnn), ("trainer", trainer), ("wandb_logger", wandb_logger), ("checkpoint_callback", checkpoint_callback)):
        try:
            if obj is not None:
                del obj
        except Exception:
            pass

    # delete dataloaders
    for obj in (train_loader, valid_loader, test_loader):
        try:
            if obj is not None:
                del obj
        except Exception:
            pass

    # run GC
    try:
        gc.collect()
    except Exception:
        pass

    # clear CUDA caches
    try:
        if accelerator == "gpu" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    logger.info("In-process memory cleanup requested (best-effort).")


# ------------------------------
# Training runner (almost verbatim from original)
# ------------------------------
def run_train(args_cli: argparse.Namespace):
    logger.info("Starting training runner")

    # parse dataset list (CLI)
    if args_cli.datasets:
        try:
            dataset_specs = parse_dataset_list(args_cli.datasets)
            if not dataset_specs:
                raise ValueError("Parsed --datasets is empty.")
        except Exception as e:
            logger.exception("Failed parsing --datasets '%s': %s", args_cli.datasets, e)
            raise
    else:
        dataset_specs = [{"task": None, "index": args_cli.dataset}]

    if args_cli.checkpoint_dir is None:
        args_cli.checkpoint_dir = f"models/FullNetwork/{args_cli.surrogate_class}"
    base_checkpoint_dir = args_cli.checkpoint_dir

    logger.info("Will run dataset specs in sequence: %s", dataset_specs)
    logger.info("Faulty static values: %s", args_cli.faulty_static_values)
    logger.info("Fault aware methodolgy: %s", args_cli.train_with_faults)

    for spec in dataset_specs:
        task_for_spec = spec.get("task")
        dset = spec.get("index")
        logger.info("=== Starting runs for dataset spec %s (task=%s) ===", dset, task_for_spec)

        # Capture the base seed for this dataset so each dataset restarts seed at args_cli.seed
        base_seed = int(args_cli.seed)

        # Run multiple independent runs per dataset (increment seed per run, but reset for each dataset)
        for run_idx in range(1, max(1, args_cli.num_runs) + 1):
            logger.info(">>> Dataset %s (task=%s) - run %d/%d starting", dset, task_for_spec, run_idx, args_cli.num_runs)

            # compute seed for this run (increment from base_seed, but base_seed is fixed per dataset)
            seed_for_run = base_seed + (run_idx - 1)

            overrides = {
                "DATASET": dset,
                "SEED": seed_for_run,
                "projectname": "pLR-SNN",
                "DEVICE": args_cli.device,
                "PROGRESSIVE": args_cli.progressive,
                "EPOCH": args_cli.epochs,
                "TIMELIMITATION": args_cli.timelimit,
                "LR_MIN": args_cli.lr_min if hasattr(args_cli, "lr_min") else args_cli.lr_min,
                "LR": args_cli.lr,
            }
            if task_for_spec is not None:
                overrides["TASK"] = task_for_spec
                overrides["task"] = task_for_spec

            logger.debug("Configuration overrides for run %d: %s", run_idx, overrides)

            try:
                args = load_args(overrides=overrides)
                logger.info("Loaded configuration via load_args (dataset=%s task=%s run=%d seed=%s)", dset, getattr(args, "task", None), run_idx, seed_for_run)
            except Exception as e:
                logger.exception("Failed to load configuration for dataset %s (run %d): %s", dset, run_idx, e)
                if not args_cli.stop_on_error:
                    raise
                else:
                    logger.warning("Continuing to next run due to --stop-on-error=False")
                    continue

            if args_cli.hidden:
                args.hidden = list(args_cli.hidden)
                logger.info("Overrode hidden topology from CLI: %s", args.hidden)
            else:
                logger.info("Using hidden topology from config: %s", getattr(args, "hidden", None))

            try:
                args = FormulateArgs(args)
                logger.info("Formulated args successfully (dataset=%s task=%s run=%d)", dset, getattr(args, "task", None), run_idx)
            except Exception as e:
                logger.exception("FormulateArgs failed for dataset %s (run %d): %s", dset, run_idx, e)
                if not args_cli.stop_on_error:
                    raise
                else:
                    logger.warning("Continuing to next run due to --stop-on-error=False")
                    continue


            try:
                SetSeed(args.SEED)
                logger.info("Set random seed to %s (dataset=%s run=%d)", args.SEED, dset, run_idx)
            except Exception:
                logger.warning("SetSeed failed; continuing without explicit seed set.")

            try:
                train_loader, datainfo = GetDataLoader(args, 'train', batch_size=args_cli.batch_size)
                valid_loader, _ = GetDataLoader(args, 'valid', batch_size=args_cli.batch_size)
                test_loader, _ = GetDataLoader(args, 'test', batch_size=args_cli.batch_size)
                logger.info("Data loaders created successfully (dataset=%s task=%s run=%d)", dset, getattr(args, "task", None), run_idx)
            except Exception as e:
                logger.exception("Failed creating data loaders for dataset %s (run %d): %s", dset, run_idx, e)
                if not args_cli.stop_on_error:
                    raise
                else:
                    logger.warning("Continuing to next run due to --stop-on-error=False")
                    continue

            logger.info("Data information (dataset=%s task=%s run=%d):\n%s", dset, getattr(args, "task", None), run_idx, pprint.pformat(datainfo))

            # logging directory per dataset and per run
            script_dir = os.getcwd()
            logging_directory = os.path.join(script_dir, args_cli.log_dir, f"dataset_{getattr(args, 'task', 'auto')}_{dset}", f"run_{run_idx}")
            logging_directory = os.path.abspath(logging_directory)
            os.makedirs(logging_directory, exist_ok=True)
            os.environ["WANDB_DIR"] = logging_directory
            logger.info("Logging directory is set to %s (WANDB_DIR) (dataset=%s run=%d)", logging_directory, dset, run_idx)

            # surrogate class selection
            if args_cli.surrogate_class == "spiking":
                surrogate_class = SpikingNetwork
            elif args_cli.surrogate_class == "baseline-gpt":
                surrogate_class = GPTLightning
            else:
                surrogate_class = NonSpikingNetwork

            faulty_ckpts_list = [p.strip() for p in args_cli.faulty_surrogates.split(",") if p.strip()] if args_cli.faulty_surrogates else []
            faulty_static_values = args_cli.faulty_static_values

            if args_cli.test_fault_modes:
                try:
                    test_fault_modes = [s.strip() for s in args_cli.test_fault_modes.split(",") if s.strip()]
                except Exception:
                    logger.exception("Could not parse --test-fault-modes '%s'", args_cli.test_fault_modes)
                    test_fault_modes = ["none", "single"]
            else:
                test_fault_modes = ["none", "single"]
            setattr(args, "test_fault_modes", test_fault_modes)

            # static param parsing helpers
            def _csv_to_floats(s):
                if s is None:
                    return None
                if isinstance(s, torch.Tensor):
                    return [float(x) for x in s.detach().cpu().view(-1).tolist()]
                if isinstance(s, (list, tuple)):
                    return [float(x) for x in s]
                if isinstance(s, str):
                    parts = [tok.strip() for tok in s.split(",") if tok.strip()]
                    return [float(x) for x in parts]
                if isinstance(s, (int, float)):
                    return [float(s)]
                raise TypeError(f"Unsupported type for static-param vector: {type(s)}. Value: {s}")

            def _maybe_tensor_from_input(inp, expected_len: int = None):
                if inp is None:
                    return None
                vals = _csv_to_floats(inp)
                t = torch.tensor(vals, dtype=torch.float32)
                if expected_len is not None and t.numel() != expected_len:
                    raise ValueError(f"Expected length {expected_len} but got {t.numel()} for vector {inp!r}")
                return t

            def _broadcast_or_trim(t_src: torch.Tensor, target_len: int) -> torch.Tensor:
                if t_src is None:
                    return torch.zeros(target_len, dtype=torch.float32)
                src = t_src.clone().detach().view(-1).float().cpu()
                src_len = int(src.numel())
                if src_len == target_len:
                    return src
                if src_len < target_len:
                    if src_len == 0:
                        return torch.zeros(target_len, dtype=src.dtype)
                    last = float(src[-1].item())
                    extra = torch.tensor([last] * (target_len - src_len), dtype=src.dtype)
                    return torch.cat([src, extra], dim=0)
                return src[:target_len]

            # parse num-static-param
            if isinstance(args_cli.num_static_param, str) and "," in args_cli.num_static_param:
                parts = [int(x.strip()) for x in args_cli.num_static_param.split(",")]
                if len(parts) != 2:
                    raise ValueError("--num-static-param must be a single int or two ints separated by a comma (main,faulty)")
                num_main, num_faulty = parts
                num_static_param_arg = (num_main, num_faulty)
            elif isinstance(args_cli.num_static_param, (list, tuple, torch.Tensor)):
                if isinstance(args_cli.num_static_param, torch.Tensor):
                    arr = args_cli.num_static_param.detach().cpu().view(-1).tolist()
                    parts = [int(x) for x in arr]
                else:
                    parts = [int(x) for x in args_cli.num_static_param]
                if len(parts) == 1:
                    num_static_param_arg = parts[0]
                elif len(parts) == 2:
                    num_static_param_arg = (parts[0], parts[1])
                else:
                    raise ValueError("--num-static-param as list/tuple must have length 1 or 2 (main[,faulty])")
            else:
                num_static_param_arg = int(args_cli.num_static_param)

            if isinstance(num_static_param_arg, tuple):
                num_main, num_faulty = num_static_param_arg
            else:
                num_main = num_static_param_arg
                num_faulty = num_main

            t_min_main = _maybe_tensor_from_input(args_cli.min_static_main, expected_len=None)
            t_max_main = _maybe_tensor_from_input(args_cli.max_static_main, expected_len=None)

            if t_min_main is None:
                t_min_main = torch.zeros(num_main, dtype=torch.float32)
            if t_max_main is None:
                t_max_main = torch.ones(num_main, dtype=torch.float32)

            if t_min_main.numel() != num_main:
                t_min_main = _broadcast_or_trim(t_min_main, num_main)
            if t_max_main.numel() != num_main:
                t_max_main = _broadcast_or_trim(t_max_main, num_main)

            t_min_faulty = _maybe_tensor_from_input(args_cli.min_static_faulty, expected_len=None)
            t_max_faulty = _maybe_tensor_from_input(args_cli.max_static_faulty, expected_len=None)

            if t_min_faulty is None:
                t_min_faulty = _broadcast_or_trim(t_min_main, num_faulty)
            else:
                if t_min_faulty.numel() != num_faulty:
                    t_min_faulty = _broadcast_or_trim(t_min_faulty, num_faulty)

            if t_max_faulty is None:
                t_max_faulty = _broadcast_or_trim(t_max_main, num_faulty)
            else:
                if t_max_faulty.numel() != num_faulty:
                    t_max_faulty = _broadcast_or_trim(t_max_faulty, num_faulty)

            if isinstance(num_static_param_arg, tuple):
                min_value_static_params_arg = (t_min_main, t_min_faulty)
                max_value_static_params_arg = (t_max_main, t_max_faulty)
            else:
                min_value_static_params_arg = t_min_main
                max_value_static_params_arg = t_max_main

            logger.debug("Static params parsed: num_main=%s num_faulty=%s", num_main, num_faulty)
            logger.debug("min_main=%s max_main=%s", t_min_main.tolist(), t_max_main.tolist())
            logger.debug("min_faulty=%s max_faulty=%s", t_min_faulty.tolist(), t_max_faulty.tolist())

            surrogate_ckpt = args_cli.surrogate_ckpt
            hidden_list = args.hidden if getattr(args, "hidden", None) else []
            topology = [datainfo['N_feature']] + hidden_list + [datainfo['N_class']]
            logger.info("Instantiating PrintedSpikingNetwork with topology: %s (dataset=%s task=%s run=%d)", topology, dset, getattr(args, "task", None), run_idx)

            try:
                psnn = pSNN.LightningPrintedSpikingNetwork(
                    topology=topology,
                    args=args,
                    model_class=surrogate_class,
                    ckpt_path=surrogate_ckpt,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    surrogate_gradient=snn.surrogate.atan(),
                    num_static_param=num_static_param_arg,
                    min_value_static_params=min_value_static_params_arg,
                    max_value_static_params=max_value_static_params_arg,
                    faulty_ckpt_paths=faulty_ckpts_list,
                    faulty_static_values=faulty_static_values,
                    mc_samples=args_cli.mc_samples,
                    use_interpolation=args_cli.use_interpolation,
                    warmup_epochs=args_cli.warmup_epochs,
                    enable_faults_during_training=args_cli.train_with_faults,
                )
                logger.info("PrintedSpikingNetwork instantiated (surrogate_ckpt=%s, surrogate_class=%s, dataset=%s task=%s run=%d)",
                            surrogate_ckpt, args_cli.surrogate_class, dset, getattr(args, "task", None), run_idx)
                logger.info("Dynamic faulty neurons: %d, Static faulty neurons: %d",
                           len(faulty_ckpts_list), len(faulty_static_values))
            except Exception as e:
                logger.exception("Failed to instantiate PrintedSpikingNetwork for dataset %s (run %d): %s", dset, run_idx, e)
                if not args_cli.stop_on_error:
                    raise
                else:
                    logger.warning("Continuing to next run due to --stop-on-error=False")
                    continue

            run_name_base = f"{args_cli.experiment}_{datainfo['dataname']}"
            run_name = f"{run_name_base}_run{run_idx}"
            logger.info("Setting up WandB logger (project=%s, run=%s) (dataset=%s task=%s run=%d)", args_cli.project, run_name, dset, getattr(args, "task", None), run_idx)
            wandb_logger = WandbLogger(log_model=True, project=args_cli.project, name=run_name, save_dir=logging_directory)

            try:
                wandb_logger.watch(psnn)
                wandb_logger.experiment.log_code(".", include_fn=lambda path: path.endswith('.py') or path.endswith('.ipynb'))
                logger.info("WandB watch and log_code succeeded (dataset=%s run=%d)", dset, run_idx)
            except Exception as e:
                logger.warning("wandb watch/log_code failed for dataset %s (run %d): %s", dset, run_idx, e)

            # make checkpoint dir per dataset and run to avoid overwrites between runs
            dataset_checkpoint_dir = os.path.join(base_checkpoint_dir, f"dataset_{getattr(args, 'task', 'auto')}_{dset}", f"run_{run_idx}")
            os.makedirs(dataset_checkpoint_dir, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=dataset_checkpoint_dir,
                filename=f"{args_cli.experiment}-{args_cli.surrogate_class}-ds{getattr(args, 'task', 'auto')}{dset}-run{run_idx}" + "-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
            )
            logger.info("ModelCheckpoint configured: dir=%s filename=%s (dataset=%s task=%s run=%d)", dataset_checkpoint_dir, checkpoint_callback.filename, dset, getattr(args, "task", None), run_idx)

            accelerator = "cpu"
            if args_cli.device == "gpu" and torch.cuda.is_available():
                accelerator = "gpu"
            logger.info("Using accelerator: %s (dataset=%s task=%s run=%d)", accelerator, dset, getattr(args, "task", None), run_idx)

            trainer = Trainer(
                fast_dev_run=args_cli.fast_dev_run,
                max_epochs=args_cli.epochs,
                logger=wandb_logger,
                accelerator=accelerator,
                callbacks=[checkpoint_callback],
            )
            logger.info("PyTorch Lightning Trainer instantiated (fast_dev_run=%s, max_epochs=%d) (dataset=%s task=%s run=%d)",
                        args_cli.fast_dev_run, args_cli.epochs, dset, getattr(args, "task", None), run_idx)

            # TRAIN
            try:
                logger.info("Starting training for %d epochs (dataset=%s task=%s run=%d)", args_cli.epochs, dset, getattr(args, "task", None), run_idx)
                trainer.fit(psnn)
                logger.info("Training finished successfully (dataset=%s task=%s run=%d)", dset, getattr(args, "task", None), run_idx)
            except Exception as e:
                logger.exception("Training failed for dataset %s (run %d) with exception: %s", dset, run_idx, e)
                try:
                    if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment:
                        wandb_logger.experiment.finish()
                        logger.info("WandB experiment finished (dataset=%s run=%d)", dset, run_idx)
                except Exception as e2:
                    logger.warning("wandb_logger.finalize() failed after training error for dataset %s (run %d): %s", dset, run_idx, e2)

                if not args_cli.stop_on_error:
                    raise
                else:
                    logger.warning("Continuing to next run due to --stop-on-error=False")
                    if accelerator == "gpu":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    _free_memory(psnn=psnn, trainer=trainer, wandb_logger=wandb_logger, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, checkpoint_callback=checkpoint_callback, accelerator=accelerator)
                    continue

            # TEST
            logger.info("Starting test using best checkpoint (dataset=%s task=%s run=%d)", dset, getattr(args, "task", None), run_idx)
            try:
                trainer.test(model=psnn, ckpt_path="best")
                logger.info("Test finished. Best checkpoint: %s (dataset=%s task=%s run=%d)", checkpoint_callback.best_model_path or "N/A", dset, getattr(args, "task", None), run_idx)
            except Exception as e:
                logger.exception("Testing failed for dataset %s (run %d): %s", dset, run_idx, e)
                if not args_cli.stop_on_error:
                    raise
                else:
                    logger.warning("Continuing to next run due to --stop-on-error=False")

            # finalize wandb
            try:
                if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment:
                    wandb_logger.experiment.finish()
                    logger.info("WandB experiment finished (dataset=%s task=%s run=%d)", dset, getattr(args, "task", None), run_idx)
            except Exception as e:
                logger.warning("wandb_logger.finalize() failed for dataset %s (run %d): %s", dset, run_idx, e)

            # cleanup
            try:
                _free_memory(psnn=psnn, trainer=trainer, wandb_logger=wandb_logger,
                             train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                             checkpoint_callback=checkpoint_callback, accelerator=accelerator)
                logger.info("Requested memory cleanup (dataset=%s run=%d)", dset, run_idx)
            except Exception as e:
                logger.warning("Memory cleanup failed for dataset %s (run %d): %s", dset, run_idx, e)

            if accelerator == "gpu":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            logger.info("=== Completed run %d for dataset %s (task=%s) ===", run_idx, dset, getattr(args, "task", None))

        # end runs for this dataset
        logger.info("Completed all %d runs for dataset %s (task=%s)", args_cli.num_runs, dset, task_for_spec)

    logger.info("run_train finished for all dataset specs")



# ------------------------------
# Wrapper logic: spawn children sequentially
# ------------------------------
def _strip_and_build_child_argv(original_argv: List[str], dataset_token: str) -> List[str]:
    """
    Build a child argv list from original_argv (sys.argv[1:]) by:
      - removing --spawn-sequential (if present)
      - removing --datasets (and its value) or --datasets=...
      - removing any --no-spawn (if present)
      - adding --datasets <dataset_token>
      - adding --no-spawn
    Returns argv suitable to pass to subprocess (excluding the interpreter).
    """
    lst = list(original_argv)[:]  # copy

    # Helper to remove key possibly in two forms: "--key" followed by value, or "--key=value"
    def remove_key(key: str, arr: List[str]):
        i = 0
        while i < len(arr):
            el = arr[i]
            if el == key:
                # remove this and next token (value) if present and doesn't start with '-'
                del arr[i]
                if i < len(arr) and not arr[i].startswith("-"):
                    del arr[i]
                continue
            if el.startswith(key + "="):
                del arr[i]
                continue
            i += 1

    remove_key("--spawn-sequential", lst)
    remove_key("--no-spawn", lst)
    remove_key("--datasets", lst)

    # If user supplied a bare positional dataset token (unlikely), we won't tamper with it.

    # Now append the desired dataset and --no-spawn
    lst += ["--datasets", dataset_token, "--no-spawn"]
    return lst


def run_wrapper_and_spawn(args: argparse.Namespace):
    datasets = parse_dataset_list(args.datasets)
    if not datasets:
        print("No datasets parsed from:", args.datasets)
        sys.exit(1)

    script_path = os.path.abspath(sys.argv[0])
    if not os.path.isfile(script_path):
        print(f"Script path not found: {script_path}")
        sys.exit(2)

    original_argv = sys.argv[1:]
    for spec in datasets:
        task = spec.get("task")
        idx = spec["index"]
        dataset_token = f"{task}:{idx}" if task is not None else str(idx)

        child_argv = _strip_and_build_child_argv(original_argv, dataset_token)
        cmd = [sys.executable, script_path] + child_argv
        print("Running child:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"Child process for dataset {dataset_token} exited with code {proc.returncode}")
            print("Stopping wrapper due to child error.")
            sys.exit(proc.returncode)
        else:
            print(f"Finished dataset {dataset_token} successfully. Starting next (if any).")

    print("All dataset runs completed successfully.")


# ------------------------------
# Entry point
# ------------------------------
def main():
    args = parse_args()

    # If wrapper requested and not suppressed, run wrapper (spawn children sequentially)
    if args.spawn_sequential and not args.no_spawn:
        logger.info("Running in spawn-sequential mode (each dataset will be run in its own subprocess sequentially).")
        run_wrapper_and_spawn(args)
        return

    # Otherwise run the in-process trainer which will iterate dataset_specs (maybe 1)
    run_train(args)


if __name__ == "__main__":
    main()