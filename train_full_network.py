#!/usr/bin/env python3
"""
train_snn.py

Usage examples:
  python train_snn.py --epochs 50 --device gpu --experiment myrun --project Spike-Synth-Full \
      --checkpoint-dir models/SRNN --surrogate-ckpt surrogate/models/SRNN/testspike_model-epoch=09-val_loss=0.09-v3.ckpt \
      --hidden 128 64

  # multiple datasets across lists:
  python train_snn.py --datasets "temporized:0, temporal:2, normal:5" --experiment multi --project Spike-Synth-Full

  # ranges:
  python train_snn.py --datasets "temporized:0-2, temporal:4-5" --experiment multi

See --help for all options.
"""

import pprint
import os
import time
import logging
import torch
import snntorch as snn
import argparse
from typing import List, Dict

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


def parse_dataset_list(s: str) -> List[Dict]:
    """
    Parse a dataset list string supporting explicit task indexes.

    Supported token forms (comma- or space-separated, mixed):
      - "3"                     -> {'task': None, 'index': 3} (uses config/default task)
      - "temporized:3"          -> {'task': 'temporized', 'index': 3}
      - "temporal:2-5"          -> expands to indexes 2,3,4,5 for task 'temporal'
      - "normal:0, temporized:1-3, 7"
      - "temporal:0,1,2,4,7"    -> `temporal` applies to all subsequent numeric tokens until a new task appears

    Returns a list of dicts: [{'task': <str>|None, 'index': int}, ...]
    Valid tasks: normal, split, temporized, temporal (case-insensitive).
    """
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
        # common aliases
        'temp': 'temporal',
        't': 'temporal',
        'tp': 'temporal',
        'tz': 'temporized',
    }

    # split on commas and whitespace, preserve order
    raw_tokens = []
    for part in s.split(","):
        for token in part.split():
            token = token.strip()
            if token:
                raw_tokens.append(token)

    last_task = None  # remember the most recent explicit task
    for token in raw_tokens:
        # token may be like "task:index" or just "index"
        if ":" in token:
            task_part, idx_part = token.split(":", 1)
            task_key = task_part.strip().lower()
            if task_key not in valid_tasks:
                raise argparse.ArgumentTypeError(
                    f"Unknown dataset task '{task_part}' in token '{token}'. "
                    f"Valid tasks: {', '.join(sorted(set(valid_tasks.keys())))}"
                )
            task = valid_tasks[task_key]
            last_task = task  # update remembered task
        else:
            # no explicit task on this token -> inherit last_task (may be None)
            task = last_task
            idx_part = token

        idx_part = idx_part.strip()
        if not idx_part:
            raise argparse.ArgumentTypeError(f"Missing index in token '{token}'")

        # support ranges like 2-5 or single ints
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


def parse_args():
    p = argparse.ArgumentParser(description="Train a Printed Spiking Network (from notebook->script)")

    # dataset / run basics
    p.add_argument("--dataset", type=int, default=0, help="Dataset index. See utils.Loader.py (used if --datasets not provided)")
    p.add_argument("--datasets", type=str, default="temporized:0-4, temporal:0,1,2,4,7,8,9,10,11,12",
                   help=("Comma- or space-separated dataset specs. "
                         "Each spec can be an integer (uses config default task) or 'task:index' or 'task:start-end'. "
                         "Tasks: normal, split, temporized, temporal. Examples: "
                         "'0,5,8', 'temporized:0-2, temporal:4', 'normal:3, 7'"))
    p.add_argument("--seed", type=int, default=42, help="SEED value")
    p.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use")
    p.add_argument("--epochs", type=int, default=200, help="Number of training epochs (overrides config EPOCH)")
    p.add_argument("--timelimit", type=float, default=10, help="TIMELIMITATION value")
        # static-parameter specification (main vs faulty)
    p.add_argument(
        "--num-static-param",
        type=str,
        default="4,0",
        help=(
            "Number of static params. Either single int '4' (used for both main+faulty) "
            "or pair '4,6' meaning main=4,faulty=6."
        ),
    )
    p.add_argument(
        "--min-static-main",
        type=str,
        default=torch.tensor([0.0, 0.1, 0.15, 0.5]),
        help="Comma-separated list of minimum static-param values for the main surrogate (e.g. '0.0,0.1,0.2'). "
             "If omitted, defaults to zeros of length num-static-param (main).",
    )
    p.add_argument(
        "--max-static-main",
        type=str,
        default=torch.tensor([1.0, 1.0,  1.0, 1.0]),
        help="Comma-separated list of maximum static-param values for the main surrogate. If omitted, defaults to ones.",
    )
    p.add_argument(
        "--min-static-faulty",
        type=str,
        default=None,
        help="Comma-separated list of minimum static-param values for the faulty surrogate. If omitted, main values are reused (expanded/truncated if lengths differ).",
    )
    p.add_argument(
        "--max-static-faulty",
        type=str,
        default=None,
        help="Comma-separated list of maximum static-param values for the faulty surrogate. If omitted, main values are reused (expanded/truncated if lengths differ).",
    )


    # optimizer / lr
    p.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    p.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")

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

    # Fault-aware training with optional gradual warm-up
    p.add_argument("--fault-prob", type=float, default=0.0,
                   help="Legacy flag: base fault probability. If --max-fault-prob is not set, this value is used as the target after warm-up.")
    p.add_argument("--max-fault-prob", type=float, default=None,
                   help="Maximum/target fault probability after warm-up phase. If not set, defaults to --fault-prob value.")
    p.add_argument("--fault-warmup-epochs", type=int, default=30,
                   help="Number of epochs to train with fault_prob = 0.0 before starting ramp-up. Default: 20")
    p.add_argument("--fault-ramp-epochs", type=int, default=50,
                   help="Number of epochs over which to linearly ramp fault_prob from 0 to max_fault_prob. Default: 50")
    # Test-time sweep
    p.add_argument("--test-fault-levels", type=str, default=None,
                   help="Comma-separated fault probabilities to evaluate at test time (e.g. '0.0,0.05,0.1'). If not given, defaults to [0.0, training_fault_prob].")
    # --------------------------------------------------------------------

    return p.parse_args()


def main():
    logger.info("Starting train_snn.py")

    args_cli = parse_args()

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
        # keep old behaviour: single dataset using CLI --dataset and default task from config
        dataset_specs = [{"task": None, "index": args_cli.dataset}]

    # default checkpoint dir if not provided
    if args_cli.checkpoint_dir is None:
        args_cli.checkpoint_dir = f"models/FullNetwork/{args_cli.surrogate_class}"
    base_checkpoint_dir = args_cli.checkpoint_dir

    logger.info("Will run dataset specs in sequence: %s", dataset_specs)

    # We'll process each dataset spec sequentially
    for spec in dataset_specs:
        task_for_spec = spec.get("task")  # may be None -> use config default
        dset = spec.get("index")
        logger.info("=== Starting run for dataset spec %s (task=%s) ===", dset, task_for_spec)

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

        # If token explicitly specified a task, tell load_args to use it via overrides.
        if task_for_spec is not None:
            overrides["TASK"] = task_for_spec
            overrides["task"] = task_for_spec

        logger.debug("Configuration overrides for dataset %s (task=%s): %s", dset, task_for_spec, overrides)

        # Load base args using the project's configuration loader
        try:
            args = load_args(overrides=overrides)
            logger.info("Loaded configuration via load_args (dataset=%s task=%s)", dset, getattr(args, "task", None))
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
            logger.info("Formulated args successfully (dataset=%s task=%s)", dset, getattr(args, "task", None))
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
            # GetDataLoader expects args.DATASET and args.task to be set on args
            # load_args + FormulateArgs should have set args.DATASET and args.task already based on overrides
            train_loader, datainfo = GetDataLoader(args, 'train')
            valid_loader, _ = GetDataLoader(args, 'valid')
            test_loader, _ = GetDataLoader(args, 'test')
            
            logger.info("Data loaders created successfully (dataset=%s task=%s)", dset, getattr(args, "task", None))
        except Exception as e:
            logger.exception("Failed creating data loaders for dataset %s: %s", dset, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                continue

        logger.info("Data information (dataset=%s task=%s):\n%s", dset, getattr(args, "task", None), pprint.pformat(datainfo))

        # prepare logging directory (make separate directories per dataset to avoid clobber)
        script_dir = os.getcwd()
        logging_directory = os.path.join(script_dir, args_cli.log_dir, f"dataset_{getattr(args, 'task', 'auto')}_{dset}")
        logging_directory = os.path.abspath(logging_directory)
        os.makedirs(logging_directory, exist_ok=True)
        os.environ["WANDB_DIR"] = logging_directory
        logger.info("Logging directory is set to %s (WANDB_DIR) (dataset=%s task=%s)", logging_directory, dset, getattr(args, "task", None))

        # select surrogate class
        if args_cli.surrogate_class == "spiking":
            surrogate_class = SpikingNetwork
        elif args_cli.surrogate_class == "baseline-gpt":
            surrogate_class = GPTLightning
        else:
            surrogate_class = NonSpikingNetwork

        if args_cli.faulty_surrogates:
            faulty_ckpts_list = [p.strip() for p in args_cli.faulty_surrogates.split(",") if p.strip()]
        else:
            faulty_ckpts_list = []

        # parse test fault-levels as list of floats if provided
        if args_cli.test_fault_levels:
            try:
                test_fault_levels = sorted({float(x.strip()) for x in args_cli.test_fault_levels.split(",") if x.strip()})
            except ValueError:
                logger.exception("Could not parse --test-fault-levels '%s'", args_cli.test_fault_levels)
                test_fault_levels = None
        else:
            test_fault_levels = None

        # place parsed test levels into args for the model to read (pSNN.UpdateArgs / test sweep expects args.test_fault_levels)
        if test_fault_levels is not None:
            setattr(args, "test_fault_levels", test_fault_levels)

         # Determine the actual max fault probability (priority: --max-fault-prob > --fault-prob > 0.0)
        effective_max_fault_prob = args_cli.fault_prob  # default fallback
        if args_cli.max_fault_prob is not None:
            effective_max_fault_prob = args_cli.max_fault_prob
        elif args_cli.fault_prob > 0.0:
            effective_max_fault_prob = args_cli.fault_prob

        # Warn if fault injection requested but no faulty surrogates
        if effective_max_fault_prob > 0.0 and len(faulty_ckpts_list) == 0:
            logger.warning("Fault injection requested (max_fault_prob=%.3f) but no --faulty-surrogates provided. Faults will be disabled.", effective_max_fault_prob)
            effective_max_fault_prob = 0.0

        # Pass warm-up settings into args so the Lightning module can read them
        setattr(args, "max_fault_prob", effective_max_fault_prob)
        setattr(args, "fault_warmup_epochs", args_cli.fault_warmup_epochs)
        setattr(args, "fault_ramp_epochs", args_cli.fault_ramp_epochs)

        logger.info("Fault-aware training config: max_fault_prob=%.3f, warmup_epochs=%d, ramp_epochs=%d",
                    effective_max_fault_prob, args_cli.fault_warmup_epochs, args_cli.fault_ramp_epochs)

        
        # ---------------------------
        # parse static-param CLI input
        # ---------------------------
                # ---------------------------
        # Robust static-param parsing
        # ---------------------------
        def _csv_to_floats(s):
            """Return a python list of floats or None. Accepts str, list/tuple, or torch.Tensor."""
            if s is None:
                return None
            # torch tensor -> convert to list
            if isinstance(s, torch.Tensor):
                return [float(x) for x in s.detach().cpu().view(-1).tolist()]
            # list/tuple -> convert elements to float
            if isinstance(s, (list, tuple)):
                return [float(x) for x in s]
            # string -> split on commas
            if isinstance(s, str):
                parts = [tok.strip() for tok in s.split(",") if tok.strip()]
                return [float(x) for x in parts]
            # single numeric value (int/float)
            if isinstance(s, (int, float)):
                return [float(s)]
            raise TypeError(f"Unsupported type for static-param vector: {type(s)}. Value: {s}")

        def _maybe_tensor_from_input(inp, expected_len: int = None):
            """
            Convert input (str/list/torch.Tensor/None) into torch.Tensor or None.
            If expected_len is provided, will raise ValueError if lengths mismatch.
            """
            if inp is None:
                return None
            vals = _csv_to_floats(inp)
            t = torch.tensor(vals, dtype=torch.float32)
            if expected_len is not None and t.numel() != expected_len:
                raise ValueError(f"Expected length {expected_len} but got {t.numel()} for vector {inp!r}")
            return t

        def _broadcast_or_trim(t_src: torch.Tensor, target_len: int) -> torch.Tensor:
            """If t_src shorter, repeat last element; if longer, trim. Returns tensor on CPU float32."""
            if t_src is None:
                # default to zeros
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
            # src_len > target_len -> trim
            return src[:target_len]

        # parse num-static-param: accept "4" or "4,6" or list/tuple or torch.Tensor
        if isinstance(args_cli.num_static_param, str) and "," in args_cli.num_static_param:
            parts = [int(x.strip()) for x in args_cli.num_static_param.split(",")]
            if len(parts) != 2:
                raise ValueError("--num-static-param must be a single int or two ints separated by a comma (main,faulty)")
            num_main, num_faulty = parts
            num_static_param_arg = (num_main, num_faulty)
        elif isinstance(args_cli.num_static_param, (list, tuple, torch.Tensor)):
            # list/tuple/tensor e.g., [4,6] or tensor([4,6])
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
            # single int like "4" or numeric
            num_static_param_arg = int(args_cli.num_static_param)

        # determine num_main / num_faulty as integers
        if isinstance(num_static_param_arg, tuple):
            num_main, num_faulty = num_static_param_arg
        else:
            num_main = num_static_param_arg
            num_faulty = num_main

        # parse main min/max (accept many input forms)
        t_min_main = _maybe_tensor_from_input(args_cli.min_static_main, expected_len=None)
        t_max_main = _maybe_tensor_from_input(args_cli.max_static_main, expected_len=None)

        # fallback defaults
        if t_min_main is None:
            t_min_main = torch.zeros(num_main, dtype=torch.float32)
        if t_max_main is None:
            t_max_main = torch.ones(num_main, dtype=torch.float32)

        # validate or broadcast main to num_main
        if t_min_main.numel() != num_main:
            t_min_main = _broadcast_or_trim(t_min_main, num_main)
        if t_max_main.numel() != num_main:
            t_max_main = _broadcast_or_trim(t_max_main, num_main)

        # parse faulty min/max if provided; else reuse/broadcast main
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

        # Final objects to pass to pSNN: either tensors or tuple(tensor_main, tensor_faulty)
        if isinstance(num_static_param_arg, tuple):
            min_value_static_params_arg = (t_min_main, t_min_faulty)
            max_value_static_params_arg = (t_max_main, t_max_faulty)
        else:
            min_value_static_params_arg = t_min_main
            max_value_static_params_arg = t_max_main

        # (Optional) log parsed values for debugging
        logger.debug("Static params parsed: num_main=%s num_faulty=%s", num_main, num_faulty)
        logger.debug("min_main=%s max_main=%s", t_min_main.tolist(), t_max_main.tolist())
        logger.debug("min_faulty=%s max_faulty=%s", t_min_faulty.tolist(), t_max_faulty.tolist())


        # instantiate the model wrapper
        surrogate_ckpt = args_cli.surrogate_ckpt
        hidden_list = args.hidden if getattr(args, "hidden", None) else []
        topology = [datainfo['N_feature']] + hidden_list + [datainfo['N_class']]
        logger.info("Instantiating PrintedSpikingNetwork with topology: %s (dataset=%s task=%s)", topology, dset, getattr(args, "task", None))
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
                fault_prob=args_cli.fault_prob,
            )

            logger.info("PrintedSpikingNetwork instantiated (surrogate_ckpt=%s, surrogate_class=%s, dataset=%s task=%s)",
                        surrogate_ckpt, args_cli.surrogate_class, dset, getattr(args, "task", None))
        except Exception as e:
            logger.exception("Failed to instantiate PrintedSpikingNetwork for dataset %s: %s", dset, e)
            if args_cli.stop_on_error:
                raise
            else:
                logger.warning("Continuing to next dataset due to --stop-on-error=False")
                continue

        # WandB logger: include dataset id and task in run name to keep separate runs
        run_name = f"{args_cli.experiment}_FaultProb{args_cli.fault_prob}_{datainfo['dataname']}"
        logger.info("Setting up WandB logger (project=%s, run=%s) (dataset=%s task=%s)", args_cli.project, run_name, dset, getattr(args, "task", None))
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
        dataset_checkpoint_dir = os.path.join(base_checkpoint_dir, f"dataset_{getattr(args, 'task', 'auto')}_{dset}")
        os.makedirs(dataset_checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=dataset_checkpoint_dir,
            filename=f"{args_cli.experiment}-{args_cli.surrogate_class}-ds{getattr(args, 'task', 'auto')}{dset}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        logger.info("ModelCheckpoint configured: dir=%s filename=%s (dataset=%s task=%s)", dataset_checkpoint_dir, checkpoint_callback.filename, dset, getattr(args, "task", None))

        accelerator = "cpu"
        if args_cli.device == "gpu" and torch.cuda.is_available():
            accelerator = "gpu"
        logger.info("Using accelerator: %s (dataset=%s task=%s)", accelerator, dset, getattr(args, "task", None))

        # instantiate trainer
        trainer = Trainer(
            fast_dev_run=args_cli.fast_dev_run,
            max_epochs=args_cli.epochs,
            logger=wandb_logger,
            accelerator=accelerator,
            callbacks=[checkpoint_callback],
        )
        logger.info("PyTorch Lightning Trainer instantiated (fast_dev_run=%s, max_epochs=%d) (dataset=%s task=%s)",
                    args_cli.fast_dev_run, args_cli.epochs, dset, getattr(args, "task", None))

        # Train
        try:
            logger.info("Starting training for %d epochs (dataset=%s task=%s)", args_cli.epochs, dset, getattr(args, "task", None))
            trainer.fit(psnn)
            logger.info("Training finished successfully (dataset=%s task=%s)", dset, getattr(args, "task", None))
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
        logger.info("Starting test using best checkpoint (dataset=%s task=%s)", dset, getattr(args, "task", None))
        try:
            trainer.test(
                model=psnn,
                ckpt_path="best"
            )
            logger.info("Test finished. Best checkpoint: %s (dataset=%s task=%s)", checkpoint_callback.best_model_path or "N/A", dset, getattr(args, "task", None))
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
                logger.info("WandB experiment finished (dataset=%s task=%s)", dset, getattr(args, "task", None))
        except Exception as e:
            logger.warning("wandb_logger.finalize() failed for dataset %s: %s", dset, e)

        # free GPU cache before next run (if applicable)
        if accelerator == "gpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("=== Completed run for dataset %s (task=%s) ===", dset, getattr(args, "task", None))

    logger.info("train_snn.py completed for all dataset specs")


if __name__ == '__main__':
    main()