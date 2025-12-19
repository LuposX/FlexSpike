import os 
import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from typing import Any, List, Optional, Union, Tuple

from utils.evaluation import Evaluator

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ===============================================================================
# ===============================================================================

class LightningPrintedSpikingNetwork(pl.LightningModule):
    def __init__(self,
                 topology,
                 args,
                 model_class,
                 ckpt_path: str,
                 train_loader,
                 valid_loader,
                 test_loader,
                 surrogate_gradient,
                 # either int (single main/faulty) or a tuple (main:int, faulty:int)
                 num_static_param: Union[int, Tuple[int, int]],
                 # either a tensor for main/faulty shared, or a pair of tensor/list (main, faulty)
                 min_value_static_params: Union[torch.Tensor, Tuple[torch.Tensor]],
                 max_value_static_params: Union[torch.Tensor, Tuple[torch.Tensor]],
                 loss_fn=None,
                 train_dataset=None,
                 valid_dataset=None,
                 faulty_ckpt_paths: Optional[List[str]] = None,  # list of faulty surrogate ckpts
                 # new args
                 mc_samples: int = 1,                              # K: MC draws per minibatch
                 use_interpolation: bool = False,                  # allow disabling interpolation
                 warmup_epochs: int = 20):                         # keep warmup (no faults)
        super().__init__()

        if ckpt_path is None or ckpt_path == "" or not isinstance(ckpt_path, str):
            raise ValueError(
                "Error: No checkpoint path provided. "
                "You must supply a valid ckpt file with --surrogate-ckpt <path>."
            )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_path}"
            )

        # Save hyperparameters (ignore heavy objects)
        self.save_hyperparameters(ignore=['model_class', 'ckpt_path', 'loss_fn',
                                          'train_loader', 'valid_loader', 'test_loader'])

        # Store MC + interpolation settings on args so downstream modules read them
        setattr(args, "mc_samples", int(mc_samples))
        setattr(args, "use_interpolation", bool(use_interpolation))
        # initialize fault mode: start with no faults (warmup)
        setattr(args, "fault_mode", "none")
        # keep warmup_epochs on the module (user-specified or fallback)
        self.warmup_epochs = int(getattr(args, "fault_warmup_epochs", warmup_epochs))

        # keep older fault-related fields for backward compatibility but we won't use them
        setattr(args, "faulty_ckpt_paths", faulty_ckpt_paths if faulty_ckpt_paths is not None else [])

        self.args = args
        self.network = PrintedSpikingNeuralNetwork(
            topology, args, model_class, ckpt_path,
            surrogate_gradient, train_dataset, valid_dataset,
            num_static_param, min_value_static_params, max_value_static_params,
            faulty_ckpt_paths=faulty_ckpt_paths, fault_prob=0.0  # pass 0.0 to keep legacy param present
        )

        # loss_fn expects (model, x, y) -> scalar (matches your LFLoss)
        self.loss_fn = loss_fn if loss_fn is not None else LFLoss(args)

        # evaluator returns (acc, power)
        self.evaluator = Evaluator(args)

        num_classes = topology[-1]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # ROC computation toggle (only used at test time)
        self.compute_roc = bool(getattr(self.args, "compute_roc", True))

        # storage for last-run ROC curves per tested fault level:
        # will be dict mapping prob -> (fpr_array, tpr_array)
        self.test_roc_curve_by_level = {}


    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        lr = getattr(self.args, "LR", getattr(self.args, "lr", 1e-3))
        params = self.network.GetParam()
        #print(">>> optimizer param count:", sum(p.numel() for p in params))
        #for p in params:
        #    print("  p.requires_grad", p.requires_grad, "shape", p.shape)
        optimizer = torch.optim.AdamW(params, lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.EPOCH)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=getattr(self.args, "LR_DECAY", 0.001),
                patience=getattr(self.args, "LR_PATIENCE", 5),
                min_lr=getattr(self.args, "LR_MIN", 1e-8))
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, T) ; y: (B,)
        loss = self.loss_fn(self.network, x, y)

        train_acc, train_power = self.evaluator(self.network, x, y)

        # Log step-level metrics; aggregate on_epoch automatically
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)
        # network.power is updated in forward pass when necessary; we log epoch-wise
        self.log("train_power", train_power, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def on_train_epoch_start(self):
        """After warmup epochs, enable single-fault injection (exactly one faulty neuron per forward).
           During warmup we keep fault_mode='none' (no faults)."""
        epoch = self.current_epoch

        if epoch < self.warmup_epochs:
            current_mode = "none"
        else:
            current_mode = "single"

        setattr(self.args, "fault_mode", current_mode)

        # mixing alpha: if interpolation is disabled via args.use_interpolation -> we treat alpha as 1.0 (full replacement)
        if not bool(getattr(self.args, "use_interpolation", False)):
            current_alpha = 1.0
        else:
            # keep any user-provided alpha (default 1.0)
            current_alpha = float(getattr(self.args, "fault_mix_alpha", 1.0))
        setattr(self.args, "fault_mix_alpha", current_alpha)

        # propagate to network
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)

        # Log simple numeric indicators (PL logs prefer numeric values)
        mode_flag = 0 if current_mode == "none" else 1
        self.log("fault_mode_single_flag", mode_flag, on_epoch=True, prog_bar=True)
        self.log("mc_samples", int(getattr(self.args, "mc_samples", 1)), on_epoch=True, prog_bar=False)


    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
        
        # Also log current fault prob (in case it wasn't logged in on_train_epoch_start)
        current_prob = float(getattr(self.args, "fault_prob", 0.0))
        self.log("fault_prob", current_prob, prog_bar=False, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.network, x, y)
        valid_acc, valid_power = self.evaluator(self.network, x, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_power", valid_power, on_step=False, on_epoch=True, prog_bar=False)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        # Temporarily disable faults for the main test metrics (use fault_mode='none')
        orig_mode = getattr(self.args, "fault_mode", "none")
        setattr(self.args, "fault_mode", "none")
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)

        x, y = batch
        loss = self.loss_fn(self.network, x, y)
        test_acc, test_power = self.evaluator(self.network, x, y)

        # Restore original mode
        setattr(self.args, "fault_mode", orig_mode)
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)

        self.log("test_loss_fault_0.0", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc_fault_0.0", test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_power_fault_0.0", test_power, on_step=False, on_epoch=True, prog_bar=False)
        return {"test_loss_fault_0.0": loss}
        

    def on_test_epoch_end(self):
        """
        Evaluate the test set under different fault modes using MC sampling.
        Uses args.eval_mc_samples if set, else args.mc_samples, else 1.
        Averages softmax probabilities across MC draws to compute accuracy & ROC.
        """
        # Save originals to restore later
        orig_mode = getattr(self.args, "fault_mode", "none")
        orig_mc = int(getattr(self.args, "mc_samples", 1))
        orig_use_interp = bool(getattr(self.args, "use_interpolation", False))
    
        # Determine modes to sweep
        configured_modes = getattr(self.args, "test_fault_modes", None)
        if configured_modes is None:
            sweep_modes = ["none", "single"]
        else:
            sweep_modes = [str(m) for m in configured_modes]
        if "none" not in sweep_modes:
            sweep_modes = ["none"] + sweep_modes
    
        # decide eval_K default: args.eval_mc_samples else args.mc_samples else 1
        eval_K_default = int(getattr(self.args, "eval_mc_samples", getattr(self.args, "mc_samples", 1)))
    
        def evaluate_mode(mode: str, eval_K: int):
            """Evaluate the entire test_loader for a single fault mode and return metrics."""
            setattr(self.args, "fault_mode", mode)
            if hasattr(self.network, "UpdateArgs"):
                self.network.UpdateArgs(self.args)
            self.network.eval()
    
            total_samples = 0
            acc_sum = 0.0
            power_sum = 0.0
            probs_list = []
            labels_list = []
    
            with torch.no_grad():
                for xb, yb in self.test_loader:
                    xb = xb.to(self.network.DEVICE)
                    yb = yb.to(self.network.DEVICE)
                    B = xb.shape[0]
                    total_samples += B
    
                    # accumulate probabilities (numpy) and power across MC draws
                    probs_accum = None
                    power_accum = 0.0
    
                    for k in range(eval_K):
                        preds = self.network(xb)  # (B, C, T) or (B, T)
                        if preds.dim() == 2:
                            preds = preds.unsqueeze(1)  # (B,1,T)
                        avg_logits = preds.mean(dim=2)  # (B, C)
                        probs_k = torch.softmax(avg_logits, dim=1).detach().cpu().numpy()  # (B, C)
    
                        if probs_accum is None:
                            probs_accum = probs_k
                        else:
                            probs_accum += probs_k
    
                        power_accum += float(self.network.power.detach().cpu().item())
    
                    probs_mean = probs_accum / float(eval_K)  # (B, C) numpy
                    mean_power_batch = power_accum / float(eval_K)
    
                    preds_labels = np.argmax(probs_mean, axis=1)
                    y_np = yb.detach().cpu().numpy()
                    batch_acc = (preds_labels == y_np).mean()
    
                    acc_sum += float(batch_acc) * B
                    power_sum += float(mean_power_batch) * B
    
                    probs_list.append(torch.from_numpy(probs_mean))
                    labels_list.append(torch.from_numpy(y_np))
    
            probs_all = torch.cat(probs_list, dim=0).numpy() if len(probs_list) else None
            labels_all = torch.cat(labels_list, dim=0).numpy() if len(labels_list) else None
            mean_acc = acc_sum / (total_samples + 1e-12)
            mean_power = power_sum / (total_samples + 1e-12)
            return mean_acc, mean_power, probs_all, labels_all
    
        # Run sweep
        for mode in sweep_modes:
            eval_K = 1 if mode == "none" else eval_K_default
            mean_acc, mean_power, probs_all, labels_all = evaluate_mode(mode, eval_K)
    
            tag_acc = f"test_acc_mode_{mode}"
            tag_power = f"test_power_mode_{mode}"
            self.log(tag_acc, mean_acc, prog_bar=True, on_epoch=True)
            self.log(tag_power, mean_power, prog_bar=False, on_epoch=True)
    
            # ROC computation (same logic as before)
            if self.compute_roc and (probs_all is not None) and (labels_all is not None):
                num_classes = probs_all.shape[1]
                if num_classes == 1:
                    try:
                        fpr, tpr, _ = roc_curve(labels_all, probs_all[:, 0])
                        auc_micro = auc(fpr, tpr)
                    except Exception:
                        fpr, tpr = np.array([]), np.array([])
                        auc_micro = float("nan")
                elif num_classes == 2:
                    y_score = probs_all[:, 1]
                    fpr, tpr, _ = roc_curve(labels_all, y_score)
                    auc_micro = auc(fpr, tpr)
                else:
                    y_true = label_binarize(labels_all, classes=list(range(num_classes)))
                    fpr, tpr, _ = roc_curve(y_true.ravel(), probs_all.ravel())
                    auc_micro = roc_auc_score(y_true, probs_all, average='micro', multi_class='ovr')
    
                tag_auc = f"test_roc_micro_auc_mode_{mode}"
                if mode == "none":
                    self.log("test_roc_micro_auc", float(auc_micro), on_epoch=True, prog_bar=True)
                self.log(tag_auc, float(auc_micro), on_epoch=True, prog_bar=True)
                self.test_roc_curve_by_level[mode] = (fpr, tpr)
    
                if hasattr(self, "logger") and isinstance(self.logger, pl.loggers.WandbLogger):
                    roc_table = wandb.Table(columns=["fpr", "tpr"], data=list(zip(fpr, tpr)))
                    self.logger.experiment.log({
                        f"roc_curve_mode_{mode}": wandb.plot.line(
                            roc_table, "fpr", "tpr", title=f"ROC Curve (mode={mode})"
                        )
                    })
    
        # Restore originals
        setattr(self.args, "fault_mode", orig_mode)
        setattr(self.args, "mc_samples", orig_mc)
        setattr(self.args, "use_interpolation", orig_use_interp)
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)



    
    def UpdateArgs(self, args):
        """Keep compatibility with the original code."""
        self.args = args
        self.network.UpdateArgs(args)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["custom_args"] = vars(self.args) if hasattr(self.args, "__dict__") else {}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

# ===============================================================================
# ============================ Single Spike Generator ===========================
# ===============================================================================


class pSpikeGenerator(nn.Module):
    def __init__(self,
                 args,
                 model_class,
                 ckpt_path: str,
                 num_static_param,
                 surrogate_gradient,
                 train_dataset,
                 valid_dataset,
                 min_value_static_params,
                 max_value_static_params,
                 faulty_ckpt_paths: Optional[List[str]] = None,
                 fault_prob: float = 0.0):
        super().__init__()
        self.args = args

        # Helper: allow either single-spec or pair-spec (main, faulty)
        def _split_pair(x, name):
            # returns (main, faulty)
            if isinstance(x, (list, tuple)) and len(x) == 2:
                return x[0], x[1]
            else:
                return x, x

        # parse number-of-static-params (can be int or (int,int))
        n_main, n_faulty = _split_pair(num_static_param, "num_static_param")
        if not (isinstance(n_main, int) or (isinstance(n_main, torch.Tensor) and n_main.numel()==1)):
            raise TypeError(f"num_static_param main must be an int or scalar tensor. Got: {type(n_main)}")
        if not (isinstance(n_faulty, int) or (isinstance(n_faulty, torch.Tensor) and n_faulty.numel()==1)):
            raise TypeError(f"num_static_param faulty must be an int or scalar tensor. Got: {type(n_faulty)}")

        self.num_static_param_main = int(n_main)
        self.num_static_param_faulty = int(n_faulty)

        # Load the *clean* spike generator
        self.spike_generator = model_class.load_from_checkpoint(
            ckpt_path,
            map_location=self.DEVICE,
            surrogate_gradient=surrogate_gradient,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )
        self.spike_generator.train(False)
        for param in self.spike_generator.parameters():
            param.requires_grad = False

        # Load faulty surrogates (if provided)
        self.faulty_spike_generators = torch.nn.ModuleList()
        faulty_ckpt_paths = faulty_ckpt_paths or getattr(args, "faulty_ckpt_paths", []) or []
        for fpath in faulty_ckpt_paths:
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Faulty surrogate checkpoint not found: {fpath}")
            fgen = model_class.load_from_checkpoint(
                fpath,
                map_location=self.DEVICE,
                surrogate_gradient=surrogate_gradient,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
            )
            fgen.train(False)
            for param in fgen.parameters():
                param.requires_grad = False
            self.faulty_spike_generators.append(fgen)

        # fault probability (per-neuron replacement prob)
        self.fault_prob = float(fault_prob) if fault_prob is not None else float(getattr(args, "fault_prob", 0.0))

        # Define raw trainable parameters (unconstrained) on correct device
        # main raw params
        self.raw_params_main = nn.Parameter(torch.randn(1, self.num_static_param_main, device=self.DEVICE)) \
                                if self.num_static_param_main > 0 else None
        # faulty raw params (shared for all faulty surrogates)
        self.raw_params_faulty = nn.Parameter(torch.randn(1, self.num_static_param_faulty, device=self.DEVICE)) \
                                 if (self.num_static_param_faulty > 0 and len(self.faulty_spike_generators) > 0) else None

        # Ensure min/max are tensors on the right device and shaped (num_static_param,)
        min_main, min_faulty = _split_pair(min_value_static_params, "min_value_static_params")
        max_main, max_faulty = _split_pair(max_value_static_params, "max_value_static_params")

        # Convert to tensors on device and view(-1)
        self.low_main = (min_main.clone().detach().to(self.DEVICE).view(-1)) if isinstance(min_main, torch.Tensor) else \
                        torch.tensor(min_main, device=self.DEVICE).view(-1)
        self.high_main = (max_main.clone().detach().to(self.DEVICE).view(-1)) if isinstance(max_main, torch.Tensor) else \
                         torch.tensor(max_main, device=self.DEVICE).view(-1)

        self.low_faulty = (min_faulty.clone().detach().to(self.DEVICE).view(-1)) if isinstance(min_faulty, torch.Tensor) else \
                          torch.tensor(min_faulty, device=self.DEVICE).view(-1)
        self.high_faulty = (max_faulty.clone().detach().to(self.DEVICE).view(-1)) if isinstance(max_faulty, torch.Tensor) else \
                           torch.tensor(max_faulty, device=self.DEVICE).view(-1)

        if self.low_main.numel() != self.num_static_param_main or self.high_main.numel() != self.num_static_param_main:
            raise ValueError(f"min/max main vectors must have length {self.num_static_param_main}; got {self.low_main.numel()}/{self.high_main.numel()}")
        if (len(self.faulty_spike_generators) > 0) and (self.low_faulty.numel() != self.num_static_param_faulty or self.high_faulty.numel() != self.num_static_param_faulty):
            raise ValueError(f"min/max faulty vectors must have length {self.num_static_param_faulty}; got {self.low_faulty.numel()}/{self.high_faulty.numel()}")

    @property
    def DEVICE(self):
        return torch.device(self.args.DEVICE) if isinstance(self.args.DEVICE, str) else self.args.DEVICE

    def _transform(self, raw_params, low, high):
        # raw_params shape: (1, num_static_param)
        if raw_params is None:
            return None
        r = (high - low) / 2
        c = (high + low) / 2
        # ensure shapes broadcastable: low/high are (num_static_param,), raw (1,num_static_param)
        return c + r * torch.tanh(raw_params)

    def forward(self, x, force_fault: Optional[bool] = None, disable_interpolation: Optional[bool] = None):
        """
        Forward with optional forced fault behavior.
         - force_fault: None => caller decided not to force (we default to 'no fault' unless SGLayer passes True).
                        True  => use faulty surrogate for this entire batch (either mixed or full, depending on interpolation).
                        False => use main (no fault).
         - disable_interpolation: override args.use_interpolation for this call.
        """
        # Determine interpolation usage
        if disable_interpolation is None:
            disable_interpolation = not bool(getattr(self.args, "use_interpolation", False))

        alpha = float(getattr(self.args, "fault_mix_alpha", 1.0))
        alpha = max(0.0, min(1.0, alpha))
        if disable_interpolation:
            alpha_eff = 1.0
        else:
            alpha_eff = alpha

        batch_size = x.shape[0]
        T = x.shape[2]
        device = self.DEVICE

        # Build MAIN input as before
        extra_main = self._transform(self.raw_params_main, self.low_main, self.high_main)
        if extra_main is None:
            expanded_main = torch.empty(batch_size, 0, T, device=device)
        else:
            expanded_main = extra_main.expand(batch_size, -1).unsqueeze(2).expand(-1, -1, T)
        x_main = torch.cat([x, expanded_main], dim=1)
        out_main = self.spike_generator(x_main)

        # If no faulty surrogates exist, always return main
        if len(self.faulty_spike_generators) == 0:
            if out_main.dim() == 3 and out_main.shape[1] == 1:
                return out_main.squeeze(1)
            return out_main

        # Build FAULTY input as before
        extra_faulty = self._transform(self.raw_params_faulty, self.low_faulty, self.high_faulty)
        if extra_faulty is None:
            expanded_faulty = torch.empty(batch_size, 0, T, device=device)
        else:
            expanded_faulty = extra_faulty.expand(batch_size, -1).unsqueeze(2).expand(-1, -1, T)
        x_faulty = torch.cat([x, expanded_faulty], dim=1)

        # If force_fault is None -> default to no-fault behavior (we rely on SGLayer to explicitly request faults)
        if force_fault is None:
            # default no-fault: return main
            if out_main.dim() == 3 and out_main.shape[1] == 1:
                return out_main.squeeze(1)
            return out_main

        # If force_fault is False -> explicitly return main
        if force_fault is False:
            if out_main.dim() == 3 and out_main.shape[1] == 1:
                return out_main.squeeze(1)
            return out_main

        # If force_fault is True -> use a randomly selected faulty surrogate for this forward
        idx = torch.randint(len(self.faulty_spike_generators), (1,), device=device).item()
        faulty_gen = self.faulty_spike_generators[idx]
        out_faulty = faulty_gen(x_faulty)

        # Normalize outputs to (B,1,T)
        def normalize_out(o, which):
            if o.dim() == 2:          # (B, T) -> (B, 1, T)
                return o.unsqueeze(1)
            elif o.dim() == 3:
                B, C, TT = o.shape
                if TT != T:
                    raise RuntimeError(f"{which} returned time-dim {TT} but input T={T}")
                if C > 1:
                    raise RuntimeError(f"{which} produced {C} output channels; expected single-channel output.")
                return o
            else:
                raise RuntimeError(f"{which} returned unexpected ndim={o.dim()}; shape={tuple(o.shape)}")

        out_m = normalize_out(out_main, "main")
        out_f = normalize_out(out_faulty, "faulty")

        # alpha_tensor is broadcasted over time. Since force_fault=True, alpha_val = alpha_eff.
        alpha_tensor = alpha_eff  # scalar in [0,1]
        out_mixed = (1.0 - alpha_tensor) * out_m + alpha_tensor * out_f  # (B,1,T)

        return out_mixed.squeeze(1)  # (B, T)



    def UpdateArgs(self, args):
        self.args = args
        # refresh fault probability from args (if changed)
        self.fault_prob = float(getattr(args, "fault_prob", self.fault_prob))



# ===============================================================================
# ============================== SG Layer =======================================
# ===============================================================================

class SGLayer(torch.nn.Module):
    def __init__(self,
                 N,
                 args,
                 model_class,
                 ckpt_path,
                 surrogate_gradient,
                 train_dataset,
                 valid_dataset,
                 num_static_param,
                 min_value_static_params,
                 max_value_static_params,
                 faulty_ckpt_paths: Optional[List[str]] = None,
                 fault_prob: float = 0.0):
        super().__init__()
        self.args = args
        self.SG_Group = torch.nn.ModuleList(
            [pSpikeGenerator(
                 args,
                 model_class,
                 ckpt_path,
                 num_static_param,
                 surrogate_gradient,
                 train_dataset,
                 valid_dataset,
                 min_value_static_params,
                 max_value_static_params,
                 faulty_ckpt_paths=faulty_ckpt_paths,
                 fault_prob=fault_prob
             ) for _ in range(N)]
        )

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        # Decide the single faulty neuron index according to args.fault_mode
        fault_mode = getattr(self.args, "fault_mode", "none")
        N = len(self.SG_Group)
        if fault_mode == "single" and N > 0:
            # pick exactly one SG index to be faulty for this forward (uniform)
            faulty_idx = torch.randint(N, (1,), device=self.DEVICE).item()
        else:
            faulty_idx = None

        result = []
        for n in range(N):
            x_temp = x[:, n, :].unsqueeze(-1)
            force_fault = (faulty_idx == n)
            result.append(self.SG_Group[n](x_temp, force_fault=force_fault))
        # result list length N with each item shaped (B, C_out, T) or (B, T)
        return torch.stack(result).permute(1, 0, 2)


    def UpdateArgs(self, args):
        self.args = args
        for sg in self.SG_Group:
            if hasattr(sg, "UpdateArgs"):
                sg.UpdateArgs(args)


# ===============================================================================
# =====================  Learnable Negative Weight Circuit  =====================
# ===============================================================================

class Inv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def forward(self, z):
        return - torch.tanh(z)


# ===============================================================================
# ============================= Printed Layer ===================================
# ===============================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, INV, model_class, ckpt_path, surrogate_gradient,
                 train_dataset, valid_dataset, num_static_param, min_value_static_params, max_value_static_params,
                 faulty_ckpt_paths: Optional[List[str]] = None, fault_prob: float = 0.0):
        super().__init__()
        self.args = args
        self.SG = SGLayer(n_out, args, model_class, ckpt_path, surrogate_gradient,
                          train_dataset, valid_dataset, num_static_param,
                          min_value_static_params, max_value_static_params,
                          faulty_ckpt_paths=faulty_ckpt_paths, fault_prob=fault_prob)
        self.INV = INV

        theta = torch.rand([n_in + 2, n_out])/10. + args.gmin
        theta[-2, :] = args.gmax - theta[-2, :]
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    # rest of pLayer unchanged (properties, MAC, MACPower etc.)
    # ensure UpdateArgs forwards to SG
    def UpdateArgs(self, args):
        self.args = args
        if hasattr(self, "SG") and hasattr(self.SG, "UpdateArgs"):
            self.SG.UpdateArgs(args)

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def W(self):
        return self.theta.abs() / torch.sum(self.theta.abs(), axis=0, keepdim=True)

    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive
        a_extend = torch.cat([a,
                              torch.ones([a.shape[0], 1]).to(self.device),
                              torch.zeros([a.shape[0], 1]).to(self.device)], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:, -1] = 0.
        z = torch.matmul(a_extend, self.W * positive) + \
            torch.matmul(a_neg, self.W * negative)
        return z

    @property
    def neg_power(self):
        # Exclude bias and dummy from power computation
        theta = self.theta.clone().detach()[:-2, :]  # [input_dim, output_dim]
        
        # Identify negative weights
        negative_mask = (theta < 0).float()
        N_neg_hard = negative_mask.sum()

        # Soft (gradient-aware) count of negative weights
        soft_count = 1 - torch.sigmoid(self.theta[:-2, :])
        soft_count = soft_count * negative_mask
        soft_N_neg = soft_count.max(dim=1)[0].sum()

        # Surrogate power from InvRT
        inv_power_scalar = self.INV.power.item() if hasattr(self, "INV") else 0.0

        # Compute final power (hard + relaxed)
        power_hard = inv_power_scalar * N_neg_hard
        power_soft = inv_power_scalar * soft_N_neg
        
        return power_hard + power_soft - power_soft.detach()

    def forward(self, x):
        T = x.shape[2]
        result = []
        self.power = torch.tensor(0.).to(self.device)
        for t in range(T):
            mac = self.MAC(x[:, :, t])
            result.append(mac)
            self.power += self.MACPower(x[:, :, t], mac)
        z_new = torch.stack(result, dim=2)
        self.power = self.power / T
        a_new = self.SG(z_new)
        return a_new

    @property
    def g_tilde(self):
        g_initial = self.theta_.abs()
        g_min = g_initial.min(dim=0, keepdim=True)[0]
        scaler = self.args.pgmin / g_min
        return g_initial * scaler

    def MACPower(self, x, y):
        x_extend = torch.cat([x,
                              torch.ones([x.shape[0], 1]).to(self.device),
                              torch.zeros([x.shape[0], 1]).to(self.device)], dim=1)
        x_neg = self.INV(x_extend)
        x_neg[:, -1] = 0.

        E = x_extend.shape[0]
        M = x_extend.shape[1]
        N = y.shape[1]

        positive = self.theta.clone().detach().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        Power = torch.tensor(0.).to(self.device)

        for m in range(M):
            for n in range(N):
                Power += self.g_tilde[m, n] * (
                    (x_extend[:, m]*positive[m, n]+x_neg[:, m]*negative[m, n])-y[:, n]).pow(2.).sum()
        Power = Power / E
        return Power


# ===============================================================================
# ======================== Printed Neural Network ===============================
# ===============================================================================


class PrintedSpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, topology, args, model_class, ckpt_path, surrogate_gradient,
                 train_dataset, valid_dataset, num_static_param, min_value_static_params, max_value_static_params,
                 faulty_ckpt_paths: Optional[List[str]] = None, fault_prob: float = 0.0):
        super().__init__()
        self.args = args
        self.INV = Inv(args)

        self.model = torch.nn.Sequential()
        num_layers = len(topology) - 1
        
        for i in range(num_layers):
            is_output_layer = (i == num_layers - 1)
            
            # If it's the output layer, force probability to 0 and remove faulty paths
            current_fault_prob = 0.0 if is_output_layer else fault_prob
            current_faulty_ckpts = [] if is_output_layer else faulty_ckpt_paths

            self.model.add_module(
                str(i) + '_pLayer',
                pLayer(
                    topology[i],
                    topology[i+1],
                    args,
                    self.INV,
                    model_class,
                    ckpt_path,
                    surrogate_gradient,
                    train_dataset,
                    valid_dataset,
                    num_static_param,
                    min_value_static_params,
                    max_value_static_params,
                    faulty_ckpt_paths=current_faulty_ckpts, # Pass empty list for output
                    fault_prob=current_fault_prob          # Pass 0.0 for output
                )
            )

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        return self.model(x)

    @property
    def power(self):
        power = torch.tensor(0.).to(self.DEVICE)
        for layer in self.model:
            if hasattr(layer, 'power'):
                power += layer.power
        return power
    
    def UpdateArgs(self, args):
        self.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)

    def GetParam(self):
        # include any parameter containing 'raw_params' (covers raw_params_main and raw_params_faulty)
        weights = [p for name, p in self.named_parameters()
                if name.endswith('theta_') or name.endswith('beta') or 'raw_params' in name]
        nonlinear = [p for name, p in self.named_parameters()
                    if name.endswith('rt_')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights



# ===============================================================================
# ============================= Loss Functin ====================================
# ===============================================================================


class LossFN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L

    def celoss(self, prediction, label):
        lossfn = torch.nn.CrossEntropyLoss()
        return lossfn(prediction, label)

    def forward(self, prediction, label):
        if self.args.loss == 'pnnloss':
            return self.standard(prediction, label)
        elif self.args.loss == 'celoss':
            return self.celoss(prediction, label)


class LFLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_fn = LossFN(args)

    def forward(self, model, x, label):
        """
        Perform K independent forward passes (Monte Carlo) and average the per-instance losses.
        K is read from model.args.mc_samples (default 1).
        We also average model.power across the K runs for the regularizer.
        """
        K = int(getattr(model.args, "mc_samples", 1))
        losses = []
        powers = []
        for k in range(K):
            prediction = model(x)  # each call performs an independent draw of the single faulty neuron
            # temporal loss averaged over time dimension
            L_steps = []
            for step in range(prediction.shape[2]):
                L_steps.append(self.loss_fn(prediction[:, :, step], label))
            L_k = torch.stack(L_steps).mean()
            losses.append(L_k)
            # accumulate the measured power for this forward (model.power is updated in forward)
            powers.append(model.power.detach().clone())

        mean_loss = torch.stack(losses).mean()
        mean_power = torch.stack(powers).mean()
        return mean_loss + 0.1 * mean_power
