import os 
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Any, List, Optional, Union, Tuple

from utils.evaluation import Evaluator

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
                 fault_prob: float = 0.0):                         # dropout-like probability
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

        if fault_prob > 0.0 and len(faulty_ckpt_paths) == 0:
            print("Warning: fault_prob > 0 but no faulty surrogate checkpoints were provided. Fault injection is disabled.")

        # Save hyperparameters (ignore heavy objects)
        self.save_hyperparameters(ignore=['model_class', 'ckpt_path', 'loss_fn',
                                          'train_loader', 'valid_loader', 'test_loader'])

        # Add fault config into args for downstream modules to read
        setattr(args, "fault_prob", float(fault_prob))
        setattr(args, "faulty_ckpt_paths", faulty_ckpt_paths if faulty_ckpt_paths is not None else [])

        # Fault warm-up configuration
        self.warmup_epochs = getattr(args, "fault_warmup_epochs", 20)   # epochs with fault_prob=0
        self.ramp_epochs = getattr(args, "fault_ramp_epochs", 50)       # epochs to ramp up
        self.max_fault_prob = getattr(args, "max_fault_prob", fault_prob)  # target prob
        if self.max_fault_prob > 0:
            print(f"Fault injection warm-up: 0 for {self.warmup_epochs} epochs, "
                  f"then ramp to {self.max_fault_prob} over {self.ramp_epochs} epochs")

        self.args = args
        self.network = PrintedSpikingNeuralNetwork(
            topology, args, model_class, ckpt_path,
            surrogate_gradient, train_dataset, valid_dataset,
            num_static_param, min_value_static_params, max_value_static_params,
            faulty_ckpt_paths=faulty_ckpt_paths, fault_prob=fault_prob
        )

        # loss_fn expects (model, x, y) -> scalar (matches your LFLoss)
        self.loss_fn = loss_fn if loss_fn is not None else LFLoss(args)

        # evaluator returns (acc, power)
        self.evaluator = Evaluator(args)

        num_classes = topology[-1]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

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
        """Gradually increase fault_prob during training."""
        if self.max_fault_prob <= 0:
            return  # no fault injection

        epoch = self.current_epoch

        if epoch < self.warmup_epochs:
            current_prob = 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # Linear ramp
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            current_prob = self.max_fault_prob * progress
        else:
            current_prob = self.max_fault_prob

        # Update the args object
        setattr(self.args, "fault_prob", float(current_prob))

        # Propagate to the network so all pSpikeGenerators see the new value
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)

        # Optional: log the current fault probability
        self.log("fault_prob", current_prob, on_epoch=True, prog_bar=True)

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
        # Temporarily disable faults for the main test metrics
        orig_fault_prob = float(getattr(self.args, "fault_prob", 0.0))
        setattr(self.args, "fault_prob", 0.0)
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)
    
        x, y = batch
        loss = self.loss_fn(self.network, x, y)
        test_acc, test_power = self.evaluator(self.network, x, y)
    
        # Restore original for consistency (though not strictly needed in test)
        setattr(self.args, "fault_prob", orig_fault_prob)
        if hasattr(self.network, "UpdateArgs"):
            self.network.UpdateArgs(self.args)
    
        self.log("test_loss_fault_0.0", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc_fault_0.0", test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_power_fault_0.0", test_power, on_step=False, on_epoch=True, prog_bar=False)
        return {"test_loss_fault_0.0": loss}

    def on_test_epoch_end(self):
        """
        After the normal test epoch, run additional evaluations (sweep) with different
        fault probabilities. Logs metrics for each fault level.
        """
        # Get original fault probability to restore later
        orig_fault_prob = float(getattr(self.args, "fault_prob", 0.0))
   
        # Determine which fault levels to test
        configured_levels = getattr(self.args, "test_fault_levels", None)
        if configured_levels is None:
            levels = sorted(set([0.0, orig_fault_prob]))
        else:
            levels = sorted(set([float(l) for l in configured_levels]))

        # Remove 0.0 from the sweep since we already logged it in test_step
        levels = [p for p in levels if p > 0.0]

        if not levels:
            return  # nothing to sweep
        
        # Helper to evaluate the entire test_loader with a given fault probability
        def evaluate_with_prob(p: float):
            # Temporarily override fault probability
            setattr(self.args, "fault_prob", p)
            if hasattr(self.network, "UpdateArgs"):
                self.network.UpdateArgs(self.args)
    
            self.network.eval()
            acc_sum = 0.0
            power_sum = 0.0
            total_samples = 0
    
            with torch.no_grad():
                for xb, yb in self.test_loader:
                    xb = xb.to(self.network.DEVICE)
                    yb = yb.to(self.network.DEVICE)
                    batch_acc, batch_power = self.evaluator(self.network, xb, yb)
                    batch_n = xb.shape[0]
                    acc_sum += float(batch_acc) * batch_n
                    power_sum += float(batch_power) * batch_n
                    total_samples += batch_n
    
            mean_acc = acc_sum / (total_samples + 1e-12)
            mean_power = power_sum / (total_samples + 1e-12)
            return mean_acc, mean_power
    
        # Run the sweep once
        for p in levels:
            mean_acc, mean_power = evaluate_with_prob(p)
            tag_acc = f"test_acc_fault_{int(p * 100)}"
            tag_power = f"test_power_fault_{int(p * 100)}"
            self.log(tag_acc, mean_acc, prog_bar=True, on_epoch=True)
            self.log(tag_power, mean_power, prog_bar=False, on_epoch=True)
    
        # Restore original fault probability
        setattr(self.args, "fault_prob", orig_fault_prob)
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

    def forward(self, x):
        # Decide whether to use faulty surrogate for this neuron
        fault_p = float(getattr(self.args, "fault_prob", self.fault_prob))
        use_fault = False
        if fault_p > 0.0 and len(self.faulty_spike_generators) > 0:
            if torch.rand(1).item() < fault_p:
                use_fault = True

        # pick spike generator to run
        spike_gen = None
        if use_fault:
            idx = torch.randint(len(self.faulty_spike_generators), (1,)).item()
            spike_gen = self.faulty_spike_generators[idx]
        else:
            spike_gen = self.spike_generator

        # Transform and expand trainable parameters according to chosen surrogate
        batch_size = x.shape[0]
        T = x.shape[2]

        if spike_gen is self.spike_generator:
            extra_params = self._transform(self.raw_params_main, self.low_main, self.high_main)  # (1, num_static_param_main)
            if extra_params is None:
                # if no static params for main, make an empty tensor with zero channels
                expanded_params = torch.empty(batch_size, 0, T, device=self.DEVICE)
            else:
                expanded_params = extra_params.expand(batch_size, -1)  # (B, num_static_param_main)
                expanded_params = expanded_params.unsqueeze(2).expand(-1, -1, T)  # (B, num_static_param_main, T)
        else:
            # using faulty surrogate
            extra_params = self._transform(self.raw_params_faulty, self.low_faulty, self.high_faulty)
            if extra_params is None:
                expanded_params = torch.empty(batch_size, 0, T, device=self.DEVICE)
            else:
                expanded_params = extra_params.expand(batch_size, -1)  # (B, num_static_param_faulty)
                expanded_params = expanded_params.unsqueeze(2).expand(-1, -1, T)  # (B, num_static_param_faulty, T)

        # Concatenate with input along channel dimension (if expanded_params has zero channels, cat works)
        x = torch.cat([x, expanded_params], dim=1)  # (B, C+num_static_param_*, T)

        return spike_gen(x)

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
        result = []
        for n in range(len(self.SG_Group)):
            x_temp = x[:, n, :].unsqueeze(-1)
            result.append(self.SG_Group[n](x_temp))
        # result is list length N with each item (B, C_out, T); stacking and permute as before
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
        prediction = model(x)
        L = []
        for step in range(prediction.shape[2]):
            L.append(self.loss_fn(prediction[:, :, step], label))
        return torch.stack(L).mean() + 0.1 * model.power
