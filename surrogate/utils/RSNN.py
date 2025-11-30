#!/usr/bin/env python

import snntorch as snn
from snntorch._layers.bntt import BatchNormTT1d

import pytorch_lightning as L
import torch.nn.functional as F
import torch
import torch.nn as nn

import re

from .src_cell import SRC

from typing import Callable, Union, Optional

class SpikeSynth(L.LightningModule):
    def __init__(
                 self,
                 num_hidden_layers,
                 num_hidden,
                 beta,
                 optimizer_class,
                 optimizer_kwargs,
                 lr,
                 batch_size,
                 dropout,
                 surrogate_gradient,
                 max_epochs,
                 temporal_skip,
                 layer_skip,
                 use_bntt,
                 scheduler_class,
                 log_every_n_steps,
                 use_layernorm,
                 neuron_type="LeakyParallel",
                 scheduler_kwargs=None,
                 bntt_time_steps=None,
                 train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 SRC_config: dict | None = None,
                 loss_fn: Union[str, Callable] = "mse",
                 loss_kwargs: Optional[dict] = None,
                 ):
        """
        Initializes the SpikeSynth model, a spiking neural network (SNN). 
        The model supports temporal and layer-wise skip connections,
        customizable surrogate gradients, and flexible optimizer/lr scheduling.
        Not all options (e.g. LayerSkip, TempralSkip) work with all neuron types.

        Args:
            num_hidden_layers (int): 
                Number of recurrent Leaky Integrate-and-Fire (LIF) layers in the network.

            num_hidden (int): 
                Number of hidden neurons (units) in each LIF layer.

            beta (float): 
                Membrane potential decay constant for the LIF neurons (typically 0 < beta < 1).

            optimizer_class (type): 
                Optimizer class to use (e.g., `torch.optim.Adam`, `torch.optim.SGD`).

            neuron_type: str, one of {"LeakyParallel", "Leaky", "SLSTM"}
               "LeakyParallel" uses snn.LeakyParallel and accepts a full (T,B,F) sequence. 
               "Leaky" uses snn.Leaky and is stepped through time manually (per-time-step). SLSTM" uses snn.SLSTM (the existing code path).

            optimizer_kwargs(dict):
                The paramaters of the optimizer.

            lr (float): 
                Inital Learning rate for the optimizer.

            batch_size (int): 
                Batch size used for both training and validation DataLoaders.

            dropout (float):
                Value between [0, 1] if non zero, introduces a dropout layer.

            surrogate_gradient (torch.autograd.Function): 
                Surrogate gradient function for backpropagation through spikes
                (e.g., `snn.surrogate.atan()`).

            max_epochs (int): 
                Maximum number of training epochs. Used for scheduling when employing
                cosine annealing or similar schedulers.

            temporal_skip (int or -1): 
                Temporal skip interval in time steps for residual connections within
                each LIF layer. If `-1`, no temporal skip is applied.

            layer_skip (int or 0): 
                Number of layers to skip for inter-layer residual connections.
                For example, `layer_skip=1` adds a connection from each layer
                to the one immediately above it. If `0`, no layer skip is applied.

            use_bntt (bool):
                Whether to use 1D Batch Normalization Layer with length time_steps. 
                Uses Batch Normalisation Through Time (BNTT).

            bntt_time_steps (None or int):
                Batch Normalization Layer requries the amount of time steps that will be used i.e. sequecne length.

            train_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for training. Required by `train_dataloader()`.

            valid_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for validation. Required by `val_dataloader()`.

            test_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for testing. Required by `test_dataloader()`.

            scheduler_class (torch.optim.lr_scheduler):
                What Learning Rate Scheduler should be used for training.

            use_slstm (bool): 
                If True, use snn.SLSTM layers for the recurrent layers.

            log_every_n_steps (int):
                Log every n steps.

            use_layernorm (bool):
                If layernorm should be used afetr every Lif layer.

            SRC_config (dict):
                Defines how SRC works possible entries. Default: float = 0.9, rho: float = 3.0, r: float = 2.0, rs: float = -7.0, z: float = 0.0, zhyp_s: float = 0.9, zdep_s: float = 0.0, bh_init: float = -6.0, bh_max: float = -4.0, detach_rec: bool = True, relu_bypass: bool = True. See paper "Spike-based computation using classical recurrent neural networks".
                loss_fn (string): 
                    name or callable for the loss. If string, supported names: 'mse','mae','huber','logcosh','pearson'.
                    
                loss_kwargs (optional dict):
                    forwarded to some losses (e.g. {'delta':0.5} for huber).
        """  
        super().__init__()

        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = self.num_inputs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.max_epochs = max_epochs
        self.surrogate_gradient = surrogate_gradient
        self.SRC_config = SRC_config

        self.save_hyperparameters(ignore=["surrogate_gradient", "train_dataset", "valid_dataset", "test_dataset"])

        if dropout != 0:
            raise ValueError("Dropout currently not supported.")

        # If user selected SRC but SRC implementation couldn't be imported, fail fast
        if neuron_type == "SRC" and SRC is None:
            raise ImportError(
                "SRC neuron type requested but `SRC.SRC` could not be imported."
                " Please ensure `SRC.py` is available and contains the SRC class."
            )

        # If SRC was selected, some features are incompatible — check and fail early.
        if neuron_type == "SRC" or neuron_type == "SLSTM":
            if use_bntt:
                raise ValueError("Neuron Type does not support BNTT (use_bntt=True).")
            if temporal_skip != -1:
                raise ValueError("Neuron Type does not support temporal skip connections (temporal_skip must be -1).")

        self.norm = nn.LayerNorm(self.num_inputs + self.num_params)
        self.lif_layers = nn.ModuleList()
        self.leaky_linears = nn.ModuleList()
        self.temp_skip_projs = nn.ModuleList()
        self.layer_skip_projs = nn.ModuleList()
        self.layer_bntt = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self._build_layers(neuron_type, bntt_time_steps)

        self.output_layer = nn.Linear(self.hparams.num_hidden, self.num_outputs)

        # Build the actual loss callable from the provided spec
        self._build_loss()

    def _build_loss(self):
        """Construct a callable self.loss(preds, targets) from the user-provided spec.
        Accepts a callable or one of a set of string names. If unknown string -> ValueError.
        """
        spec = self.hparams.loss_fn
        kwargs = dict(self.hparams.loss_kwargs or {})
        
        # If user passed a callable, use it directly
        if callable(spec):
            self.loss = spec
            try:
                self._loss_name = getattr(spec, "__name__", "callable_loss")
            except Exception:
                self._loss_name = "callable_loss"
            return
          
        # Otherwise expect a string
        if not isinstance(spec, str):
            raise ValueError("loss_fn must be a string name or a callable")
        
        s = spec.strip().lower()
        if s in ("mse", "mse_loss"):
            crit = nn.MSELoss()
            self.loss = lambda p, t: crit(p, t)
            self._loss_name = "mse"
        elif s in ("mae", "l1", "l1_loss"):
            crit = nn.L1Loss()
            self.loss = lambda p, t: crit(p, t)
            self._loss_name = "mae"
        elif s in ("huber", "smooth_l1", "smooth_l1_loss"):
            delta = kwargs.get("delta", 1.0)
            self.loss = lambda p, t, _d=delta: F.smooth_l1_loss(p, t, beta=_d)
            self._loss_name = f"huber(delta={delta})"
        elif s in ("logcosh", "log_cosh"):
            self.loss = lambda p, t: torch.mean(torch.log(torch.cosh(p - t + 1e-12)))
            self._loss_name = "logcosh"
        elif s in ("pearson", "corr", "correlation"):
            def _pearson(p, t, eps=1e-8):
                p_c = p - p.mean(dim=1, keepdim=True)
                t_c = t - t.mean(dim=1, keepdim=True)
                num = (p_c * t_c).sum(dim=1)
                den = torch.sqrt((p_c ** 2).sum(dim=1) * (t_c ** 2).sum(dim=1)) + eps
                corr = num / den
                return torch.mean(1.0 - corr)
            self.loss = _pearson
            self._loss_name = "pearson"
        else:
            raise ValueError(f"Unknown loss_fn string '{spec}'. Supported: 'mse','mae','huber','logcosh','pearson' or pass a callable.")

    def _build_layers(self, neuron_type, bntt_time_steps):
        input_size = self.num_inputs + self.num_params
        for i in range(self.hparams.num_hidden_layers):
            if neuron_type == "SLSTM":
                layer = snn.SLSTM(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    spike_grad=self.surrogate_gradient
                )
            elif neuron_type == "LeakyParallel":
                layer = snn.LeakyParallel(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient,
                    dropout=self.hparams.dropout
                )
            elif neuron_type == "Leaky":
                layer = snn.Leaky(
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient
                )
                if i == 0:
                    self.leaky_linears.append(nn.Linear(input_size, self.hparams.num_hidden))
                else:
                    self.leaky_linears.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            elif neuron_type == "RLeaky":
                layer = snn.RLeaky(
                    beta=self.hparams.beta,
                    linear_features=self.hparams.num_hidden,
                    spike_grad=self.surrogate_gradient,
                    learn_recurrent=True
                )
                if i == 0:
                    self.leaky_linears.append(nn.Linear(input_size, self.hparams.num_hidden))
                else:
                    self.leaky_linears.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            elif neuron_type == "SRC":
                # Create SRC layer. We purposely do not wire up BNTT, LayerNorm
                # or skip projections for SRC (they were validated in __init__).
                # The SRC implementation expects `input_size` and `hidden_size`.
                layer = SRC(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    **(self.SRC_config or {})
                )
            else:
                raise ValueError(f"Unknown neuron_type: {neuron_type}")

            self.lif_layers.append(layer)

            if input_size != self.hparams.num_hidden and self.hparams.temporal_skip != -1:
                self.temp_skip_projs.append(nn.Identity())#nn.Linear(input_size, self.hparams.num_hidden))
            else:
                self.temp_skip_projs.append(nn.Identity())

            if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                self.layer_skip_projs.append(nn.Identity())#nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            else:
                self.layer_skip_projs.append(None)

            if self.hparams.use_bntt:
                if bntt_time_steps is None and self.hparams.bntt_time_steps is None:
                    raise ValueError("bntt_time_steps must be specified if use_bntt=True")
                time_steps = self.hparams.bntt_time_steps or bntt_time_steps
                self.layer_bntt.append(BatchNormTT1d(self.hparams.num_hidden, time_steps))
            else:
                self.layer_bntt.append(None)

            if self.hparams.use_layernorm:
                self.layer_norms.append(nn.LayerNorm(self.hparams.num_hidden))
            else:
                self.layer_norms.append(None)

            input_size = self.hparams.num_hidden

    def _norm_and_bntt(self, idx, tensor, t=None, per_timestep=False):
        """
        Apply LayerNorm (per-feature) and BNTT if present for layer idx.
        """
        ln = self.layer_norms[idx]
        bntt = self.layer_bntt[idx]
        expected_feat = self.hparams.num_hidden

        # LayerNorm
        if ln is not None and tensor.shape[-1] == ln.normalized_shape[0]:
            if per_timestep:
                tensor = ln(tensor)
            else:
                # (T, B, F) -> (B, T, F) for LayerNorm across features
                tensor = tensor.permute(1, 0, 2)  # (B, T, F)
                tensor = ln(tensor)
                tensor = tensor.permute(1, 0, 2)  # (T, B, F)
        else:
            # If feature size doesn't match, skip LayerNorm for safety.
            # (This commonly happens for the first layer where input_size != num_hidden.)
            pass

        # BNTT
        if bntt is not None and tensor.shape[-1] == expected_feat:
            if per_timestep:
                if t is None:
                    raise ValueError("time index t required for per-timestep BNTT")
                tensor = bntt[t](tensor)
            else:
                normed = []
                for tt in range(tensor.shape[0]):  # iterate T
                    normed.append(bntt[tt](tensor[tt]))
                tensor = torch.stack(normed, dim=0)
        else:
            # If the feature-size doesn't match the BNTT expected features, skip it.
            pass

        return tensor

    def _temporal_proj(self, idx, t, x_seq, raw_input_t=None):
        skip_k = self.hparams.temporal_skip
        if skip_k == -1 or t < skip_k:
            proj = None
        else:
            prev = x_seq[t - skip_k]
            proj = self.temp_skip_projs[idx](prev)
        return proj

    def _layer_skip_proj(self, idx, i, t, track_layer_time, layer_outputs_time):
        if self.hparams.layer_skip <= 0 or i < self.hparams.layer_skip:
            return None

        if track_layer_time:
            skip_layer_out = layer_outputs_time[i - self.hparams.layer_skip][t]
        else:
            skip_layer_out = None

        if skip_layer_out is None:
            return None

        proj_mod = self.layer_skip_projs[i]
        if proj_mod is not None:
            return proj_mod(skip_layer_out)
        else:
            return skip_layer_out

    def _stack_last_layer_seq(self, layer_outputs_time, T, curr_input):
        if layer_outputs_time is not None:
            return torch.stack(layer_outputs_time[-1], dim=0)
        else:
            return curr_input.unsqueeze(0).repeat(T, 1, 1)

    def forward(self, x, params=None):
        self.spike_counts = []

        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"x must be 2D or 3D, got {x.shape}")

        batch, seq_len, _ = x.shape

        if params is not None:
            if params.dim() < 2 or params.shape[1] < self.num_params:
                raise ValueError(f"params must have shape (batch, >= {self.num_params})")
            params_expanded = params.unsqueeze(2)
            x = torch.cat([params_expanded, x], dim=1)
        elif seq_len <= self.num_params:
            raise ValueError(f"x sequence too short to contain static params, got seq_len={seq_len}")

        static_params = x[:, :self.num_params, 0]
        time_series = x[:, self.num_params:, 0]
        time_steps = time_series.shape[1]

        static_repeated = static_params.unsqueeze(1).repeat(1, time_steps, 1)
        time_series_features = time_series.unsqueeze(2)
        x_transformed = torch.cat([static_repeated, time_series_features], dim=2)

        x_seq = x_transformed.permute(1, 0, 2)
        T, B, F_in = x_seq.shape

        skip_k = self.hparams.temporal_skip
        neuron_type = getattr(self.hparams, "neuron_type", "LeakyParallel")

        # ------------------ SLSTM branch ------------------
        if neuron_type == "SLSTM":
            # (unchanged)
            num_layers = len(self.lif_layers)
            syn_states = []
            mem_states = []
            for layer in self.lif_layers:
                syn, mem = layer.reset_mem()
                syn_states.append(syn.to(x_seq.device))
                mem_states.append(mem.to(x_seq.device))

            track_layer_time = (skip_k != -1)
            layer_outputs_time = [ [] for _ in range(num_layers) ] if track_layer_time else None
            spike_acc = [0.0] * num_layers

            for t in range(T):
                curr_input = x_seq[t]  # (B, F_in or F_hidden)
                for i, layer in enumerate(self.lif_layers):
                    # --- ORDER: 1) skipping (temporal & layer), 2) normalization/BNTT, 3) layer call ---
                    input_to_layer = curr_input

                    # temporal skip (adds projected previous input sequence element)
                    prev_proj = self._temporal_proj(i, t, x_seq)
                    if prev_proj is not None:
                        input_to_layer = input_to_layer + prev_proj

                    # layer skip: use earlier layer output from the same time step (if tracking)
                    skip_proj = self._layer_skip_proj(i, i, t, track_layer_time, layer_outputs_time)
                    if skip_proj is not None:
                        input_to_layer = input_to_layer + skip_proj

                    # normalization & BNTT (per-timestep)
                    # Only applied if feature dims match the per-layer norm/BNTT expectations.
                    input_to_layer = self._norm_and_bntt(i, input_to_layer, t=t, per_timestep=True)

                    # now call layer with normalized/skipped input
                    spk, syn_states[i], mem_states[i] = layer(input_to_layer, syn_states[i], mem_states[i])

                    if track_layer_time:
                        layer_outputs_time[i].append(spk)

                    curr_input = spk
                    spike_acc[i] += spk.mean().item()

            last_layer_seq = self._stack_last_layer_seq(layer_outputs_time, T, curr_input)
            self.spike_counts = [c / T for c in spike_acc]
            x_seq = last_layer_seq

        # ------------------ Leaky (per-step) branch ------------------
        elif neuron_type == "Leaky":
            # (unchanged)
            num_layers = len(self.lif_layers)
            mem_states = [layer.init_leaky().to(x_seq.device) for layer in self.lif_layers]
            track_layer_time = (skip_k != -1)
            layer_outputs_time = [[] for _ in range(num_layers)] if track_layer_time else None
            spike_acc = [0.0] * num_layers
            curr_input = None

            for t in range(T):
                raw_input_t = x_seq[t]
                for i, layer in enumerate(self.lif_layers):
                    lin = self.leaky_linears[i]
                    if i == 0:
                        weighted_input = lin(raw_input_t)
                    else:
                        weighted_input = lin(curr_input)

                    prev_proj = self._temporal_proj(i, t, x_seq)
                    if prev_proj is not None:
                        weighted_input = weighted_input + prev_proj

                    skip_proj = self._layer_skip_proj(i, i, t, track_layer_time, layer_outputs_time)
                    if skip_proj is not None:
                        weighted_input = weighted_input + skip_proj

                    weighted_input = self._norm_and_bntt(i, weighted_input, t=t, per_timestep=True)

                    spk, mem_states[i] = layer(weighted_input, mem_states[i])

                    if track_layer_time:
                        layer_outputs_time[i].append(spk)

                    curr_input = spk
                    spike_acc[i] += spk.mean().item()

            last_layer_seq = self._stack_last_layer_seq(layer_outputs_time, T, curr_input)
            self.spike_counts = [c / T for c in spike_acc]
            x_seq = last_layer_seq

        # ------------------ RLeaky (per-step) branch ------------------
        elif neuron_type == "RLeaky":
            # (unchanged)
            num_layers = len(self.lif_layers)
            spk_states = []
            mem_states = []
            for layer in self.lif_layers:
                spk0, mem0 = layer.init_rleaky()
                spk_states.append(spk0.to(x_seq.device))
                mem_states.append(mem0.to(x_seq.device))

            track_layer_time = (skip_k != -1)
            layer_outputs_time = [[] for _ in range(num_layers)] if track_layer_time else None
            spike_acc = [0.0] * num_layers
            curr_input = None

            for t in range(T):
                raw_input_t = x_seq[t]
                for i, layer in enumerate(self.lif_layers):
                    lin = self.leaky_linears[i]
                    if i == 0:
                        weighted_input = lin(raw_input_t)
                    else:
                        weighted_input = lin(curr_input)

                    prev_proj = self._temporal_proj(i, t, x_seq)
                    if prev_proj is not None:
                        weighted_input = weighted_input + prev_proj

                    skip_proj = self._layer_skip_proj(i, i, t, track_layer_time, layer_outputs_time)
                    if skip_proj is not None:
                        weighted_input = weighted_input + skip_proj

                    weighted_input = self._norm_and_bntt(i, weighted_input, t=t, per_timestep=True)

                    spk, mem_states[i] = layer(weighted_input, spk_states[i], mem_states[i])
                    spk_states[i] = spk

                    if track_layer_time:
                        layer_outputs_time[i].append(spk)

                    curr_input = spk
                    spike_acc[i] += spk.mean().item()

            last_layer_seq = self._stack_last_layer_seq(layer_outputs_time, T, curr_input)
            self.spike_counts = [c / T for c in spike_acc]
            x_seq = last_layer_seq

        # ------------------ LeakyParallel / default sequence-processing neuron ------------------
        elif neuron_type == "SRC":
            # SRC handles the full sequence input (T, B, F). It returns (sout, (i,h,hs)).
            layer_outputs = []
            for i, lif in enumerate(self.lif_layers):
                input_seq = x_seq  # (T, B, F_in or F_hidden)

                # --- temporal skip projection (sequence-level) ---
                if skip_k == -1 or skip_k >= T:
                    prev_proj = None
                else:
                    prev = torch.zeros_like(x_seq)
                    prev[skip_k:] = x_seq[:-skip_k]
                    prev_proj = self.temp_skip_projs[i](prev)

                if prev_proj is not None:
                    input_seq = input_seq + prev_proj

                # --- layer-wise skip (non-time-tracked): add previous layer's output seq if available
                if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                    skip_layer_out = layer_outputs[i - self.hparams.layer_skip]
                    if skip_layer_out is not None:
                        if self.layer_skip_projs[i] is not None:
                            skip_proj = self.layer_skip_projs[i](skip_layer_out)
                        else:
                            skip_proj = skip_layer_out
                        input_seq = input_seq + skip_proj

                # --- Now normalization & BNTT across the sequence BEFORE calling the SRC layer
                input_seq = self._norm_and_bntt(i, input_seq, per_timestep=False)

                # feed into SRC layer (expects full sequence)
                lif_out, _states = lif(input_seq)  # lif_out: (T, B, F_hidden)

                # record spikes (keep numeric mean similar to other branches)
                # lif_out.mean() returns a tensor, convert to float for consistency
                try:
                    self.spike_counts.append(float(lif_out.mean().item()))
                except Exception:
                    # fallback: append raw tensor mean
                    self.spike_counts.append(lif_out.mean())

                # update x_seq for next layer
                x_seq = lif_out

                layer_outputs.append(x_seq)

        else:
            # existing LeakyParallel behavior (unchanged)
            layer_outputs = []
            for i, lif in enumerate(self.lif_layers):
                # --- ORDER: 1) compute temporal + layer skip on input sequence, 2) normalize, 3) feed into lif ---
                input_seq = x_seq  # (T, B, F_in or F_hidden)

                # temporal skip projection (sequence-level)
                if skip_k == -1 or skip_k >= T:
                    prev_proj = None
                else:
                    prev = torch.zeros_like(x_seq)
                    prev[skip_k:] = x_seq[:-skip_k]
                    prev_proj = self.temp_skip_projs[i](prev)

                if prev_proj is not None:
                    input_seq = input_seq + prev_proj

                # layer-wise skip (non-time-tracked): add previous layer's output seq if available
                if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                    skip_layer_out = layer_outputs[i - self.hparams.layer_skip]
                    if skip_layer_out is not None:
                        if self.layer_skip_projs[i] is not None:
                            skip_proj = self.layer_skip_projs[i](skip_layer_out)
                        else:
                            skip_proj = skip_layer_out
                        input_seq = input_seq + skip_proj

                # Now normalization & BNTT across the sequence BEFORE calling the layer
                input_seq = self._norm_and_bntt(i, input_seq, per_timestep=False)

                # feed into layer
                lif_out = lif(input_seq)  # (T, B, F_hidden)

                # record spikes
                self.spike_counts.append(lif_out.mean())

                # update x_seq for next layer
                x_seq = lif_out

                layer_outputs.append(x_seq)

        # x_seq is now (T, B, F_hidden). Project and return (B, T, outputs) squeezed as original.
        x_seq_b = x_seq.permute(1, 0, 2)  # (B, T, F_hidden)
        out = self.output_layer(x_seq_b).squeeze()
        return out

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        # loss = torch.nn.MSELoss()(outputs, y_batch.float())
        loss = self.loss(outputs, y_batch.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        spike_logs = {
            f"spikes/train_avg_layer_{i}": avg_spikes
            for i, avg_spikes in enumerate(self.spike_counts)
        }
        self.log_dict(spike_logs, on_step=False, on_epoch=True, sync_dist=True)

        # --- replace current gradient-aggregation block with this ---
        grad_means = {}
        for name, param in self.named_parameters():
            if param.grad is None:
                continue

            # name examples: "lif_layers.0.rleaky.weight", "leaky_linears.2.weight", "output_layer.weight"
            parts = name.split('.')

            # If it's one of our ModuleLists, try to keep the index (e.g. 'leaky_linears.2')
            module_lists = {
                "lif_layers",
                "leaky_linears",
                "layer_skip_projs",
                "temp_skip_projs",
                "layer_bntt",
                "layer_norms",
            }

            if parts[0] in module_lists and len(parts) > 1 and parts[1].isdigit():
                layer_name = f"{parts[0]}.{parts[1]}"   # e.g. "leaky_linears.2"
            elif parts[0] == "lif_layers" and len(parts) > 1:
                # lif_layers may have deeper nested names (e.g. lif_layers.0.<submodule>),
                # but we still want lif_layers.<index>
                layer_name = ".".join(parts[:2])
            else:
                # fallback to top-level module name for other params (e.g. output_layer)
                layer_name = parts[0]

            grad_means.setdefault(layer_name, []).append(param.grad.abs().mean().item())

        # average and prefix with a clear metric name
        grad_means = {
            f"grad_mean/{layer}": sum(vals) / len(vals)
            for layer, vals in grad_means.items()
        }
        self.log_dict(grad_means, on_step=True, on_epoch=False, sync_dist=False)


        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        # loss = torch.nn.MSELoss()(outputs, y_batch.float())
        loss = self.loss(outputs, y_batch.float())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch).detach()
    
        y_true = y_batch.float()
    
        # Mean Squared Error
        mse = F.mse_loss(outputs, y_true)
    
        # Mean Absolute Error
        mae = F.l1_loss(outputs, y_true)
    
        # R^2 score
        ss_res = torch.sum((y_true - outputs) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
    
        # Pearson correlation
        p_mean = outputs.mean(dim=0, keepdim=True)
        t_mean = y_true.mean(dim=0, keepdim=True)
        numerator = torch.sum((outputs - p_mean) * (y_true - t_mean))
        denominator = torch.sqrt(torch.sum((outputs - p_mean) ** 2) * torch.sum((y_true - t_mean) ** 2) + 1e-8)
        pearson = numerator / denominator
    
        # Log metrics
        self.log_dict({
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2,
            "test_pearson": pearson
        }, prog_bar=True, on_step=False, on_epoch=True)
    
        return {
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2,
            "test_pearson": pearson
        }

    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_kwargs = dict(self.hparams.optimizer_kwargs or {})
        optimizer_kwargs.setdefault("lr", self.hparams.lr)

        optimizer = self.hparams.optimizer_class(self.parameters(), **optimizer_kwargs)

        if self.hparams.scheduler_class is not None:
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.CosineAnnealingLR:
                self.hparams.scheduler_kwargs.setdefault("T_max", self.hparams.max_epochs)
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.ExponentialLR:
                self.hparams.scheduler_kwargs.setdefault("gamma", 0.95)
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.StepLR:
                self.hparams.scheduler_kwargs.setdefault("step_size", 5)
            scheduler = self.hparams.scheduler_class(optimizer, **self.hparams.scheduler_kwargs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )