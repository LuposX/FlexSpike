import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

"""
PyTorch implementation of the SRC (Spiking Recurrent Cell) neuron described in
"Spike-based computation using classical recurrent neural networks".

API summary (read the code for full details):
- SRC(input_size, hidden_size, alpha=0.9, rho=3., r=2., rs=-7.,
      z=0., zhyp_s=0.9, zdep_s=0., bh_init=-6., bh_max=-4., detach_rec=True,
      relu_bypass=True)

- forward(sin, states=None)
    sin: Tensor of shape (seq_len, batch, input_size)
    states: optional tuple (i0, h0, hs0) each of shape (batch, hidden_size)
    returns: sout (seq_len, batch, hidden_size), (i, h, hs)

Notes and implementation choices made to match the paper's description:
- Ws is a learnable weight matrix of shape (hidden_size, input_size).
- Only bias `bh` and Ws are registered as learnable parameters. Other scalars
  are fixed attributes.
- The leaky integrator i[t] = alpha * i[t-1] + Ws @ sin[t] is implemented and
  gradients are allowed to flow through the integrator across time.
- To prevent gradients from flowing through the recurrent connections of h and
  hs across time (as described in the paper), the previous h and hs are
  detached before being used in recurrence (controlled by detach_rec flag).
- The output spike isolation uses a ReLU in the forward pass. When
  relu_bypass=True, the backward pass uses a straight-through estimator so
  gradients are passed through as if the activation were identity (i.e. d/dh
  sout = 1 for all h), matching the paper's trick.
- The neuron bias `bh` is clamped to be <= bh_max at each forward call.
- zs (the slow timescale gate) is computed from the previous h via a
  Heaviside-step-like gating (implemented with a threshold at 0.5). We use
  zhyp_s when h_prev <= 0.5 and zdep_s when h_prev > 0.5. For stability and
  straightforward computation we compute zs from h_prev (this avoids circular
  dependency and matches the intended voltage-dependence).

"""



class _StraightThroughReLU(torch.autograd.Function):
    """ReLU forward, but backward returns gradient * identity (pass-through).

    Forward: out = max(h, 0)
    Backward: grad_h = grad_output * 1  (for all h)
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradients through unchanged (straight-through estimator)
        grad_input = grad_output.clone()
        return grad_input


class SRC(nn.Module):
    """Spiking Recurrent Cell (SRC) layer.

    This module processes a sequence of input pulses and returns the sequence of
    output pulses (sout) together with the final states (i, h, hs).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        alpha: float = 0.9,
        rho: float = 3.0,
        r: float = 2.0,
        rs: float = -7.0,
        z: float = 0.0,
        zhyp_s: float = 0.9,
        zdep_s: float = 0.0,
        bh_init: float = -6.0,
        bh_max: float = -4.0,
        detach_rec: bool = True,
        relu_bypass: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable weights
        self.Ws = nn.Parameter(torch.empty(hidden_size, input_size))
        nn.init.kaiming_uniform_(self.Ws, a=math.sqrt(5))

        # Learnable bias bh (clamped at forward time)
        self.bh = nn.Parameter(torch.full((hidden_size,), bh_init))
        self.bh_max = float(bh_max)

        # Fixed hyper-parameters (not learnable)
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.rho = float(rho)
        self.r = float(r)
        self.rs = float(rs)
        self.z = float(z)
        self.zhyp_s = float(zhyp_s)
        self.zdep_s = float(zdep_s)

        # Behavior flags
        self.detach_rec = bool(detach_rec)
        self.relu_bypass = bool(relu_bypass)

        # choose ReLU autograd behavior
        if self.relu_bypass:
            self._relu = _StraightThroughReLU.apply
        else:
            self._relu = torch.relu

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Ws, a=math.sqrt(5))
        with torch.no_grad():
            self.bh.fill_(-6.0)

    def forward(self, sin: Tensor, states: Optional[Tuple[Tensor, Tensor, Tensor]] = None):
        """Forward pass over a sequence.

        Args:
            sin: Tensor with shape (seq_len, batch, input_size)
            states: optional tuple (i0, h0, hs0) each (batch, hidden_size). If
                    None they are initialized to zeros.

        Returns:
            sout: (seq_len, batch, hidden_size) spike outputs (float tensor)
            (i, h, hs): final states (each (batch, hidden_size))
        """
        if sin.dim() != 3:
            raise ValueError("sin must have shape (seq_len, batch, input_size)")
        seq_len, batch, input_size = sin.shape
        if input_size != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {input_size}")

        device = sin.device

        if states is None:
            i = torch.zeros(batch, self.hidden_size, device=device)
            h = torch.zeros(batch, self.hidden_size, device=device)
            hs = torch.zeros(batch, self.hidden_size, device=device)
        else:
            i, h, hs = states

        # clamp bias to be <= bh_max
        bh_clamped = torch.clamp(self.bh, max=self.bh_max)

        # pre-allocate outputs
        sout_seq = []

        for t in range(seq_len):
            sin_t = sin[t]  # (batch, input_size)

            # Integrator: i[t] = alpha * i[t-1] + Ws @ sin[t]
            # Ws is (hidden, input); we want (batch, hidden) result
            delta = torch.matmul(sin_t, self.Ws.t())  # (batch, hidden)
            i = self.alpha * i + delta

            # x[t] = rho * tanh(i / rho)
            x = self.rho * torch.tanh(i / self.rho)

            # detach recurrent states so gradients don't flow through them in time
            if self.detach_rec:
                h_prev = h.detach()
                hs_prev = hs.detach()
            else:
                h_prev = h
                hs_prev = hs

            # compute zs from previous h (voltage-dependent step)
            # zs = zhyp_s + (zdep_s - zhyp_s) * H(h_prev - 0.5)
            # H implemented as (h_prev > 0.5).float()
            H = (h_prev > 0.5).float()
            zs = self.zhyp_s + (self.zdep_s - self.zhyp_s) * H

            # candidate hidden state: h_candidate = tanh(x + r * h_prev + rs * hs_prev + bh)
            preact = x + self.r * h_prev + self.rs * hs_prev + bh_clamped
            h_candidate = torch.tanh(preact)

            # update h and hs
            h = self.z * h_prev + (1.0 - self.z) * h_candidate
            hs = zs * hs_prev + (1.0 - zs) * h_prev

            # output spikes (forward ReLU, backward either bypassed or normal)
            sout_t = self._relu(h)
            sout_seq.append(sout_t)

        sout = torch.stack(sout_seq, dim=0)
        return sout, (i, h, hs)