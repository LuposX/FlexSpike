from typing import Optional, Union, Callable, Any, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl


class RNNLightning(pl.LightningModule):
    def __init__(
        self,
        num_hidden_layers: int = 2,
        num_hidden: int = 64,
        rnn_type: str = "lstm",  # "rnn", "gru", "lstm"
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        dropout: float = 0.0,
        max_epochs: int = 100,
        layer_skip: int = 0,
        use_layernorm: bool = False,
        scheduler_class=None,
        scheduler_kwargs: Optional[dict] = None,
        log_every_n_steps: int = 50,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        num_inputs: int = 1,
        loss_fn: Union[str, Callable] = "mse",
        loss_kwargs: Optional[dict] = None,
        num_outputs: Optional[int] = None,
        # NEW: how many future steps to predict in one-shot (1 = next step)
        predict_horizon: int = 1,
    ):
        super().__init__()
        # save hyperparameters so Lightning can checkpoint + log them
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
        self.save_hyperparameters(ignore=["train_dataset", "valid_dataset", "test_dataset"])


        hp = self.hparams  # convenience alias

        # build loss (store as module/func)
        self.loss_fn = self._build_loss(hp.loss_fn, hp.loss_kwargs or {})

        # safe resolution of rnn type
        RNN_TYPES = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }
        rnn_key = str(hp.rnn_type).lower()
        if rnn_key not in RNN_TYPES:
            valid = ", ".join(RNN_TYPES.keys())
            raise ValueError(f"Invalid rnn_type='{hp.rnn_type}'. Choose from: {valid}")
        rnn_class = RNN_TYPES[rnn_key]

        # build rnn layers
        self.rnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if hp.use_layernorm else None

        in_dim = int(hp.num_inputs)
        num_hidden = int(hp.num_hidden)
        num_hidden_layers = int(hp.num_hidden_layers)

        for i in range(num_hidden_layers):
            layer = rnn_class(
                input_size=in_dim,
                hidden_size=num_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.rnn_layers.append(layer)
            if hp.use_layernorm:
                self.layer_norms.append(nn.LayerNorm(num_hidden))
            in_dim = num_hidden  # subsequent layer input dim == hidden size

        # dropout between layers
        self.dropout_layer = nn.Dropout(hp.dropout) if (hp.dropout and hp.dropout > 0) else nn.Identity()

        # pre-create skip projection modules (one per layer index)
        self.skip_projs = nn.ModuleList([nn.Identity() for _ in range(num_hidden_layers)])
        if int(hp.layer_skip) > 0:
            for i in range(num_hidden_layers):
                src_idx = i - int(hp.layer_skip)
                if src_idx >= 0:
                    src_dim = num_hidden  # all layers use same hidden size here
                    tgt_dim = num_hidden
                    if src_dim != tgt_dim:
                        self.skip_projs[i] = nn.Linear(src_dim, tgt_dim)
                    else:
                        self.skip_projs[i] = nn.Identity()
                else:
                    self.skip_projs[i] = nn.Identity()

        # final projector -> predict last-step or predict_horizon steps (one-shot)
        out_dim = int(hp.num_outputs) if hp.num_outputs is not None else int(hp.num_inputs)
        ph = int(hp.predict_horizon) if getattr(hp, "predict_horizon", None) is not None else 1
        if ph < 1:
            raise ValueError("predict_horizon must be >= 1")
        # If predicting multiple steps, project to (out_dim * predict_horizon) and reshape later
        self.predict_horizon = ph
        self.out_dim = out_dim
        self.proj = nn.Linear(num_hidden, out_dim * self.predict_horizon)

    def _build_loss(self, loss_fn: Union[str, Callable], loss_kwargs: Dict[str, Any]):
        if callable(loss_fn):
            return loss_fn
        s = loss_fn.lower() if isinstance(loss_fn, str) else None
        if s in ("mse", "mse_loss", "mean_squared_error"):
            return nn.MSELoss(**loss_kwargs)
        if s in ("mae", "l1", "l1_loss", "mean_absolute_error"):
            return nn.L1Loss(**loss_kwargs)
        if s in ("bce", "binary_cross_entropy"):
            return nn.BCEWithLogitsLoss(**loss_kwargs)
        if s in ("cross_entropy", "nll"):
            return nn.CrossEntropyLoss(**loss_kwargs)
        raise ValueError(f"Unknown loss_fn string: {loss_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_inputs)
        returns:
          - if predict_horizon == 1: (batch, out_dim)
          - if predict_horizon > 1: (batch, predict_horizon, out_dim)
        """
        hp = self.hparams
        out = x
        prev_outs = []

        for i, layer in enumerate(self.rnn_layers):
            out, _ = layer(out)  # out: (batch, seq, hidden)

            if hp.use_layernorm:
                ln = self.layer_norms[i]
                out = ln(out)

            out = self.dropout_layer(out)

            if int(hp.layer_skip) and i - int(hp.layer_skip) >= 0:
                skip_from = prev_outs[i - int(hp.layer_skip)]
                proj = self.skip_projs[i]
                skip_proj = proj(skip_from)
                if skip_proj.shape == out.shape:
                    out = out + skip_proj

            prev_outs.append(out)

        # use last timestep hidden
        last = out[:, -1, :]  # (batch, hidden)

        yhat = self.proj(last)  # (batch, out_dim * predict_horizon)
        if self.predict_horizon == 1:
            # return (batch, out_dim)
            if self.out_dim == 1:
                return yhat.view(-1, 1)
            return yhat.view(yhat.size(0), self.out_dim)
        else:
            # return (batch, horizon, out_dim)
            return yhat.view(yhat.size(0), self.predict_horizon, self.out_dim)

    def _prepare_target_for_loss(self, y: torch.Tensor):
        """
        Normalize/reshape provided target `y` to match the model output shape:
        - If predict_horizon==1 -> return (batch, out_dim)
        - If predict_horizon>1 -> return (batch, predict_horizon, out_dim)
    
        The function tries to handle common shapes:
          - y dim == 3: (batch, seq_len_or_horizon, out_dim)
          - y dim == 2: (batch, out_dim) or (batch, horizon) or (batch, horizon*out_dim)
          - Special convenience: if y is 2D sequence (batch, seq_len) and we expect a
            single scalar next-step target (predict_horizon==1, out_dim==1), we will
            take the last timestep: y[:, -1] -> (batch, 1)
        """
        ph = self.predict_horizon
        out_dim = self.out_dim
    
        # If user provided sequence-style targets (batch, seq_len, out_dim)
        if y.dim() == 3:
            # If y has at least ph timesteps, take first ph timesteps as targets
            if y.size(1) >= ph:
                target = y[:, :ph, :]
                if ph == 1:
                    return target[:, -1, :]  # (batch, out_dim)
                return target  # (batch, ph, out_dim)
            else:
                raise ValueError(
                    f"Provided 3D target has seq_len={y.size(1)} < predict_horizon={ph}. "
                    "Either supply longer target sequences or reduce predict_horizon."
                )
    
        # If user provided 2D targets
        if y.dim() == 2:
            b, s = y.size()
            # --- Convenience: sequence-of-history provided but only next-step is needed ---
            # If y looks like a history sequence (s > 1) and we expect a single scalar
            # output (out_dim==1) and predict_horizon==1, take the last timestep.
            if s > 1 and ph == 1 and out_dim == 1:
                # treat y as (batch, seq_len) and use last timestep as target
                return y[:, -1].view(b, 1)
    
            # case: they provided single-step target with out_dim columns
            if s == out_dim and ph == 1:
                return y  # (batch, out_dim)
            # case: they provided single-step scalar per batch and out_dim==1
            if s == 1 and out_dim == 1 and ph == 1:
                return y.view(b, 1)
            # case: they provided horizon entries for scalar-output -> (batch, horizon) and out_dim==1
            if s == ph and out_dim == 1:
                return y.unsqueeze(-1)  # (batch, horizon, 1)
            # case: they provided flattened horizon*outs -> reshape
            if s == ph * out_dim:
                return y.view(b, ph, out_dim)
            # ambiguous -> raise instructive error
            raise ValueError(
                "Ambiguous target shape (2D). Expected one of:\n"
                f" - single-step target: (batch, {out_dim}) when predict_horizon==1\n"
                f" - multi-step flattened target: (batch, {ph * out_dim}) to be reshaped to (batch, {ph}, {out_dim})\n"
                f" - multi-step scalar target (when out_dim==1): (batch, {ph})\n"
                "Or provide a 3D target tensor (batch, horizon, out_dim).\n"
                f"Your target shape: {tuple(y.shape)}"
            )
    
        # higher dims invalid
        raise ValueError(f"Unsupported target tensor dims: {y.dim()}. Expected 2 or 3 dims.")


    def _compute_loss_and_log(self, yhat: torch.Tensor, y: torch.Tensor, prefix: str = "train"):
        """
        Standard loss computation and logging. yhat and returned target will have identical shapes.
        """
        # prepare target
        target = self._prepare_target_for_loss(y)

        # ensure shapes match (except possibly last-dim ordering for 1-step)
        if yhat.shape != target.shape:
            # attempt to squeeze/unsqueeze when out_dim==1 / single-step vs (batch,)
            raise ValueError(f"Model output shape {tuple(yhat.shape)} does not match target shape {tuple(target.shape)}. "
                             "See `_prepare_target_for_loss` and ensure your dataset's target is shaped correctly.")

        loss = self.loss_fn(yhat, target)
        # log
        self.log(f"{prefix}_loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self._compute_loss_and_log(yhat, y, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self._compute_loss_and_log(yhat, y, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self._compute_loss_and_log(yhat, y, prefix="test")
        return loss

    def configure_optimizers(self):
        hp = self.hparams
        opt = hp.optimizer_class(self.parameters(), lr=float(hp.lr), **(hp.optimizer_kwargs or {}))
        if hp.scheduler_class is None:
            return opt
        scheduler = hp.scheduler_class(opt, **(hp.scheduler_kwargs or {}))
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # dataloaders if datasets provided
    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=int(self.hparams.batch_size), shuffle=True)

    def val_dataloader(self):
        if self.valid_dataset is None:
            return None
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=int(self.hparams.batch_size), shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=int(self.hparams.batch_size), shuffle=False)