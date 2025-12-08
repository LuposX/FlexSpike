import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Any
from utils.evaluation import Evaluator

# ===============================================================================
# ======================== Lightning Wrapper for PSNN ===========================
# ===============================================================================

class LightningPrintedSpikingNetwork(pl.LightningModule):
    def __init__(self, topology, args, model_class, ckpt_path, train_loader, valid_loader, test_loader, surrogate_gradient, loss_fn=None, train_dataset=None, valid_dataset=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class', 'ckpt_path', 'loss_fn'])

        self.args = args
        self.network = PrintedSpikingNeuralNetwork(topology, args, model_class, ckpt_path, surrogate_gradient, train_dataset, valid_dataset)

        # loss_fn expects (model, x, y) -> scalar (matches your LFLoss)
        self.loss_fn = loss_fn if loss_fn is not None else LFLoss(args)

        # evaluator returns (acc, power)
        self.evaluator = Evaluator(args)

        num_classes = topology[-1]
        #self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        #self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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
                factor=getattr(self.args, "LR_DECAY", 0.1),
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

    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.network, x, y)
        valid_acc, valid_power = self.evaluator(self.network, x, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_power", valid_power, on_step=False, on_epoch=True, prog_bar=False)

        return {"val_loss": loss}

    
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
    def __init__(self, args, model_class, ckpt_path, surrogate_gradient, train_dataset, valid_dataset):
        super().__init__()
        self.args = args

        # Load frozen spike generator
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

        # Define raw trainable parameters (unconstrained)
        self.raw_params = nn.Parameter(torch.randn(1, 6))

        # Define target per-parameter ranges (shape: (6,))
        self.low = torch.tensor([0.1, 0.1, 0.1, 0.4, 0.4, 0.6], device=self.DEVICE)
        self.high = torch.tensor([1.0, 1.0,  1.0, 1.0, 1.0,  1.0], device=self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def transform_params(self):
        # Apply tanh transformation and scale to [low, high] for each parameter
        # raw_params shape: (1, 6) → transformed_params shape: (1, 6)
        r = (self.high - self.low) / 2
        c = (self.high + self.low) / 2
        return c + r * torch.tanh(self.raw_params)

    def forward(self, x):
        batch_size = x.shape[0]
        T = x.shape[2]

        # Transform and expand trainable parameters
        extra_params = self.transform_params()  # (1, 6)
        #print(extra_params)
        expanded_params = extra_params.expand(batch_size, -1)  # (B, 6)
        expanded_params = expanded_params.unsqueeze(2).expand(-1, -1, T)  # (B, 6, T)

        # Concatenate with input
        x = torch.cat([x, expanded_params], dim=1)  # (B, C+6, T)
        return self.spike_generator(x)

    def UpdateArgs(self, args):
        self.args = args


# ===============================================================================
# ============================== SG Layer =======================================
# ===============================================================================

class SGLayer(torch.nn.Module):
    def __init__(self, N, args, model_class, ckpt_path, surrogate_gradient, train_dataset, valid_dataset):
        super().__init__()
        self.args = args
        self.SG_Group = torch.nn.ModuleList(
            [pSpikeGenerator(args, model_class, ckpt_path, surrogate_gradient, train_dataset, valid_dataset) for _ in range(N)])

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        result = []
        for n in range(len(self.SG_Group)):
            x_temp = x[:, n, :].unsqueeze(-1)
            result.append(self.SG_Group[n](x_temp))
        return torch.stack(result).permute(1, 0, 2)

    def UpdateArgs(self, args):
        self.args = args


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
    def __init__(self, n_in, n_out, args, INV, model_class, ckpt_path,  surrogate_gradient, train_dataset, valid_dataset):
        super().__init__()
        self.args = args
        # define spike generators
        self.SG = SGLayer(n_out, args, model_class, ckpt_path, surrogate_gradient, train_dataset, valid_dataset)
        # define nonlinear circuits
        self.INV = INV
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/10. + args.gmin
        theta[-2, :] = args.gmax - theta[-2, :]
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

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

    def UpdateArgs(self, args):
        self.args = args

# ===============================================================================
# ======================== Printed Neural Network ===============================
# ===============================================================================


class PrintedSpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, topology, args, model_class, ckpt_path,  surrogate_gradient, train_dataset, valid_dataset):
        super().__init__()
        self.args = args

        self.INV = Inv(args)

        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module(str(i)+'_pLayer', pLayer(topology[i], topology[i+1], args, self.INV,  model_class, ckpt_path, surrogate_gradient, train_dataset, valid_dataset))

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
        weights = [p for name, p in self.named_parameters()
                if name.endswith('theta_') or name.endswith('beta') or name.endswith('raw_params')]
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
