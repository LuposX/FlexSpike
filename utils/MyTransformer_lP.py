import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_extra_params = 6  # Store extra parameters count

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Dynamically create bias to handle (T + N_extra) x (T + N_extra)
        full_size = config.block_size + self.n_extra_params  # New total length
        self.register_buffer("bias", torch.tril(torch.ones(full_size, full_size))
                             .view(1, 1, full_size, full_size))  # Adjust mask

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding size
        seq_len = T - self.n_extra_params  # Separate sequence length

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute self-attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask only to the sequence part
        att[:, :, :seq_len, :seq_len] = att[:, :, :seq_len, :seq_len].masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf')
        )

        # Ensure extra parameters are visible to all tokens
        att[:, :, seq_len:, :] = 0  # No masking for extra parameters

        # Softmax normalization
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Reassemble multi-head attention output
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=GELU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))
    
    def mlp_forward(self, x):
        m = self.mlp
        x = m.c_fc(x)
        x = m.act(x)
        x = m.c_proj(x)
        x = m.dropout(x)
        return x

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp_forward(self.ln_2(x))
        return x



class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)


class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.block_size = None
        C.n_extra_params = 6  # Number of additional parameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.block_size = config.block_size
        self.n_extra_params = config.n_extra_params  # Store extra parameter count

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given
        if type_given:
            config.merge_from_dict({
                'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
                'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512),
                'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(1, config.n_embd),  # Token embedding for sequence and extra parameters
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embedding only for sequence
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.regression_head = nn.Linear(config.n_embd, 1, bias=True)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        device = x.device
        extra_params = self.n_extra_params  # Retrieve extra parameter count
        b, t, c = x.size()  # (B, T + N_extra, 1)
        
        assert t > extra_params, "Total sequence length must be greater than the number of extra parameters"

        # Split input into sequence and extra parameters
        seq_len = t - extra_params
        seq_x = x[:, :seq_len, :]  # (B, T, 1)
        extra_x = x[:, seq_len:, :]  # (B, N_extra, 1)

        # Positional indices for sequence
        pos_seq = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_emb_seq = self.transformer.wpe(pos_seq)  # (1, T, n_embd)

        # Token embedding for sequence (B, T, n_embd)
        seq_tok_emb = self.transformer.wte(seq_x)
        seq_x = seq_tok_emb + pos_emb_seq  # (B, T, n_embd)

        # Token embedding for extra parameters (B, N_extra, n_embd)
        extra_emb = self.transformer.wte(extra_x)

        # **New Positional Embedding for Extra Parameters**
        pos_extra = torch.arange(seq_len, seq_len + extra_params, dtype=torch.long, device=device).unsqueeze(0)  # (1, N_extra)
        pos_emb_extra = self.transformer.wpe(pos_extra)  # (1, N_extra, n_embd)
        extra_emb = extra_emb + pos_emb_extra  # Add positional embeddings

        # Concatenate embeddings along sequence dimension
        x = torch.cat([seq_x, extra_emb], dim=1)  # (B, T + N_extra, n_embd)

        # Transformer processing
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Regression head (projects back to 1-dimensional output per token)
        outputs = self.regression_head(x).squeeze()
        return outputs[:, :seq_len]  # Return only the sequence-related outputs

