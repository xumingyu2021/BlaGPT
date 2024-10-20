import math

import torch
import torch.nn.functional as F
from kernel.rotary import apply_rotary_emb as apply_rotary_emb_kernel
from torch import nn
from torch.nn import functional as F

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from norms import RMSNorm


def soft_cap(x, cap):
    return cap * torch.tanh(x / cap)


def soft_cap_mod(score, b, h, q_idx, kv_idx):
    score = score / 50
    score = torch.tanh(score)
    score = score * 50
    return score


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # RMSNorm before q and k projections
        if config.rmsnorm_before_qk:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        if config.use_rotary_emb:
            self.rotary = Rotary(self.head_dim)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)

        if hasattr(self, "rotary"):
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalGroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head  # New parameter for number of key-value heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.soft_cap = 50.0 if config.use_soft_logit_capping else 0.0

        # Projections for query, key, and value
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(
            config.n_embd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias,
        )

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # RMSNorm before q and k projections
        if config.rmsnorm_before_qk:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        if config.use_rotary_emb:
            self.rotary = Rotary(self.head_dim)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Project queries, keys, and values
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)  # (B, nh, T, hs)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)  # (B, T, n_kv_head, hs)

        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)

        if hasattr(self, "rotary"):
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, hs)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, hs)

        # Repeat keys and values to match the number of query heads
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)  # (B, nh, T, hs)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)  # (B, nh, T, hs)

        # Perform attention
        if self.flash and self.soft_cap == 0:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.soft_cap > 0:
                att = soft_cap(att, self.soft_cap)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)

        # Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


"""
Differential Attention - WIP
"""


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        config,
        depth,
    ):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        # num_heads set to half of Transformer's #heads
        self.num_kv_heads = (
            config.n_kv_head if config.n_kv_head is not None else self.num_heads
        )
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_heads // 2
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embed_dim, self.embed_dim // self.n_rep, bias=False
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.embed_dim // self.n_rep, bias=False
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim)
        self.rotary = Rotary(self.head_dim)

    def forward(
        self,
        x,
        attn_mask=None,
    ):
        bsz, tgt_len, _ = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        cos, sin = self.rotary(q)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        attn = self.out_proj(attn)
        return attn
