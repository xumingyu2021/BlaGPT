import math

import torch
import torch.nn.functional as F
from kernel.rotary import apply_rotary_emb as apply_rotary_emb_kernel
from modules.pattention import Pattention
from torch import nn
from torch.nn import functional as F

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from norms import RMSNorm


def soft_cap(x, cap):
    return x.div_(cap).tanh_().mul_(cap)


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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head  # New parameter for number of key-value heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.soft_cap = 50.0 if config.use_soft_logit_capping else 0.0

        # RMSNorm before q and k projections
        if config.rmsnorm_before_qk:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        if config.pos_encoding == "rotary":
            self.rotary = Rotary(self.head_dim, base=config.rope_theta)
        elif config.pos_encoding == "relative":
            self.rel_pos_emb = nn.Parameter(
                torch.zeros(2 * config.block_size - 1, self.head_dim)
            )
            nn.init.normal_(self.rel_pos_emb, std=0.02)
        elif config.pos_encoding == "none" or config.pos_encoding is None:
            pass
        else:
            raise ValueError(f"Unknown positional encoding: {config.pos_encoding}")

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.set_layers(config)

    def set_layers(self, config):
        # Projections for query, key, and value
        self.q_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias or config.use_qkv_bias
        )
        self.kv_proj = nn.Linear(
            config.n_embd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, q=None, mask=None):
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Update mask if provided
        if mask is not None:
            self.mask = mask

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply normalization and rotary embeddings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Prepare attention inputs
        q, k, v = self._prepare_qkv(q, k, v)

        # Compute attention
        if self.flash and self.soft_cap == 0:
            y = self._flash_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v, T)

        # Project output
        return self._project_output(y, B, T_q, C)

    def _project_query(self, x, B, T):
        return self.q_proj(x).view(B, T, self.n_head, self.head_dim)

    def _project_kv(self, x, B, T):
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        return kv.unbind(dim=2)

    def _apply_norm(self, q, k):
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def _apply_rotary(self, q, k, T_q, T):
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        cos, sin = self.rotary(k) if T_q != T else (cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    def _apply_relative_pos(self, q, k, T_q, T):
        # Get relative position embeddings
        pos_emb = self._get_rel_pos_emb(T_q, T)

        # Apply relative position embeddings
        q = q + pos_emb[:T_q].unsqueeze(0).unsqueeze(0)
        k = k + pos_emb[:T].unsqueeze(0).unsqueeze(0)

        return q, k

    def _get_rel_pos_emb(self, T_q, T):
        # Get relative position embeddings centered around each position
        seq_length = max(T_q, T)
        positions = torch.arange(seq_length, device=self.rel_pos_emb.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = (
            relative_positions + seq_length - 1
        )  # shift to all positive
        return self.rel_pos_emb[relative_positions]

    def _prepare_qkv(self, q, k, v):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat k,v for multi-query attention
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        return q, k, v

    def _flash_attention(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

    def _manual_attention(self, q, k, v, T):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.soft_cap > 0:
            att = soft_cap(att, self.soft_cap)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def _project_output(self, y, B, T_q, C):
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.resid_dropout(self.c_proj(y))


class PattentionSelfAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

    def set_layers(self, config):
        self.q_proj = Pattention(config)
        self.k_proj = Pattention(config)
        self.v_proj = Pattention(config)
        self.c_proj = Pattention(config)

    def _project_kv(self, x, B, T):
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        return k, v


class KVShiftingAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

        # Initialize KV shifting parameters
        # Following paper's initialization: randomly initialize from U(0,1)
        # and make them sum to 1
        self.alpha1 = nn.Parameter(torch.rand(self.n_kv_head))
        self.alpha2 = nn.Parameter(torch.ones(self.n_kv_head) - self.alpha1)
        self.beta1 = nn.Parameter(torch.rand(self.n_kv_head))
        self.beta2 = nn.Parameter(torch.ones(self.n_kv_head) - self.beta1)

    def _shift_kv(self, x):
        """Perform shifting operation on key/value tensors.
        Shifts the sequence by padding a zero at the beginning and dropping last element.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_kv_head, head_dim)

        Returns:
            Shifted tensor of same shape
        """
        # Get shifted version by padding front and removing last element
        # Keep same dimensions by dropping last element after padding
        x_shifted = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        return x_shifted

    def _project_kv(self, x, B, T):
        """Override parent's _project_kv to add KV shifting.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            B: Batch size
            T: Sequence length

        Returns:
            Tuple of processed key and value tensors
        """
        # Get initial K,V projections using parent method
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)

        # Get shifted versions
        k_shifted = self._shift_kv(k)
        v_shifted = self._shift_kv(v)

        # Combine original and shifted versions with learned parameters
        k = (
            self.alpha1.view(1, 1, -1, 1) * k
            + self.alpha2.view(1, 1, -1, 1) * k_shifted
        )
        v = self.beta1.view(1, 1, -1, 1) * v + self.beta2.view(1, 1, -1, 1) * v_shifted

        return k, v


"""
Differential Attention - WIP (Getting OOM)
"""


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
