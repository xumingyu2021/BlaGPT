"""
- Use GeGLU instead of GLU
- Grouped Query Attention
- RMSNorm before attention for Query and Key
- Cap logits in attention
- Share embedding weights and output weights
- QKV Bias
- Soft attention logit capping
- Rotary Position Embedding
- zero-init projection layers

- Residual Attention
- Differential Attention

TODO:
- Pre and post RMSNorm
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cap(x, cap):
    return cap * torch.tanh(x / cap)


class GPTConfig:
    def __init__(
        self, vocab_size, n_embd, n_query_groups, n_head, n_layer, dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_query_groups = n_query_groups
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        rms = norm / math.sqrt(x.shape[-1])
        x_normed = x / (rms + self.eps)
        return self.weight * x_normed


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis

# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(*shape)

# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


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


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_query_groups = config.n_query_groups
        self.n_head = config.n_head
        self.dropout = config.dropout
        assert self.n_head % self.n_query_groups == 0

        self.head_dim = self.n_embd // self.n_head
        self.kv_heads = self.n_query_groups
        self.kv_dim = self.kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.soft_cap = 50.0

        self.rotary = Rotary(self.head_dim)

        # Zero-init the output projection
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim)

        cos, sin = self.rotary(q)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.n_head // self.kv_heads, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.n_head // self.kv_heads, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = soft_cap(att, self.soft_cap)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.final_soft_cap = 30.0

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Zero-init the output projection in attention layers
        if isinstance(module, GroupedQueryAttention):
            torch.nn.init.zeros_(module.out_proj.weight)
            torch.nn.init.zeros_(module.out_proj.bias)

        # Zero-init the second layer in MLP blocks
        if isinstance(module, FeedForward):
            torch.nn.init.zeros_(module.down_proj.weight)
            torch.nn.init.zeros_(module.down_proj.bias)

    def forward(self, idx, targets=None, return_logits=True):
        x = self.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            logits = soft_cap(logits, self.final_soft_cap)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    import unittest

    class TestGPT(unittest.TestCase):
        def setUp(self):
            self.config = GPTConfig(
                vocab_size=100,
                n_embd=32,
                n_query_groups=2,
                n_head=4,
                n_layer=2,
                block_size=8,
                dropout=0.1,
            )
            self.model = GPT(self.config)

        def test_forward_pass(self):
            batch_size, seq_length = 2, 6
            x = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
            logits, _ = self.model(x)
            self.assertEqual(
                logits.shape, (batch_size, seq_length, self.config.vocab_size)
            )

        def test_loss_calculation(self):
            batch_size, seq_length = 2, 6
            x = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
            y = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
            _, loss = self.model(x, y)
            self.assertIsInstance(loss.item(), float)

        def test_generate(self):
            x = torch.tensor([[1, 2, 3]], dtype=torch.long)
            output = self.model.generate(x, max_new_tokens=5)
            self.assertEqual(
                output.shape, (1, 8)
            )  # Input length (3) + max_new_tokens (5)

        def test_grouped_query_attention(self):
            batch_size, seq_length = 2, 6
            x = torch.randn(batch_size, seq_length, self.config.n_embd)
            attention = self.model.blocks[0].attn
            freqs_cis = self.model.freqs_cis[:seq_length]

            # Perform a forward pass
            output = attention(x, freqs_cis)
            self.assertEqual(output.shape, (batch_size, seq_length, self.config.n_embd))

        def test_qknorm(self):
            batch_size, seq_length = 2, 6
            x = torch.randn(batch_size, seq_length, self.config.n_embd)
            attention = self.model.blocks[0].attn
            freqs_cis = self.model.freqs_cis[:seq_length]

            # Check if RMSNorm is applied
            self.assertIsInstance(attention.q_norm, RMSNorm)
            self.assertIsInstance(attention.k_norm, RMSNorm)

            # Perform a forward pass
            output = attention(x, freqs_cis)
            self.assertEqual(output.shape, (batch_size, seq_length, self.config.n_embd))

        def test_rotary_embeddings(self):
            batch_size, seq_length = 2, 6
            x = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))

            # Check if freqs_cis is precomputed
            self.assertIsNotNone(self.model.freqs_cis)
            self.assertEqual(
                self.model.freqs_cis.shape,
                (
                    self.config.block_size * 2,
                    self.config.n_embd // (2 * self.config.n_head),
                ),
            )

            # Perform a forward pass
            logits, _ = self.model(x)

            # Check if the output has the correct shape
            self.assertEqual(
                logits.shape, (batch_size, seq_length, self.config.vocab_size)
            )

            # Test if the model can handle sequences of different lengths
            x_short = torch.randint(
                0, self.config.vocab_size, (batch_size, seq_length - 2)
            )
            logits_short, _ = self.model(x_short)
            self.assertEqual(
                logits_short.shape, (batch_size, seq_length - 2, self.config.vocab_size)
            )

        def test_weight_tying(self):
            # Check if the embedding and output weights are the same object
            self.assertIs(self.model.wte.weight, self.model.lm_head.weight)

            # Perform a forward pass
            batch_size, seq_length = 2, 6
            x = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
            logits, _ = self.model(x)

            # Check if the output has the correct shape
            self.assertEqual(
                logits.shape, (batch_size, seq_length, self.config.vocab_size)
            )

            # Verify that modifying the embedding weight affects the output
            original_output = logits.clone()
            self.model.wte.weight.data[0, 0] += 1.0
            new_logits, _ = self.model(x)
            self.assertFalse(torch.allclose(original_output, new_logits))

        def test_logit_soft_capping(self):
            batch_size, seq_length = 2, 6
            x = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))

            # Test attention logit capping
            attention = self.model.blocks[0].attn
            freqs_cis = self.model.freqs_cis[:seq_length]

            # Perform a forward pass through the attention layer
            x_emb = self.model.wte(x)
            att_output = attention(x_emb, freqs_cis)

            # We can't directly access the attention logits, so we'll just check the output shape
            self.assertEqual(
                att_output.shape, (batch_size, seq_length, self.config.n_embd)
            )

            # Test final layer logit capping
            logits, _ = self.model(x)
            self.assertTrue(torch.all(logits >= -self.model.final_soft_cap))
            self.assertTrue(torch.all(logits <= self.model.final_soft_cap))

        def test_zero_init(self):
            # Test zero-init of attention output projection
            for block in self.model.blocks:
                self.assertTrue(
                    torch.allclose(
                        block.attn.out_proj.weight,
                        torch.zeros_like(block.attn.out_proj.weight),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        block.attn.out_proj.bias,
                        torch.zeros_like(block.attn.out_proj.bias),
                    )
                )

            # Test zero-init of MLP second layer
            for block in self.model.blocks:
                self.assertTrue(
                    torch.allclose(
                        block.mlp.c_proj.weight,
                        torch.zeros_like(block.mlp.c_proj.weight),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        block.mlp.c_proj.bias, torch.zeros_like(block.mlp.c_proj.bias)
                    )
                )

            # Test non-zero init of other layers
            self.assertFalse(
                torch.allclose(
                    self.model.wte.weight, torch.zeros_like(self.model.wte.weight)
                )
            )
            for block in self.model.blocks:
                self.assertFalse(
                    torch.allclose(
                        block.mlp.c_fc.weight, torch.zeros_like(block.mlp.c_fc.weight)
                    )
                )
                self.assertFalse(
                    torch.allclose(
                        block.attn.q_proj.weight,
                        torch.zeros_like(block.attn.q_proj.weight),
                    )
                )
                self.assertFalse(
                    torch.allclose(
                        block.attn.k_proj.weight,
                        torch.zeros_like(block.attn.k_proj.weight),
                    )
                )
                self.assertFalse(
                    torch.allclose(
                        block.attn.v_proj.weight,
                        torch.zeros_like(block.attn.v_proj.weight),
                    )
                )

    unittest.main()
