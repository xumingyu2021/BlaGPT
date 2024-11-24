import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions import Rotary, apply_rotary_emb
from causal_conv1d import causal_conv1d_fn
from coqpit import Coqpit
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from utils import register_model


@dataclass
class HymbaConfig(Coqpit):
    block_size: int = 1024
    vocab_size: int = 50304
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    ssm_state_size: int = 8
    n_hidden: int = 768 * 4
    num_full_attn_layers: int = 3
    n_meta_tokens: int = 128
    layer_norm_eps: float = 1e-5
    tie_embed_weights: bool = True
    # Added configuration parameters for gradient clipping
    ssm_grad_clip_min: float = -3.0
    ssm_grad_clip_max: float = 3.0
    ssm_d_clip_min: float = 1e-4
    ssm_d_clip_max: float = 1.0


class HymbaSelfAttention(nn.Module):
    def __init__(self, config: HymbaConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head

        assert (
            self.n_head % self.n_kv_head == 0
        ), "n_head must be divisible by n_kv_head"

        self.num_key_value_heads = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.layer_idx = layer_idx
        self.use_full_attn = layer_idx in [
            0,
            config.n_layer // 2,
            config.n_layer - 1,
        ]

        # Adjusted scale factor for grouped queries
        self.scale = (self.head_dim * self.n_kv_head / self.n_head) ** -0.5

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.n_embd, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

        self.rotary = Rotary(self.head_dim)

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Repeat key/value heads to match the number of attention heads.
        Input shape: [batch_size, num_key_value_heads, seq_len, head_dim]
        Output shape: [batch_size, n_head, seq_len, head_dim]
        """
        if self.n_kv_head == 1:
            return hidden_states.expand(-1, self.n_head, -1, -1)

        batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch_size, num_kv_heads, self.n_kv_head, seq_len, head_dim
        )
        hidden_states = hidden_states.reshape(
            batch_size, num_kv_heads * self.n_kv_head, seq_len, head_dim
        )
        return hidden_states

    def _apply_rotary(self, q, k, T_q, T):
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        cos, sin = self.rotary(k) if T_q != T else (cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if past_key_value is None else None

        # Repeat KV heads to match number of attention heads
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        query_states, key_states = self._apply_rotary(
            query_states, key_states, query_states.shape[1], key_states.shape[1]
        )

        # Compute attention with sliding window if needed
        if not self.use_full_attn and kv_seq_len > self.block_size:
            attn_output = self._sliding_window_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                kv_seq_len,
                q_len,
                bsz,
            )
        else:
            # Full attention
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.n_embd)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _sliding_window_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        kv_seq_len,
        q_len,
        bsz,
    ):
        # Initialize output tensor
        attn_output = torch.zeros(
            bsz,
            self.n_head,
            q_len,
            self.head_dim,
            device=query_states.device,
            dtype=query_states.dtype,
        )

        window_size = self.block_size
        for i in range(q_len):
            window_start = max(0, min(i - window_size // 2, kv_seq_len - window_size))
            window_end = min(kv_seq_len, window_start + window_size)

            local_q = query_states[:, :, i : i + 1]
            local_k = key_states[:, :, window_start:window_end]
            local_v = value_states[:, :, window_start:window_end]

            local_attn = torch.matmul(local_q, local_k.transpose(-1, -2)) * self.scale
            if attention_mask is not None:
                local_mask = attention_mask[:, :, i : i + 1, window_start:window_end]
                local_attn = local_attn + local_mask

            local_attn = F.softmax(local_attn, dim=-1)
            attn_output[:, :, i : i + 1] = torch.matmul(local_attn, local_v)

        return attn_output


# from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path
            and causal_conv1d_fn is not None
            and inference_params is None
        ):  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(x, (self.d_conv - x.shape[-1], 0))
                )  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out


class HymbaBlock(nn.Module):
    def __init__(self, config: HymbaConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.layer_idx = layer_idx

        self.input_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attention = HymbaSelfAttention(config, layer_idx)
        self.ssm = MambaLayer(
            d_model=config.n_embd,
            d_state=config.ssm_state_size,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=next(self.parameters()).device if self.parameters() else "cpu",
        )

        # Simplified layer norm structure following paper
        self.post_attention_layernorm = nn.LayerNorm(
            config.n_embd, eps=config.layer_norm_eps
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_hidden),
            nn.GELU(),
            nn.Linear(config.n_hidden, config.n_embd),
        )

        # Learnable scaling factors for hybrid fusion with proper initialization
        self.attn_scale = nn.Parameter(torch.ones(1) / math.sqrt(2.0))
        self.ssm_scale = nn.Parameter(torch.ones(1) / math.sqrt(2.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Layer norm before hybrid heads
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel processing through attention and SSM heads
        attn_output, past_key_value = self.attention(
            hidden_states, attention_mask, past_key_value
        )
        ssm_output = self.ssm(hidden_states)

        # Normalized fusion following paper's equations
        attn_norm = F.normalize(attn_output, dim=-1)
        ssm_norm = F.normalize(ssm_output, dim=-1)

        combined_output = attn_norm * self.attn_scale + ssm_norm * self.ssm_scale

        # First residual connection
        hidden_states = residual + combined_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Second residual connection
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class HymbaModel(nn.Module):
    def __init__(self, config: HymbaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)

        # Improved meta tokens initialization following paper
        meta_tokens = torch.zeros(1, config.n_meta_tokens, config.n_embd)
        # Initialize with learned patterns as described in paper Section 2.3
        position = torch.arange(config.n_meta_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * -(math.log(10000.0) / config.n_embd)
        )
        meta_tokens[0, :, 0::2] = torch.sin(position * div_term)
        meta_tokens[0, :, 1::2] = torch.cos(position * div_term)
        self.meta_tokens = nn.Parameter(meta_tokens)

        # Initialize KV cache sharing groups
        n_shared_groups = config.n_layer // 2  # Share KV cache every 2 layers
        self.kv_cache_groups = [i // 2 for i in range(config.n_layer)]

        self.layers = nn.ModuleList(
            [HymbaBlock(config, i) for i in range(config.n_layer)]
        )

        self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

        # Special initialization for meta tokens projection
        self._init_meta_token_projection()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _init_meta_token_projection(self):
        # Initialize special projection for meta tokens
        with torch.no_grad():
            std = 1.0 / math.sqrt(self.config.n_embd)
            self.meta_tokens.data.uniform_(-std, std)

    def _share_kv_cache(self, past_key_values):
        """Implement KV cache sharing between layers"""
        if not past_key_values:
            return past_key_values

        shared_past_key_values = []
        for group_idx in range(max(self.kv_cache_groups) + 1):
            # Find first layer in group
            first_layer_idx = self.kv_cache_groups.index(group_idx)
            group_kv = past_key_values[first_layer_idx]
            shared_past_key_values.extend(
                [group_kv] * self.kv_cache_groups.count(group_idx)
            )

        return shared_past_key_values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_length = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepend meta tokens
        meta_tokens = self.meta_tokens.expand(batch_size, -1, -1)
        hidden_states = torch.cat([meta_tokens, hidden_states], dim=1)

        # Initialize or use provided past_key_values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        else:
            # Apply KV cache sharing
            past_key_values = self._share_kv_cache(past_key_values)

        # Store new key/values
        new_past_key_values = []

        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if i < len(past_key_values) else None
            hidden_states, past_key_value = layer(
                hidden_states, attention_mask, layer_past
            )
            new_past_key_values.append(past_key_value)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Remove meta tokens from output
        hidden_states = hidden_states[:, self.config.n_meta_tokens :, :]

        return hidden_states, new_past_key_values


class HymbaForCausalLM(nn.Module):
    def __init__(self, config: HymbaConfig):
        super().__init__()
        self.config = config

        self.model = HymbaModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_embed_weights:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        bs, seq_len = idx.size()
        device = idx.device

        past_key_values = None

        # Create causal mask for the sequence length plus meta tokens
        total_seq_len = seq_len + self.config.n_meta_tokens
        mask = torch.full(
            (bs, 1, total_seq_len, total_seq_len), float("-inf"), device=device
        )
        mask = torch.triu(mask, diagonal=1)  # zeros on and below diagonal, -inf above

        # Allow meta tokens to attend to each other
        meta_size = self.config.n_meta_tokens
        mask[:, :, :meta_size, :meta_size] = 0

        hidden_states, new_past_key_values = self.model(
            idx, attention_mask=mask, past_key_values=past_key_values
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )

        return logits, loss


@register_model
def register_hymba():
    return HymbaConfig, HymbaForCausalLM


if __name__ == "__main__":

    def create_sample_model():
        """Create a small Hymba model for demonstration"""
        config = HymbaConfig(
            vocab_size=32000,
            n_embd=768,  # Smaller for demonstration
            n_layer=12,
            n_head=12,
            n_kv_head=4,
            ssm_state_size=16,
            n_hidden=2048,
            block_size=1024,
            num_full_attn_layers=3,
            n_meta_tokens=128,
        )

        model = HymbaForCausalLM(config)
        return model

    def generate_sample_input(batch_size=2, seq_length=512, vocab_size=32000):
        """Generate sample inputs for testing"""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        return input_ids, labels

    # Example training step
    def training_step(model, optimizer, input_ids, labels):
        optimizer.zero_grad()
        logits, loss = model(idx=input_ids, targets=labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = create_sample_model()
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Generate sample data
    input_ids, labels = generate_sample_input()
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Training loop
    model.train()
    for epoch in range(3):
        loss = training_step(model, optimizer, input_ids, labels)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
