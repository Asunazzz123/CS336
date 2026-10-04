"""Transformer LM building blocks (CS336 assignment 1, handout §3).

Attribute names below are not arbitrary: they must match the state_dict keys
that tests/adapters.py receives (e.g. ``attn.q_proj.weight``, ``ln1.weight``,
``layers.0.ffn.w1.weight``). With matching names the adapter can simply do
``module.load_state_dict(weights)``.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from einops import rearrange
from train.utils import softmax

class Linear(nn.Module):
    """y = x W^T, no bias (handout §3.3.2).

    Weight is stored as (out_features, in_features), same as nn.Linear.
    Init: N(0, 2 / (d_in + d_out)) truncated at ±3σ (handout §3.3.1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. 先分配一个未初始化的张量, device/dtype 在这里设置
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)

        # 2. 按 handout 的方差截断初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

        # 3. 包成 nn.Parameter 并挂到 self 上 -> 自动进入 parameters() 和 state_dict()
        #    state_dict key 就是属性名 "weight"
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., in_features) -> (..., out_features)."""
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token id -> d_model vector (handout §3.3.3).

    weight: (num_embeddings, embedding_dim); init N(0, 1) truncated at [-3, 3].
    forward(token_ids: Int[..., ]) -> (..., embedding_dim)   # 直接按行索引
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)

        # 初始化为正态分布
        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
        self.weight = nn.Parameter(weight)
        # raise NotImplementedError

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm (handout §3.4.1). weight: (d_model,), init to ones.

    forward: 先 `x.to(torch.float32)` 算 rms, 乘 weight 后再 `.to(in_dtype)`,
    避免 bf16/fp16 下 x**2 溢出。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        # raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # norm(x) = sqrt(bar{x^2}+eps)
        var = x.square().mean(dim=-1, keepdim=True) + self.eps
        # rmsnorm = weight * x / norm(x)
        rmsnorm = ((torch.rsqrt(var) * x) * self.weight).to(in_dtype)
        return rmsnorm

        # raise NotImplementedError


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) * (W3 x) ) (handout §3.4.2).

        w1: Linear(d_model, d_ff)
        w3: Linear(d_model, d_ff)
        w2: Linear(d_ff, d_model)
    训练时 d_ff 取 8/3 * d_model 并向上取到 64 的倍数; 测试会直接传 d_ff。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)

        # raise NotImplementedError

    def Silu(self, input: Tensor) -> Tensor:
        return torch.div(input,1+torch.exp(-input))

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.Silu(self.w1(x)) * self.w3(x))
        # raise NotImplementedError


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (handout §3.4.3).

    构造时预先算好 cos/sin 表, 形状 (max_seq_len, d_k // 2):
        inv_freq[i] = theta ** (-2i / d_k)
        angle[pos, i] = pos * inv_freq[i]
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    forward(x: (..., seq_len, d_k), token_positions: (..., seq_len)):
        用 token_positions 去索引 cos/sin, 对 x 的偶/奇维做旋转

    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 频率 w_i = theta * (-2i/d_k)
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        # 生成位置编码 0～max_seq_len
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]

        # 将cos/sin 注册为非训练状态
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        # 将输入token 正弦编码
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # Positions may omit head dimensions: (B, S) -> (B, 1, S, d_k/2).
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        original_dtype = x.dtype
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        cos, sin = cos.to(work.dtype), sin.to(work.dtype)
        even, odd = work[..., 0::2], work[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(original_dtype)


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal MHA with RoPE on Q/K (handout §3.5.3).

    q_proj / k_proj / v_proj / output_proj, 各为 Linear(d_model, d_model)。
    rope: RotaryPositionalEmbedding(theta, d_k=d_model // num_heads, max_seq_len);


    forward(x: (..., seq_len, d_model), token_positions: (..., seq_len) | None):
        1. 投影后 reshape 成 (..., num_heads, seq_len, d_k)
        2. token_positions 为 None 时用 arange(seq_len)
        3. Q/K 过 rope, V 不过
        4. causal mask = tril(ones(seq_len, seq_len, dtype=bool)) 在类内部构造,
           不依赖外部传入
        5. SDPA -> 合并 head -> output_proj
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_k = d_model // num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            max_seq_len=max_seq_len,
            d_k=d_k,
            device=device
        )
        # raise NotImplementedError

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        Q = rearrange(
            self.q_proj(x),
            "... s (h d) -> ... h s d",
            h=self.num_heads
        )
        K = rearrange(
            self.k_proj(x),
            "... s (h d) -> ... h s d",
            h=self.num_heads
        )
        V = rearrange(
            self.v_proj(x),
            "... s (h d) -> ... h s d",
            h=self.num_heads
        )
        if (token_positions is None):
            token_positions = torch.arange(x.shape[-2], device=x.device)

        Q_rope = self.rope(Q, token_positions)
        K_rope = self.rope(K, token_positions)
        seq_len = x.shape[-2]

        mul = torch.matmul(Q_rope,K_rope.transpose(-2,-1)) / torch.sqrt(torch.tensor(K_rope.shape[-1],device = K_rope.device, dtype = K_rope.dtype))
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool,device=x.device).tril()
        mask = torch.logical_not(causal_mask)
        mask_mul = mul.masked_fill(mask,-torch.inf)
        attn = torch.matmul(softmax(mask_mul,dim=-1),V)
        attn = rearrange(attn, "... h s d -> ... s (h d)")
        return self.output_proj(attn)
        # raise NotImplementedError


class TransformerBlock(nn.Module):
    """Pre-norm block (handout §3.5, Figure 2).

    属性名: ln1, attn, ln2, ffn  (匹配 ln1.weight / attn.*.weight / ffn.w*.weight)
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.CMHA = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        # raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        x = torch.add(x,self.CMHA(self.ln1(x)))
        output = torch.add(x,self.ffn(self.ln2(x)))
        return output
        # raise NotImplementedError


class TransformerLM(nn.Module):
    """Full LM (handout §3.6, Figure 1).

    属性名:
        token_embeddings: Embedding(vocab_size, d_model)
        layers: nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
                -> key 为 layers.{i}.xxx
        ln_final: RMSNorm(d_model)
        lm_head: Linear(d_model, vocab_size)

    forward(in_indices: (batch, seq_len)) -> logits (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        if in_indices.ndim != 2:
            raise ValueError("in_indices must have shape (batch, sequence_length)")
        if in_indices.shape[-1] > self.context_length:
            raise ValueError("sequence_length exceeds context_length")
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
