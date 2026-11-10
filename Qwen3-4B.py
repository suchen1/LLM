import os
os.system('cls' if os.name == 'nt' else 'clear')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from numpy import dtype
from dataclasses import dataclass
import datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer,AutoModelForCausalLM, PreTrainedTokenizer
from typing import List
from rich import print

@dataclass
class Qwen3Config:
    """原版 Qwen/Qwen3-4B 的模型配置，以及 SFT 相关配置。"""

    # 模型来源与精度；构造模型时显式使用该 dtype。
    model_name_or_path: str = "Qwen/Qwen3-4B"
    torch_dtype: torch.dtype = torch.bfloat16

    # Embedding / Decoder / MLP
    vocab_size: int = 151936
    hidden_size: int = 2560           # RMSNorm 和 SwiGLU 的 d_model
    intermediate_size: int = 9728    # SwiGLU 的 hidden_dim
    num_hidden_layers: int = 36
    hidden_act: str = "silu"
    initializer_range: float = 0.02

    # GQA：32 个 Q 头共享 8 个 KV 头，每组对应 4 个 Q 头。
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128              
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # RMSNorm / RoPE；该模型使用全局因果注意力。
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960  # 模型配置上限，不是训练截断长度
    use_sliding_window: bool = False
    sliding_window: int | None = None

    # 输入 Embedding 与输出 LM Head 共享权重。
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    use_cache: bool = True           # 推理启用


def compute_rope_params(head_dim=Qwen3Config.head_dim, theta_base=Qwen3Config.rope_theta, context_length=Qwen3Config.max_position_embeddings):
    """返回 cos, sin，形状都是 (context_length, head_dim)"""
    even_idx = torch.arange(0, head_dim, 2,dtype=torch.float32)  # (head_dim/2,)
    # even_idx = even_idx[:head_dim//2]  # (head_dim/2,)
    theta = 1.0 / theta_base ** (even_idx / head_dim)  # (head_dim/2,)  1000000 ^(-(2i)/d_model)
    seq_idx = torch.arange(context_length)  # (context_length,)
    angle = seq_idx.unsqueeze(1) * theta.unsqueeze(0)  # (context_length, head_dim/2)
    angle = torch.cat([angle,angle],dim = 1)
    return torch.cos(angle), torch.sin(angle)

def apply_rope(x, cos, sin, offset=0):
    """x: (B, H, T, head_dim)，返回同形状"""
    batch_size,head_size,seq_len,head_dim = x.shape
    x1 = x[..., :head_dim//2]  # (B, H, T, head_dim/2)
    x2 = x[..., head_dim//2:]  # (B, H, T, head_dim/2)
    cos = cos[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    rotated = torch.cat([-x2, x1], dim=-1)  # (B, H, T, head_dim)
    x_rotated = x * cos + rotated * sin  # (B, H, T, head_dim)   [a,b] -> [a*cos - b*sin, a*sin + b*cos] = [a,b] * cos + [-b,a] * sin
    return x_rotated


# y = weight * x / sqrt(mean(x², 最后一维) + eps)
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class GroupQueryAttention(nn.Module):
    """GQA 注意力机制，32 个 Q 头共享 8 个 KV 头，每组对应 4 个 Q 头。"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.kv_head_dim = self.head_dim * self.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads # 4   
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False, dtype=config.torch_dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_head_dim, bias=False, dtype=config.torch_dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_head_dim, bias=False, dtype=config.torch_dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.torch_dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.torch_dtype)
        self.o_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=False, dtype=config.torch_dtype)
        if config.attention_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_attention_heads, 1, 1, dtype=config.torch_dtype))
        else:
            self.bias = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.size()
        q = self.q_proj(x)  # (B, T, H*head_dim)
        k = self.k_proj(x)  # (B, T, KV*head_dim)
        v = self.v_proj(x)  # (B, T, KV*head_dim)
        queries = q.view(x.size(0), x.size(1), self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        keys_new = k.view(b, num_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (B, KV, T, head_dim)
        values_new = v.view(b, num_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (B, KV, T, head_dim)
        queries = self.q_norm(queries)
        keys_new = self.k_norm(keys_new)
        position_q = apply_rope(queries, cos, sin, offset=start_pos)  # (B, H, T, head_dim)
        position_k = apply_rope(keys_new, cos, sin, offset=start_pos) # (B, KV, T, head_dim)

        # kv cache存储
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, position_k], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # reset RoPE
            keys, values = position_k, values_new
        next_cache = (keys, values)

        keys = keys.repeat_interleave(self.num_kv_groups, dim=1)  # (B, H, T, head_dim)
        values = values.repeat_interleave(self.num_kv_groups, dim=1)  # (B, H, T, head_dim)
        score = torch.matmul(position_q, keys.transpose(2, 3)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        attention_score = score.masked_fill(mask, -torch.inf)
        # softmax 用 FP32 计算，再转回 V 的精度以进行矩阵乘法。
        attention_probs = F.softmax(attention_score, dim=-1, dtype=torch.float32).to(values.dtype)
        out = torch.matmul(attention_probs, values)  # (B, H, T, head_dim)
        out = out.transpose(1,2).contiguous().view(b, num_tokens, self.all_head_size)  # (B, T, H*head_dim)
        return self.o_proj(out), next_cache

class MLP(nn.Module):
    """SwiGLU MLP 层"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.torch_dtype)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.torch_dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.torch_dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    """Transformer Block，包含 GQA 注意力和 SwiGLU MLP。"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype)
        self.self_attn = GroupQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype)
        self.mlp= MLP(config)
    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos=0, cache=None):
        residual = x
        x = self.input_layernorm(x)
        attention_out, next_cache = self.self_attn(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = residual + attention_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, next_cache

class Qwen3Model(nn.Module):
    """Qwen3-4B 模型主体，包含 Embedding 层和多个 Transformer Block。"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.torch_dtype)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype)
        self.current_pos = 0

        cos, sin = compute_rope_params(
            head_dim=self.config.head_dim,
            theta_base=self.config.rope_theta,
            context_length=self.config.max_position_embeddings
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids, cache=None):
        x = self.embed_tokens(input_ids)  # (B, T, hidden_size)
        b, seq_len,_ = x.size()

        if cache is not None and len(cache) > 0: #decode
            start_pos = self.current_pos
            end_pos = start_pos + seq_len
            self.current_pos = end_pos
            mask = torch.triu(
                torch.ones(end_pos, end_pos, device=x.device, dtype=torch.bool), diagonal=1
            )[start_pos:end_pos, :end_pos]  # (T, T)
        else: #train
            start_pos = 0
            end_pos = seq_len
            mask = torch.triu(
                torch.ones(end_pos, end_pos, device=x.device, dtype=torch.bool), diagonal=1
            )  # (T, T)
            self.current_pos = end_pos

        mask = mask.unsqueeze(0).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            layer_cache = cache.get(i) if cache is not None else None
            x, next_layer_cache = layer(x, mask, self.cos, self.sin, start_pos=start_pos, cache=layer_cache)
            if cache is not None:
                cache[i] = next_layer_cache
        x = self.norm(x)
        return x, cache

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, cache=None):
        hidden_states, cache = self.model(input_ids, cache=cache)
        logits = self.lm_head(hidden_states)
        return logits, cache


# ==================== 生成：加载官方权重 + 对话 ====================
import json as _json
from safetensors import safe_open as _safe_open
from transformers import AutoTokenizer as _AutoTokenizer

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-4B")
_MODEL = None
_TOKENIZER = None


def _load_model(path=_MODEL_DIR, device="cuda"):
    """在 device 上直接建模型并灌权重，不经过 CPU（省 8 GB 内存）。"""
    with open(os.path.join(path, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = _json.load(f)["weight_map"]

    with torch.device(device):
        model = Qwen3ForCausalLM(Qwen3Config())

    state = dict(model.state_dict())
    n = 0
    for shard in sorted(set(weight_map.values())):
        with _safe_open(os.path.join(path, shard), framework="pt", device=device) as f:
            for key in f.keys():
                # lm_head 和 embed_tokens 共享权重，官方 checkpoint 里没有这个键，
                # 所以“找不到”是正常的，不是加载失败。
                if key in state:
                    with torch.no_grad():
                        state[key].copy_(f.get_tensor(key))
                    n += 1
    model.eval()
    print("[加载] %d 个张量 -> %s" % (n, device))
    return model


@torch.no_grad()
def generate(question, max_new_tokens=256, temperature=0.7, top_p=0.8,
             top_k=20, repetition_penalty=1.05, greedy=False, device=None):
    """问一句、答一句。第一次调用加载模型（约 7 s），之后复用。"""
    global _MODEL, _TOKENIZER

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _MODEL is None:
        _MODEL = _load_model(device=device)
        _TOKENIZER = _AutoTokenizer.from_pretrained(_MODEL_DIR)

    messages = [{"role": "user", "content": question}]
    try:
        # Qwen3 默认先吐  thinking 思考块，演示时关掉更直观
        text = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    ids = _TOKENIZER(text, return_tensors="pt").input_ids.to(device)

    # current_pos 是模型里的手动状态，不重置的话 RoPE 会接着上次的位置算
    _MODEL.model.current_pos = 0
    cache = {}
    logits, cache = _MODEL(ids, cache=cache)

    eos = {_TOKENIZER.eos_token_id}
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = _TOKENIZER.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos.add(tid)

    out = []
    for _ in range(max_new_tokens):
        step_logits = logits[:, -1, :].float()

        # 重复惩罚：出现过的 token 降权，避免车轱辘话和多余换行
        if repetition_penalty != 1.0:
            for tid in set(out):
                if step_logits[0, tid] > 0:
                    step_logits[0, tid] /= repetition_penalty
                else:
                    step_logits[0, tid] *= repetition_penalty

        if greedy:
            nxt = step_logits.argmax(-1, keepdim=True)
        else:
            step_logits = step_logits / temperature
            if top_k:
                k = min(top_k, step_logits.size(-1))
                kth = torch.topk(step_logits, k, dim=-1).values[..., -1, None]
                step_logits = step_logits.masked_fill(step_logits < kth, float("-inf"))
            probs = F.softmax(step_logits, dim=-1)
            if 0 < top_p < 1:
                sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_p, dim=-1)
                sorted_p = sorted_p.masked_fill(cum - sorted_p > top_p, 0.0)
                sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                nxt = sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
            else:
                nxt = torch.multinomial(probs, 1)

        tid = nxt.item()
        if tid in eos:
            break
        out.append(tid)
        logits, cache = _MODEL(nxt, cache=cache)

    answer = _TOKENIZER.decode(out, skip_special_tokens=True)
    print("\n用户：%s\n助手：%s\n" % (question, answer))
    return answer


if __name__ == "__main__":
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config())
    print(model)
    print("参数量：", sum(p.numel() for p in model.parameters()))

    generate("用三句话解释什么是 Transformer。", max_new_tokens=128, greedy=True)
