# Day01 任务：从零实现 Qwen3-0.6B，用官方权重判卷

## 0. 目标

不看课程 `01_Qwen3.py`（484 行），自己写出等价实现。**判卷标准**：

```
同一份官方权重、同一个输入 -> 你的 logits 与 transformers 的 logits 差 < 1e-4
```

差多少就是差多少，没有主观空间。

**不做**：KVCache（day02 再补）、mask 的 decode 分支（只做 prefill）。`cache` 一律当 `None`，`start_pos` 一律当 0。

---

## 1. 参数图

### 1.1 参数量树（总 596,049,920）

```
Qwen3Model                                        596,049,920
│
├─ tok_emb        151,936 × 1,024 ─────────────  155,582,464
│
├─ trf_blocks     × 28 ────────────────────────  440,466,432
│   └─ TransformerBlock                          15,730,944
│       │
│       ├─ norm1   RMSNorm(1024) ──────────────       1,024
│       │
│       ├─ att     GroupedQueryAttention ──────   6,291,712
│       │    ├─ q_norm    RMSNorm(128) ────────         128
│       │    ├─ k_norm    RMSNorm(128) ────────         128
│       │    ├─ W_query   1,024 → 2,048 ───────   2,097,152
│       │    ├─ W_key     1,024 → 1,024 ───────   1,048,576
│       │    ├─ W_value   1,024 → 1,024 ───────   1,048,576
│       │    └─ out_proj  2,048 → 1,024 ───────   2,097,152
│       │
│       ├─ norm2   RMSNorm(1024) ──────────────       1,024
│       │
│       └─ ff      SwiGLU ─────────────────────   9,437,184
│            ├─ gate_proj  1,024 → 3,072 ──────   3,145,728
│            ├─ up_proj    1,024 → 3,072 ──────   3,145,728
│            └─ down_proj  3,072 → 1,024 ──────   3,145,728
│
├─ final_norm    RMSNorm(1024) ─────────────────       1,024
│
└─ out_head      与 tok_emb 共享权重 ──────────           0
```

三个关键比值，记住比记住数字有用：

| 关系 | 值 | 含义 |
|---|---|---|
| `W_query` 输出宽 / `d_in` | 2048 / 1024 = **2 倍** | head_dim=128 与 hidden 解耦 |
| `W_key` 输出宽 / `W_query` 输出宽 | 1024 / 2048 = **1/2** | GQA 的 8 组 vs 16 头 |
| `ff` / `att` 参数 | 9.44M / 6.29M = **1.5 倍** | 前馈层永远是参数大头 |

### 1.2 张量形状流（B=1, T=5）

```
in_idx (1, 5)
  └─ tok_emb ─────────────────────────────► x (1, 5, 1024)
       │
       └─ × 28 TransformerBlock:
            │
            ├─ norm1 ──────────────────────► (1, 5, 1024)
            │    ├─ W_query ───────────────► (1, 5, 2048)
            │    │   view+transpose ───────► (1, 16, 5, 128)   q
            │    ├─ W_key ─────────────────► (1, 5, 1024)
            │    │   view+transpose ───────► (1,  8, 5, 128)   k
            │    └─ W_value ───────────────► (1, 5, 1024)
            │        view+transpose ───────► (1,  8, 5, 128)   v
            │
            │    q_norm/k_norm (各自 128 维上) + RoPE
            │    k, v: repeat_interleave(2) ► (1, 16, 5, 128)
            │
            │    q @ kᵀ ───────────────────► (1, 16, 5, 5)
            │    /√128 → masked_fill → softmax
            │    @ v ──────────────────────► (1, 16, 5, 128)
            │    transpose(1,2).reshape ───► (1, 5, 2048)
            │    out_proj ─────────────────► (1, 5, 1024)
            │
            ├─ + x (残差) ─────────────────► (1, 5, 1024)
            │
            ├─ norm2 ──────────────────────► (1, 5, 1024)
            │    gate → silu ──────────────► (1, 5, 3072)
            │    up ───────────────────────► (1, 5, 3072)
            │    两者相乘 → down ──────────► (1, 5, 1024)
            │
            └─ + x (残差) ─────────────────► (1, 5, 1024)
       │
       ├─ final_norm ─────────────────────► (1, 5, 1024)
       └─ out_head(共享) ─────────────────► logits (1, 5, 151936)
```

---

## 2. 六个文件的任务卡

体例同你前三个仓库：一个组件一个文件，`__main__` 里调用 `use_xxx()`。

---

### `dm01_rmsnorm.py` — 归一化

**要实现**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6)   # 只一个 weight，初始化为全 1
    def forward(self, x)                # x: (..., dim)
```

公式：`y = weight * x / sqrt(mean(x², 最后一维) + eps)`

| 要求 | 说明 |
|---|---|
| **不减均值** | 这是和 LayerNorm 唯一的区别，也是最容易写错的地方 |
| **eps 在 sqrt 里面** | `sqrt(mean(x²) + eps)`，不是 `sqrt(mean(x²)) + eps` |
| **用 float32 算** | 输入可能是 bf16，先 `x.float()`，算完 `.to(x.dtype)`，否则误差到 1e-2 |

**输入 / 输出**：`(..., 1024)` → `(..., 1024)`

**验收（四条全过）**

1. `sum(p.numel() for p in m.parameters())` = **1,024**
2. `x = torch.ones(1,5,1024)` → 输出**逐元素等于 weight**（因为 RMS=1）
3. 随机 `x`：先把 weight 除掉，每行的 RMS 应 ≈ 1（`(x**2).mean(-1).sqrt()`）
4. 和 `F.rms_norm(x, (1024,), weight, 1e-6)` 比，`max|diff| < 1e-6`

**常见错**：写成 LayerNorm（减均值）；eps 加到 sqrt 外面；bf16 直接算。

---

### `dm02_swiglu.py` — 前馈层

**要实现**

```python
class SwiGLU(nn.Module):
    def __init__(self, dim=1024, hidden=3072)
    # gate_proj: Linear(1024, 3072, bias=False)
    # up_proj:   Linear(1024, 3072, bias=False)
    # down_proj: Linear(3072, 1024, bias=False)
    def forward(self, x):   # down_proj( silu(gate_proj(x)) * up_proj(x) )
```

`silu(x) = x * sigmoid(x)`

**输入 / 输出**：`(1,5,1024)` → 中间 `(1,5,3072)` → `(1,5,1024)`

**验收**

1. 参数量 = **9,437,184**
2. 三个 Linear 都 `bias=False`，逐个打印 `weight.shape` 应为 `(3072,1024)`、`(3072,1024)`、`(1024,3072)`
3. 输出形状 `(1,5,1024)`

**常见错**：只走 gate 一个分支（那就变成普通 MLP 了）；把 `gate`/`up` 顺序写反（数值上等价，但权重名和 HF 对不上，后面加载会错位）。

---

### `dm03_rope.py` — 旋转位置编码

两个**纯函数**，不含 `nn.Module`，最容易单独测。

**要实现**

```python
def compute_rope_params(head_dim=128, theta_base=1_000_000, context_length=40_960):
    """返回 cos, sin，形状都是 (context_length, head_dim)"""

def apply_rope(x, cos, sin, offset=0):
    """x: (B, H, T, head_dim)，返回同形状"""
```

**核心是配对方式：半劈，不是交错。**

```
head_dim = 8 时：
  本实现（= HF 的 rotate_half）：(0,4) (1,5) (2,6) (3,7)
  论文示意图的画法：              (0,1) (2,3) (4,5) (6,7)
```

实现要点：`angles` 算出来是 `(T, 64)`，然后 `cat([angles, angles], dim=1)` 变成 `(T, 128)`；旋转时把 `x` 前后各劈一半：

```python
x1, x2 = x[..., :64], x[..., 64:]
rotated = torch.cat([-x2, x1], dim=-1)
out = x * cos + rotated * sin
```

**验收（五条）**

1. `cos.shape == sin.shape == (40960, 128)`
2. `cos[0]` 全 1、`sin[0]` 全 0（位置 0 角度为 0）
3. `T=1, offset=0` 时，`apply_rope` 输出**等于输入**（差 0）
4. 任意位置：旋转后**每个 head 的范数不变**（`max|diff| < 1e-5`）
5. **手算一例**：取 `x[0,0,1,0]=1`，其余为 0，位置 1，验证 `out[0] == cos(θ₀)`、`out[64] == sin(θ₀)`，其中 `θ₀ = 1 / 1000000^(0/128)`

**常见错**：用交错配对（形状全对、logits 全错，这是本任务最大的坑）；`cat` 顺序反了；忘了 `offset` 切片。

---

### `dm04_gqa.py` — 分组查询注意力

**要实现**

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in=1024, num_heads=16, num_kv_groups=8,
                 head_dim=128, qk_norm=True)
    def forward(self, x, mask, cos, sin):   # (1,5,1024) -> (1,5,1024)
```

**必须按这个顺序写 forward**

| 步 | 动作 | 形状 |
|---|---|---|
| 1 | `W_query(x)` | `(1,5,2048)` |
| 2 | `W_key(x)` / `W_value(x)` | `(1,5,1024)` |
| 3 | Q 拆头：`view(B,T,16,128).transpose(1,2)` | `(1,16,5,128)` |
| 4 | K/V 拆组：`view(B,T,8,128).transpose(1,2)` | `(1,8,5,128)` |
| 5 | `q_norm` / `k_norm`（作用在 **128 维**上） | 不变 |
| 6 | `apply_rope` 作用在 q 和 k 上 | 不变 |
| 7 | **`repeat_interleave(2, dim=1)`** 把 k、v 扩成 16 | `(1,16,5,128)` |
| 8 | `q @ kᵀ / sqrt(128)` | `(1,16,5,5)` |
| 9 | `masked_fill(mask, -inf)` → `softmax(dim=-1)` | `(1,16,5,5)` |
| 10 | `@ v` → `transpose(1,2).reshape(B,T,2048)` | `(1,5,2048)` |
| 11 | `out_proj` | `(1,5,1024)` |

**验收**

1. 参数量 = **6,291,712**（含 `q_norm`/`k_norm` 各 128）
2. 第 3、4、7、8 步的中间形状全部打印出来，逐个对照上表
3. **`T=1` 专项**：此时 `scores` 是 `(1,16,1,1)`，softmax 恒等于 1，输出应等于 `out_proj(v 的第 0 位)` —— 这是最简单的端到端验证
4. `mask` 全 `False` 时，输出不应含 NaN

**常见错**

- `repeat_interleave` 写成 `repeat`（形状都是 16 头，**不报错，结果全错**）
- `q_norm` 作用在 2048 维上（错，必须每个 head 的 128 维各自归一化）
- 缩放 `1/√128` 漏了，或加在 softmax 之后

---

### `dm05_block.py` — 一个 Transformer 层

**要实现**

```python
class TransformerBlock(nn.Module):
    # norm1, att(GroupedQueryAttention), norm2, ff(SwiGLU)
    def forward(self, x, mask, cos, sin):
        x = x + self.att(self.norm1(x), mask, cos, sin)   # Pre-Norm
        x = x + self.ff(self.norm2(x))
        return x
```

**顺序是 Pre-Norm**：先归一化，再过子层，最后加残差。残差加的是**进入 norm 之前**的那个 `x`。

**验收**

1. 参数量 = **15,730,944**（= 1,024 + 6,291,712 + 1,024 + 9,437,184）
2. 输入 `(1,5,1024)` → 输出 `(1,5,1024)`
3. 打印残差那一步的 `shortcut.shape`，必须也是 `(1,5,1024)`

**常见错**：写成 Post-Norm（norm 放在残差之后）；`shortcut` 存成了 `norm1(x)` 而不是 `x`。

---

### `dm06_model.py` — 整个模型

**要实现**

```python
class Qwen3Model(nn.Module):
    # tok_emb:    Embedding(151936, 1024)
    # trf_blocks: ModuleList([TransformerBlock(cfg) for _ in range(28)])
    # final_norm: RMSNorm(1024)
    # out_head:   与 tok_emb 共享权重
    # 注册 cos / sin 为 buffer (persistent=False)
    def forward(self, in_idx):   # (1,T) -> logits (1,T,151936)
```

**mask 构造**（`True = 挡住`，和你 `nlp` 仓库相反）：

```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # 上方为 True = 挡
mask = mask.unsqueeze(0).unsqueeze(0)                                # (1,1,T,T)
```

**`out_head` 的处理**（这里有个坑，见 §4）——建议共享：

```python
self.out_head = nn.Linear(1024, 151936, bias=False)
self.out_head.weight = self.tok_emb.weight      # 指向同一块内存
```

**验收**

1. 参数量：共享 = **596,049,920**；不共享 = **751,632,384**
2. `logits.shape == (1, 5, 151936)`
3. 与 HF 对比 `max|diff| < 1e-4`
4. 贪心生成 20 个 token，逐个相同

**常见错**：mask 方向反了（变成只能看未来）；`cos`/`sin` 没注册成 buffer，权重搬到 cuda 后它们还在 cpu。

---

## 3. 验收数字一览

| 组件 | 参数量 | 累计 |
|---|---|---|
| `RMSNorm(1024)` | 1,024 | — |
| `SwiGLU` | 9,437,184 | — |
| `GroupedQueryAttention` | 6,291,712 | — |
| `TransformerBlock` | 15,730,944 | — |
| 28 层 | 440,466,432 | 440,466,432 |
| `tok_emb` | 155,582,464 | 596,048,896 |
| `final_norm` | 1,024 | **596,049,920** |

---

## 4. 关于 `out_head`：一个容易判错的细节

官方 `config.json` 写 `tie_word_embeddings = True`，但**官方 checkpoint 文件里 `lm_head.weight` 是单独存了一份的**。

怎么确认的：`model.safetensors` 是 1,503,300,328 字节，减掉 35,560 字节 header 后是 1,503,264,768 字节，即 **751,632,384** 个 bf16 参数 = `596,049,920 + 155,582,464`。

所以：

| 你的写法 | 参数量 | 能跑通吗 | 结果对吗 |
|---|---|---|---|
| 共享 `out_head.weight = tok_emb.weight` | 596,049,920 | ✓ | ✓ 符合官方语义 |
| 独立 `out_head`，并加载 `lm_head.weight` | 751,632,384 | ✓ | ✓ 因为两份权重数值相同 |
| 独立 `out_head`，但不加载 | 751,632,384 | ✓ | ✗ 随机初始化，输出乱码 |

**结论**：独立建 `out_head` 本身不是 bug，只有当你不去加载它才是 bug。建议按共享写（省 596 MB 权重），但要能解释"为什么官方文件里有那一份"——**保存格式和模型语义是两件事**。

---

## 5. 终极验收

```python
logits_hf   = hf_model(input_ids).logits
logits_mine = my_model(input_ids)
print((logits_hf - logits_mine).abs().max())     # 目标 < 1e-4

# 再比贪心生成：前 20 个 token 必须逐个相同
```

对不上时的定位顺序（不要乱猜）：

```
tok_emb 输出  →  final_norm 之后  →  第 0 层输出  →  第 13 层  →  第 27 层
哪一层开始分叉，问题就在那一层
```

## 6. 已知缺陷（课程代码里的，别跟着抄）

| 位置 | 问题 |
|---|---|
| 第 63–71 行 | mask 每个 token 重建 `pos_end × pos_end`，40k 上下文一次分配约 1.56 GB，是教学写法 |
| 第 49、60–62 行 | `current_pos` 挂在实例上，不 `reset` 会越跑越偏 |
| 第 164 行 | 注释举例 `d_out=1024, head_dim=512`（2 头），与真实 16 头（`d_out=2048`）不符 |
| 第 31 行注释 | `model.parameter()` 应为 `parameters()` |
