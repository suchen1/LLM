# Day01 任务：从零实现 Qwen3-0.6B，用官方权重判卷

## 为什么是这个任务

课程给了 `01_Qwen3.py`（484 行）。**抄一遍没用，读懂也不算完**——因为你无法判断自己读懂了没有。

这个任务的验收标准是硬的：**同一份官方权重、同一个输入，你的 logits 和 transformers 的 logits 差 < 1e-4**。差多少就是差多少，没有主观空间。

## 不做的事

| 不做 | 原因 |
|---|---|
| KVCache | 归 day02（`03_Prefill和Decode` / `05_KVCache的实现过程`），现在跳过 |
| mask 的 decode 分支 | 同上。只做 prefill 的普通 causal mask |
| 抄课程注释 | 注释写你自己想通的那句话。课程注释有错（见「已知缺陷」） |
| 抄课程代码结构 | 按你 `nlp` 仓库的体例：一个组件一个文件 + 一个 `use_xxx()` |

## 分六个文件

每个文件独立可跑，`__main__` 里调 `use_xxx()` 打印形状。

| 文件 | 实现什么 | 验收数字 / 现象 |
|---|---|---|
| `dm01_rmsnorm.py` | RMSNorm | 输入 `ones(1,5,1024)` → 输出**逐元素等于 weight**（RMS=1，所以只乘不减） |
| `dm02_swiglu.py` | SwiGLU 前馈层 | 单层参数 **9,437,184** |
| `dm03_rope.py` | `compute_rope_params` + `apply_rope` | 位置 0 处输出**与原值完全相同**；任意位置旋转后 **范数不变**（差 < 1e-5） |
| `dm04_gqa.py` | GQA（`cache=None`） | 单层参数 **6,291,456**；输出 `(1, T, 1024)` |
| `dm05_block.py` | TransformerBlock | 单块参数 **15,732,992** |
| `dm06_model.py` | Qwen3Model（28 层） | 总参数 **596,107,264** |

> 这些数字全部由官方 `config.json` 推算，不是估的：
> `hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072, vocab=151936, layers=28`

## 五个静默出错点（形状对、不报错、结果全错）

写之前先看一遍，写完逐个对照。

**1. 词表是共享的。** 官方 `tie_word_embeddings=True`，所以 `out_head.weight` 必须**就是** `tok_emb.weight`：

```python
self.out_head.weight = self.tok_emb.weight     # 不是新建一个 Linear 就完事
```

建了独立的 `out_head`，总参数会从 5.96 亿变成 **7.51 亿**，而且加载官方权重时 `lm_head` 拿不到值，输出乱码。

**2. RoPE 的配对方式是"半劈"，不是"交错"。** `head_dim=8` 时：

```
本实现（= HF 的 rotate_half）：(0,4) (1,5) (2,6) (3,7)
论文示意图的画法：              (0,1) (2,3) (4,5) (6,7)
```

按论文画法写，形状全对、不报错，**logits 全错**。

**3. GQA 复制是连续复制。** `repeat_interleave(2)` → `[0,0,1,1,...]`（头 `2k` 和 `2k+1` 共用第 `k` 组）；写成 `repeat(2)` → `[0,1,...,7,0,1,...,7]`（头 `k` 和 `k+8` 共用）。**两者形状都是 16 头**。

**4. mask 方向：`True = 挡住`。** 和你 `nlp` 仓库里 `True = 真词` 相反，别照搬 `~mask`。

**5. 所有 Linear 都 `bias=False`。** Qwen3 的 q/k/v/o、gate/up/down 全无 bias。多一个 bias 就跑不出 596M。

## 终极验收

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# 同一份权重、同一 prompt、同一 dtype
logits_hf   = hf_model(input_ids).logits
logits_mine = my_model(input_ids)

print((logits_hf - logits_mine).abs().max())     # 目标 < 1e-4

# 再比贪心生成：前 20 个 token 必须逐个相同
```

对不上时，**从浅到深定位**：先比 `tok_emb` 输出，再比 `final_norm` 之后，最后比每层输出。哪一层开始分叉，问题就在哪。

## 已知缺陷（课程代码里的，别跟着抄）

| 位置 | 问题 |
|---|---|
| 第 33 行 | `out_head` 独立建了 `Linear`，但官方是绑定词表；不共享权重则参数量与官方不符 |
| 第 63–71 行 | mask 每个 token 重建 `pos_end × pos_end`，40k 上下文一次分配约 1.56 GB，是教学写法 |
| 第 49、60–62 行 | `current_pos` 挂在实例上，不 `reset` 会越跑越偏 |
| 第 164 行 | 注释举例 `d_out=1024, head_dim=512` 是 2 头的情况，与真实 16 头（d_out=2048）不符 |

## 交作业的标准

`dm06_model.py` 跑完能打印出：

```
总参数 596,107,264
与官方 logits 最大差 3.2e-05
贪心生成 20 token 完全一致
```

三行对上，day01 就算真过了。
