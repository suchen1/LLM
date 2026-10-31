# Day02 · SFT 全流程

> 对应 `21_LLM` day02：`03_Prefill和Decode` · `05_KVCache实现` · `14-16_ChatTemplate` · `18-20_assistant_answer_mask` · `21-24_SFT 损失与训练流程`
> 代码：`SFT微调/SFT_demo.py`

---

## 一、一个 batch 的完整变换链

```
ultrachat 样本 (messages: system + user/assistant 交替)
  │  apply_chat_template(truncation=True, max_length=1024)
  ▼  list[int]，长度不定
  │  collate_fn：右 padding 到本批最长（pad_token_id = 151643）
  ▼  old  (B, L)
  ├── input_idx = old[:, :-1]      → (B, L-1)
  └── labels    = old[:, 1:]       → (B, L-1)
        │
        ├── answer_mask = creat_answer_mask(input_idx)   # 0/1，答案段为 1
        ├── valid_mask  = (labels != pad_id)             # 0/1，去掉 padding
        └── really      = answer_mask & valid_mask
        ▼
  loss = cross_entropy(logits.view(-1,V), ce_labels.view(-1), ignore_index=-100)
         ce_labels = labels.masked_fill(really == 0, -100)
```

**全篇最关键的一句**：`labels` 右移了一位，所以 `mask[p] = 1` 的真实含义是
「用位置 `p` 的 logits 去预测位置 `p+1` 的 token」。所有看着奇怪的 `+1 / +3 / -1` 都从这句来。

---

## 二、知识点

### 1. 模板的 token 排布（偏移量的唯一来源）

```
... <|im_end|>  "\n"  <|im_start|>  "assistant"  "\n"  答案内容...  <|im_end|>  "\n"
      im_end     +0        +1           +2         +3     +4
```

| 位置 | 是什么 |
|---|---|
| `im_end` | 上一句（user）结束 |
| `im_end+1` | `\n`，`_parse_conversation_turns` 里的 `+1` 就是为了指向它 |
| `im_end+2` | `<|im_start|>` |
| `im_end+3` | `assistant` |
| `im_end+4` | 答案第一个 token（真正的内容起点） |

### 2. 三个偏移量各管什么

| 偏移 | 写在哪 | 作用 |
|---|---|---|
| `+1` | `_parse_conversation_turns` | `im_end` 下标 → `im_end` 后面那个 token 的下标 |
| `+3` | `answer_mask[i, u+3 : a-1]` | 从 `\n`（assistant 后）开始标。**因为 mask 要早一格**，从这格开始监督到的第一格才是答案第一个字 |
| `-1` | 同上，结尾 | 切到 `<|im_end|>` 前面一格；左闭右开，所以最后一格监督的正是 `<|im_end|>` —— **结束符也在训练范围内** |

### 3. 截断分支（不加会静默丢数据）

`max_length=1024` 会把长对话从中间切断，最后一轮 assistant 没有收尾的 `<|im_end|>`，
于是 `user_ends` 比 `assistant_ends` 多一个，`zip` 自动配对时会把最后一轮丢掉。

```python
if   len(user_ends) == len(assistant_ends):          # 正常
    pairs = list(zip(user_ends, assistant_ends))
elif len(user_ends) == len(assistant_ends) + 1:      # 被截断
    pairs = list(zip(user_ends[:-1], assistant_ends))
    answer_mask[i, user_ends[-1] + 3:] = 1           # 尾巴一路标到序列末尾
else:
    pairs = []
```

末尾的 pad 由 `valid_mask` 兜住，所以敢一路标到底。

### 4. loss 的三种写法（数值完全等价）

| 写法 | 代码 | 备注 |
|---|---|---|
| A 手写 | `-log_softmax(logits).gather(-1, labels).squeeze(-1)` 再乘 mask | 课程用它讲原理 |
| B `reduction='none'` | `(cross_entropy(..., reduction='none') * mask).sum() / mask.sum()` | 分母必须自己算对 |
| C `ignore_index` | `labels.masked_fill(mask==0, -100)` + `cross_entropy(..., ignore_index=-100)` | **最推荐**，分母自动 |

实测 A 和 B、B 和 C 都逐位一致（`5.329139232635498`）。

### 5. 学习率调度

手写 `cosine_decay` 和 `torch.optim.lr_scheduler.CosineAnnealingLR` **本质是同一件事**：
都是算一个数，写进 `optimizer.param_groups[0]["lr"]`。

- `optimizer` 改**权重**，`lr_scheduler` 只改**一个数字**，两者独立
- 手写的必须在 `optimizer.step()` **之前**写入 lr；现成的 `scheduler.step()` 在 `optimizer.step()` **之后**调用
- warmup 用 `current_step * max_lr / warmup_steps` 时，**第 0 步 lr = 0**，第一步白走（想严谨就传 `step+1`）

### 6. 混合精度：bf16 和 fp16 不是一回事

`config.json` 里写的是 `torch_dtype = bfloat16`，但**先加载到 CPU 时 transformers 会把它降级成 fp16**，
再 `.to("cuda")` 也变不回来。必须显式指定：

```python
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16)
```

（新版 transformers 用 `dtype=`；`torch_dtype=` 已弃用会警告。）

### 7. 显存账（16 GB 卡的边界）

logits 的体积 = `batch × seq × vocab × 2 字节`，vocab = 151936，这一项就是大头。

| batch | seq | 峰值（含 AdamW） | |
| --- | --- | --- | --- |
| 1 | 1024 | 约 4 GB | ✅ 很宽裕 |
| 2 | 1024 | **10.27 GB** | ✅ 当前配置 |
| 2 | 2500 | 17.7 GB | ❌ OOM |
| 4 | 1500 | 20.9 GB | ❌ OOM |
| 8 | 1500 | OOM | ❌ |

模型权重只占 1.19 GB，**剩下 15 GB 全是中间结果**（logits、梯度、AdamW 的 m/v、28 层激活值）。

---

## 三、踩坑记录

### 1. `_parse_conversation_turns` 漏了 `+1` → 首 token 学不到 + 模型学不会停

- **现象**：训练能跑，loss 也降，但生成时模型不会输出 `<|im_end|>`，只能靠 `max_new_tokens` 硬截断
- **根因**：返回的是 `im_end` 自己的下标，而后续逻辑期望的是"`im_end` 后面那个 token"的下标。差 1 之后，答案范围整体前移一格
- **实测**（同一段对话，两种写法的监督内容）：

  | 写法 | 含 `<|im_end|>` | 监督到的内容 |
  | --- | --- | --- |
  | 不加 `+1` | **False** | `'\n重复执行的行为。\n<think>...</think>21天法。'` |
  | 加 `+1` | **True** | `'重复执行的行为。<|im_end|><think>...</think>21天法。<|im_end|>'` |

- **教训**：偏移量只要有一处差 1，**不报错、loss 照降**，只有把"被监督的 labels 解码出来看"才能发现

### 2. `answer_mask[i, ...]` 写在了 `for i` 循环外面

- **现象**：`TypeError: 'int' object is not subscriptable`，第一个 batch 就崩
- **两个原因叠加**：
  1. 那行缩进比 `for` 少一级，用的是循环结束后残留的 `i`
  2. 内层 `for user_end, assistant_end in zip(...)` 把**列表**重绑成了**整数**，于是 `user_end[-1]` 直接炸
- **修法**：那行移进循环体；内层循环变量改名 `u, a`，别和外层的列表同名

### 3. `old[:, :len(x)]` 少了行下标 → 整批数据变成最后一条

- **现象**：不报错。三行样本最后全一样，还混着前面残留的值
- **实测**：

  ```
  写法 old[:, :len(x)]  -> [[31,32,33,14], [31,32,33,14], [31,32,33,14]]
  写法 old[i, :len(x)]  -> [[11,12,13,14], [21,22,pad,pad], [31,32,33,pad]]
  ```

- **根因**：`:` 表示所有行，每次循环都把每一行重写一遍；一维张量能广播到 `(B, len(x))`，所以**不报错**
- **连带**：切片那两行要用 `:`（`old[:, :-1]`），当时一并写成了 `old[i, :-1]`，返回的是 1 维张量

### 4. `torch.zeros` 初始化 → pad 位置变成 `!`

- **现象**：`(labels != pad_id)` 永远判不出 padding，pad 被当成真实内容参与训练
- **根因**：`token id 0` 解出来是字符 `'!'`，不是 pad。`pad_token_id` 是 151643（`<|endoftext|>`）
- **修法**：用 `torch.full(..., tokenizer.pad_token_id)`

### 5. loss 的分子分母不同源

- **现象**：加了截断修复后 loss 明显偏小，而且比例会随 mask 覆盖率飘
- **根因**：分子用 `really_answer_mask`、分母用 `answer_mask.sum()`。截断修复后 `answer_mask` 会含 pad，分母虚大
- **实测**（一条被截断的样本）：`answer_mask=508`，`& valid` 之后只剩 `25`，差 **483** 个 pad 位置
- **修法**：要么分母改成 `really_answer_mask.sum()`，要么直接用 `ignore_index=-100` 让 PyTorch 自己算

### 6. 加载成 fp16 → 第一步就 nan

- **现象**：loss 从 1.41 → `nan`，之后全是 nan，但**不报任何错**
- **根因**：`config.json` 是 bf16，但先加载到 CPU 被降级成 **fp16**；fp16 + AdamW（`eps=1e-8` 在 fp16 下溢成 0）
- **实测**（同一批数据，3 步）：

  | dtype | step0 | step1 | step2 | 含 nan 的参数 |
  | --- | --- | --- | --- | --- |
  | **fp16** | 1.4092 | nan | nan | **225 → 310** |
  | bf16 | 1.4141 | 0.9336 | 0.5859 | 0 |
  | fp32 | 1.4095 | 0.8071 | 0.3254 | 0 |

- **教训**：`loss = nan` 时**先查 dtype**，别急着怀疑 lr 或数据

### 7. `from time import time` 把模块遮成了函数

- **现象**：跑到打日志那行才崩：`AttributeError: 'builtin_function_or_method' object has no attribute 'strftime'`
- **根因**：`time` 变成了函数 `time()`，不是模块
- **修法**：`import time`，用 `time.strftime(...)`

### 8. `attention_mask` 在"右 padding + 因果注意力"下是多余的

- **原本以为**：不传 `attention_mask` 会让 pad 污染真实 token 的表示
- **实测**：`max|Δlogits| = 0.0000`，argmax 差异 0.0%
- **原因**：因果注意力的第 p 个 token 只能看到 ≤ p，而右 padding 全在末尾（未来），**永远看不到**
- **什么时候才需要**：左 padding（生成时常用）、BERT 那种双向注意力

### 9. `log_dir` 用中文相对路径 → TensorBoard 直接崩

- **现象**：`FailedPreconditionError: ./SFT微调/logs is not a directory`
- **根因**：`SummaryWriter` 走 `tf.io.gfile`，Windows 下解析含中文的路径会出问题（和 `nlp` 仓库踩坑第 3 条同源）
- **实测**：绝对路径（哪怕含中文）✅ / 相对路径 + cwd 在中文目录 ❌
- **修法**：`log_dir` 用**全英文绝对路径**

### 10. `from rich import print` 在窄终端里把日志吃掉

- **现象**：日志变成了 `最近 批次平均损失： . 9`，字符被空格替掉
- **根因**：rich 会按自己算的中文宽度折行 + 给数字上色，和终端实际宽度对不上就互相覆盖
- **修法**：删掉 `from rich import print`（本来也没用到它的特性）；或者把输出重定向到文件确认内容是对的

### 11. 只记了 `train_loss`，没记 `train_lr`

- **现象**：loss 横盘，无法判断是"lr 没生效"还是"数据没得学"
- **修法**：`writer.add_scalar("train_lr", current_learning_rate, step)`

---

## 四、实测数据

| 项 | 数值 |
| --- | --- |
| 100 条样本平均长度 | 927 token（最长 1024） |
| import + 加载模型 | 9.8 s |
| 取 100 条 + tokenize | 0.9 s |
| 单步耗时（batch 2 / seq 1024 / bf16） | 0.68 s |
| 峰值显存（batch 2） | 10.27 GB |
| 通用数据上的 loss 地板 | ≈ 1.2 ~ 1.3 |

---

## 五、结论与下一步

**流程跑通了，但证明不了"微调有用"。**

- train loss 在 1.25~1.47 之间横盘：ultrachat 是通用数据，Qwen3-Base 预训练时已经见过大量同类内容，**loss 本来就贴着地板**
- 1000 步 × batch 2 = 只看过 2000 条样本，且 lr 3e-5 偏小

要做的事：

- [ ] 留出 100 条**不参与训练**的验证集，对比 base 与微调后的 loss 和生成质量
- [ ] 用梯度累积把有效 batch 提到 16，曲线会平滑很多
- [ ] 想看明显下降，**换成模型不擅长的垂直领域数据**（参见 `项目候选.md`）
