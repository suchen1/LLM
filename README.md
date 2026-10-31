# LLM

跟随课程学习大语言模型的练习与笔记仓库，体例与前两个仓库一致：

- [Deep Learn](../Deep%20Learn) —— 神经网络与深度学习
- [nlp_learing](../nlp_learing) —— NLP 与 Transformer

约定同前：**代码按小节逐个练习并实际跑通**，每次学习结束后同步刷新本文件与 [`课程进度.md`](课程进度.md)，
提交信息固定为「更新文档」，`git push` 由我自己执行。

## 学习资料

| 角色 | 资料 | 说明 |
| --- | --- | --- |
| 主线 | 尚硅谷 大模型 2608 线上同步班（30 个模块） | 当前做 `21_LLM`（微调 / 工程 / 推理）。**不追求全学**，取舍与执行顺序见 [`课程进度.md`](课程进度.md) |
| 参考 | [黑马程序员《AI 大模型微调实战》](https://www.bilibili.com/video/BV1ZhWw6AE1V/)（163 P） | 2026-10-14 上架，但内容栈偏 2023：GPT-2 / BERT-PET / ChatGLM-6B / LangChain 旧 Chains API / Flask 部署 |

黑马的用法：当**原理参考书**，不当"现在怎么写"的参照。值得单独回头看的段落：

| 分P | 内容 | 为什么值得看 |
| --- | --- | --- |
| P5–P11 | N-Gram、BLEU / ROUGE / PPL 手写实现 | 评测指标自己实现一遍，比调库理解深 |
| P33–P44 | Prompt-Tuning / P-Tuning / Prefix / Adapter / LoRA 演进 | PEFT 家族的关系图谱 |
| P107–P131 | BERT-PET 的 HardTemplate + Verbalizer | 提示学习里最容易讲不清的一段 |
| P151–P156 | ChatGLM + LoRA 的 config / 数据转换 / 梯度检查点 | 微调工程细节，换个模型照样适用 |

## 环境

| 项 | 说明 |
| --- | --- |
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 5060 Ti（16 GB，sm_120 / Blackwell） |
| Python | 待建独立 conda 环境（不沿用 nlpbase，避免依赖冲突） |
| PyTorch | 需 CUDA 12.8+ 构建，否则 sm_120 无 kernel |
| IDE | PyCharm |

> 16 GB 显存的实际边界：7B 模型 QLoRA 可行；7B 全参微调不行；0.5B–1.5B 的 LoRA 可以放开训。

## 目录结构

```
LLM/
├── 笔记/                 每次学习后的知识点整理与踩坑记录
│   └── day02_SFT.md      day02：chat template → 数据 → mask → loss → 训练循环
├── 作业/                 对照课程自己重写的小实现（day01：RMSNorm / SwiGLU / RoPE）
│   └── day01_任务/
├── SFT微调/              day02 的完整 SFT 代码
│   ├── SFT_demo.py
│   ├── 01_chat_template.ipynb
│   ├── 02_data_pre.ipynb
│   ├── model/            本地模型（已 gitignore）
│   └── data/             ultrachat_200k（已 gitignore）
├── 项目候选.md           选题清单与判据
└── 课程进度.md           30 个模块取舍表 + day01–06 逐日拆解
```

训练产物（`finetuned/`、`runs_sft/`）与模型/数据均不进版本库。

## 学习记录

| 天 | 内容 | 产出 |
| --- | --- | --- |
| day01 | LLM 架构：RoPE / GQA / MoE / Qwen3 代码 | [`作业/day01_任务/`](作业/day01_任务/) —— 自己重写了 RMSNorm、SwiGLU、RoPE，并和官方权重对齐 |
| day02 | SFT 全流程：chat template → padding → answer mask → loss → 训练循环 | [`SFT微调/SFT_demo.py`](SFT微调/SFT_demo.py) · [`笔记/day02_SFT.md`](笔记/day02_SFT.md) |

## 与本仓库相关的前置经验

前两个仓库里已经踩过、这里会再遇到的坑：

- 相对路径相对于**工作目录**，不是脚本所在目录（nlp 踩坑第 6 条）
- CrossEntropyLoss 的类别维、view 与 reshape 的区别（Deep Learn 踩坑第 6 条）
- 训练统计的分子分母必须同一口径（nlp 踩坑第 9、24 条）
- 模型权重与语料不进 git，单文件 100 MB 是硬墙（nlp 踩坑第 2、20 条）
- TensorBoard 在中文路径下会崩，`--logdir` 别拼成 `--logdis`（nlp 踩坑第 3、7 条）

## 参考

- [黑马程序员《AI 大模型微调实战》](https://www.bilibili.com/video/BV1ZhWw6AE1V/)
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/index)