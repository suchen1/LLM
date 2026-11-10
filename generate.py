# -*- coding: utf-8 -*-
"""
Qwen3-4B 生成脚本（使用你自己手写的 Qwen3-4B.py 里的模型）

做的事情：
    1. 从 Qwen3-4B.py 里拿到 Qwen3Config / Qwen3ForCausalLM
    2. 把 model.safetensors 里的权重灌进去（直接在 GPU 上构建，不占 CPU 内存）
    3. 套 Qwen3 的 chat template，跑一轮问答，逐字流式输出

用法：
    python generate.py
    python generate.py --prompt "用三句话解释什么是注意力机制"
    python generate.py --prompt "..." --max-new-tokens 512 --temperature 0.6
    python generate.py --cpu          # 没有显卡时（很慢，仅验证）
"""

import argparse
import builtins
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "Qwen3-4B"
IMPL_FILE = HERE / "Qwen3-4B.py"


# ----------------------------------------------------------------------
# 1. 载入手写模型
# ----------------------------------------------------------------------
def import_impl():
    """Qwen3-4B.py 的文件名带连字符，不能直接 import，用 importlib 按路径加载。"""
    if not IMPL_FILE.exists():
        sys.exit("找不到 %s" % IMPL_FILE)

    # 那个文件开头有 from rich import print，但 rich 不一定装了。
    # 没装就用内置 print 顶替，不影响生成结果。
    try:
        import rich  # noqa: F401
    except ImportError:
        shim = types.ModuleType("rich")
        shim.print = builtins.print
        sys.modules["rich"] = shim
        print("[提示] 没装 rich，已用内置 print 顶替（不影响结果）\n")

    spec = importlib.util.spec_from_file_location("qwen3_impl", IMPL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# 2. 选设备
# ----------------------------------------------------------------------
def pick_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        print("[设备] CPU（会很慢，仅用于验证）")
        return "cpu"

    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    print("[设备] %s   可用显存 %.1f GB / 共 %.1f GB" % (name, free / 2**30, total / 2**30))

    # 4B 用 bf16 是 8 GB 权重，加上 KV cache 和临时张量，留 1.5 GB 余量
    need = 9.5 * 2**30
    if free < need:
        print("\n[警告] 可用显存不足 9.5 GB，4B 模型大概率 OOM。")
        print("       6 GB 显卡跑不了 4B 全精度，请改用 4-bit 量化（QLoRA），")
        print("       或换一台 16 GB 的机器。加 --cpu 可以先验证代码逻辑。")
    return "cuda"


# ----------------------------------------------------------------------
# 3. 灌权重
# ----------------------------------------------------------------------
def load_weights(model, device):
    index_file = MODEL_DIR / "model.safetensors.index.json"
    if not index_file.exists():
        sys.exit("找不到 %s" % index_file)

    with open(index_file, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    state = dict(model.state_dict())
    loaded = 0
    skipped = []
    t0 = time.time()

    for shard in sorted(set(weight_map.values())):
        with safe_open(str(MODEL_DIR / shard), framework="pt", device=device) as f:
            for key in f.keys():
                if key in state:
                    with torch.no_grad():
                        # copy_ 而不是整体 load_state_dict：权重直接在显存里，
                        # CPU 内存不参与，8 GB 模型也不需要 8 GB 空闲内存。
                        state[key].copy_(f.get_tensor(key))
                    loaded += 1
                else:
                    skipped.append(key)

    # 官方 checkpoint 里没有 lm_head.weight（和 embed_tokens 共享），
    # 所以这里“跳过”是正常的，不是错误。
    if skipped:
        print("[权重] 跳过 %d 个（tied 权重，正常）：%s" % (len(skipped), ", ".join(skipped)))

    print("[权重] 已加载 %d 个张量，耗时 %.1f s" % (loaded, time.time() - t0))


# ----------------------------------------------------------------------
# 4. 生成
# ----------------------------------------------------------------------
def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        # Qwen3 默认会先输出  thinking...<｜end▁of▁thinking｜> 思考块，演示时关掉更直观
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return text


@torch.no_grad()
def generate(model, tokenizer, device, question, max_new_tokens=256,
             temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
             stream=True):
    prompt = build_prompt(tokenizer, question)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print("\n" + "=" * 70)
    print("用户：%s" % question)
    print("-" * 70)
    print("助手：", end="", flush=True)

    # 每次生成都必须重置位置，否则 RoPE 会接着上次的位置算（这是那个类的手动状态）
    model.model.current_pos = 0
    cache = {}

    # prefill
    t0 = time.time()
    logits, cache = model(ids, cache=cache)
    prefill_ms = (time.time() - t0) * 1000

    eos_ids = {tokenizer.eos_token_id}
    for extra in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(extra)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)

    seen = list(ids[0].tolist())
    generated = []
    printed = ""
    stop = False

    for step in range(max_new_tokens):
        logits = logits[:, -1, :].float()

        # 重复惩罚：对已出现过的 token 降权
        if repetition_penalty != 1.0:
            for tid in set(seen):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            if top_p and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(sorted_probs, 1)
                nxt = sorted_idx.gather(-1, choice)
            else:
                nxt = torch.multinomial(probs, 1)

        tid = nxt.item()
        if tid in eos_ids:
            stop = True
            break

        seen.append(tid)
        generated.append(tid)

        if stream:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if len(text) > len(printed):
                print(text[len(printed):], end="", flush=True)
                printed = text

        logits, cache = model(nxt, cache=cache)

    if not stream or not printed:
        print(tokenizer.decode(generated, skip_special_tokens=True), end="")
    print()

    total_s = time.time() - t0
    n = len(generated)
    print("-" * 70)
    print("生成 %d 个 token | prefill %.0f ms | 总计 %.2f s | %.1f tok/s%s"
          % (n, prefill_ms, total_s, n / total_s if total_s else 0,
             "（遇到结束符）" if stop else "（达到上限）"))
    print("=" * 70)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="用三句话解释什么是 Transformer。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true", help="贪心解码，输出可复现")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        sys.exit("找不到模型目录 %s" % MODEL_DIR)

    impl = import_impl()
    device = pick_device(force_cpu=args.cpu)

    print("[构建] 在 %s 上初始化模型…" % device, end=" ", flush=True)
    t0 = time.time()
    with torch.device(device):
        model = impl.Qwen3ForCausalLM(impl.Qwen3Config())
    print("%.1f s" % (time.time() - t0))

    load_weights(model, device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    generate(
        model, tokenizer, device, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0 if args.greedy else args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
