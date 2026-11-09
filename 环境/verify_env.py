"""环境验证：确认 Python、PyTorch、CUDA 与关键库是否就位。

用法：
    python verify_env.py

任何一项致命问题（比如显卡算力不在 wheel 支持列表里）都会以非零码退出，
这样 setup.ps1 能直接捕获到。
"""
import sys

FAIL = False


def line(k, v):
    print("%-14s %s" % (k + ":", v))


def main():
    global FAIL

    line("Python", "%s  (%s)" % (sys.version.split()[0], sys.executable))

    try:
        import torch
    except Exception as e:
        line("torch", "导入失败: %s" % e)
        return 1

    line("torch", torch.__version__)
    line("cuda(build)", torch.version.cuda)

    available = torch.cuda.is_available()
    line("可用 GPU", available)

    if available:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch = "sm_%d%d" % cap
        archs = torch.cuda.get_arch_list()
        supported = arch in archs

        line("设备", name)
        line("算力", "%s  (%d, %d)" % (arch, cap[0], cap[1]))
        line("本 wheel 支持", supported)

        if not supported:
            print("\n[!] 这个 torch 版本没有编译 %s，训练会报 no kernel image" % arch)
            print("    请改装 CUDA 12.8 及以上的 wheel：")
            print("    pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu128")
            FAIL = True
    else:
        print("\n[i] 当前是 CPU 模式，训练会很慢，仅适合调通流程")

    for mod in ("transformers", "datasets", "tokenizers", "safetensors",
                "numpy", "pandas", "tensorboard"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "?"))
        except Exception:
            line(mod, "未安装")
            if mod in ("transformers", "datasets"):
                FAIL = True

    print()
    if FAIL:
        print("环境验证未通过")
        return 1

    print("环境验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
