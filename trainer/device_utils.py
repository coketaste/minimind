"""GPU 设备检测与抽象 —— 兼容 NVIDIA CUDA 和 AMD ROCm。

设计要点：
  PyTorch-ROCm 在底层把 `torch.cuda.*` API 复用到 HIP/RCCL 上，
  所以 `torch.cuda.is_available()`、`torch.cuda.set_device(N)`、
  字符串 `"cuda:N"`、分布式后端 `"nccl"` 在两套栈上都能工作。
  本模块只负责统一入口，避免代码里散落 `'cuda:0' if torch.cuda.is_available() else 'cpu'`，
  并提供 `is_rocm()` 用于必要时区分。
"""
import torch


def is_rocm() -> bool:
    """当前 PyTorch wheel 是否为 ROCm 构建。"""
    return getattr(torch.version, "hip", None) is not None


def is_cuda() -> bool:
    """当前 PyTorch wheel 是否为原生 NVIDIA CUDA 构建（且有可用设备）。"""
    return torch.cuda.is_available() and not is_rocm()


def get_device_type() -> str:
    """返回 torch.amp 期望的 device_type 字符串。

    ROCm 在 PyTorch 中也走 "cuda" 这条 dispatch key，所以两套栈都返回 "cuda"。
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_default_device(local_rank: int = 0) -> str:
    """返回默认训练/推理设备的字符串表示，例如 "cuda:0" 或 "cpu"。"""
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def get_dist_backend() -> str:
    """返回 torch.distributed 的 backend 名。

    ROCm 的 RCCL 在 PyTorch 里仍然以 "nccl" 字符串调用（PyTorch 内部做 alias）。
    无 GPU 时回退到 "gloo"，便于 CPU 多机调试。
    """
    return "nccl" if torch.cuda.is_available() else "gloo"


def get_vendor_name() -> str:
    """返回 GPU 厂商可读名称，仅用于日志/诊断。"""
    if is_rocm():
        return "AMD ROCm"
    if is_cuda():
        return "NVIDIA CUDA"
    return "CPU"


def print_device_info() -> None:
    """启动时打印一次 GPU 诊断信息，方便在两套栈之间快速排错。"""
    vendor = get_vendor_name()
    if not torch.cuda.is_available():
        print(f"[device] {vendor}: 未检测到 GPU，将使用 CPU")
        return

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)

    if is_rocm():
        ver = f"ROCm/HIP {torch.version.hip}"
    else:
        ver = f"CUDA {torch.version.cuda}"

    bf16_ok = torch.cuda.is_bf16_supported()
    print(
        f"[device] {vendor}: cuda:{idx} ({name}, {total_mem_gb:.1f} GiB), "
        f"{ver}, bf16={bf16_ok}, dist_backend={get_dist_backend()}"
    )
