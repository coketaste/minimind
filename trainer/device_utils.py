"""GPU 设备检测与抽象 —— 兼容 NVIDIA CUDA 和 AMD ROCm，CPU fallback。

设计要点：
  - 模块级自由函数提供 *环境* 事实（wheel 类型、默认设备字符串、分布式后端等）。
  - DeviceCtx dataclass 提供 *单进程* 状态（autocast、GradScaler、DDP 内 rank 重绑）。
  - 底层尽量走 torch.accelerator (PyTorch 2.6+) 这套统一 API；
    只有在 torch.accelerator 暂未覆盖的地方（如 bf16 capability 查询、显存统计）才退回 torch.cuda.*。
  - PyTorch-ROCm 把 torch.cuda.* 别名到 HIP/RCCL，所以 'cuda:N' 字符串、
    autocast device_type='cuda'、dist backend 'nccl' 在两套栈上都能工作。
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Literal, Union

import torch

VendorLit = Literal["cuda", "rocm", "cpu"]

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# =========================================================================
# Layer 1 — environment facts (no per-trainer state)
# =========================================================================


def is_rocm() -> bool:
    """当前 PyTorch wheel 是否为 ROCm 构建（看 torch.version.hip）。"""
    return getattr(torch.version, "hip", None) is not None


def is_cuda_native() -> bool:
    """当前 PyTorch wheel 是否为原生 NVIDIA CUDA 构建（且至少有一块可用 GPU）。"""
    return torch.cuda.is_available() and not is_rocm()


def vendor_name() -> str:
    """日志友好的厂商名："AMD ROCm" / "NVIDIA CUDA" / "CPU"。"""
    if is_rocm():
        return "AMD ROCm"
    if is_cuda_native():
        return "NVIDIA CUDA"
    return "CPU"


def dist_backend() -> str:
    """torch.distributed 的 backend 名。

    ROCm 的 RCCL 在 PyTorch 里仍以 "nccl" 字符串调用（PyTorch 内部 alias）。
    无 GPU 时回退到 "gloo"，便于 CPU 多机调试。
    """
    return "nccl" if torch.cuda.is_available() else "gloo"


def default_device_str(local_rank: int = 0) -> str:
    """argparse `--device` 的默认值。"""
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def set_current_device(local_rank: int) -> None:
    """绑定当前进程到指定 local_rank 的设备。

    优先使用 torch.accelerator.set_device_index (PyTorch 2.6+)，
    回退到 torch.cuda.set_device 以兼容老版本。
    """
    accel = getattr(torch, "accelerator", None)
    if accel is not None and hasattr(accel, "set_device_index"):
        accel.set_device_index(local_rank)
    else:
        torch.cuda.set_device(local_rank)


_DEVICE_INFO_PRINTED = False


def print_device_info() -> None:
    """启动时打印一次 GPU 诊断信息，方便在两套栈之间快速排错。

    模块级幂等：多次调用只打印一次（distillation/PPO/GRPO 这种 init_model 被
    调用多次的训练阶段，避免 banner 重复刷屏掩盖第一行）。测试可通过
    reset_device_info_printed() 重置。
    """
    global _DEVICE_INFO_PRINTED
    if _DEVICE_INFO_PRINTED:
        return
    _DEVICE_INFO_PRINTED = True

    vendor = vendor_name()
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
        f"{ver}, bf16={bf16_ok}, dist_backend={dist_backend()}"
    )


def reset_device_info_printed() -> None:
    """测试钩子：重置 print_device_info 的幂等标志。"""
    global _DEVICE_INFO_PRINTED
    _DEVICE_INFO_PRINTED = False


def seed_all(seed: int) -> None:
    """对当前 accelerator 的全部设备播种。

    包装 torch.cuda.manual_seed[_all]——ROCm 别名同名 API，所以两套栈通用。
    无 GPU 时是 no-op，避免在 CPU-only 环境下触发 lazy init。调用方仍需自行
    处理 random/numpy/torch.manual_seed。
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def empty_cache() -> None:
    """释放当前 accelerator 的缓存显存。无 GPU 时是 no-op。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================================
# Layer 2 — per-trainer state (DeviceCtx)
# =========================================================================


@dataclass(frozen=True)
class DeviceCtx:
    """每个训练进程持有一个 DeviceCtx，描述本进程要跑在哪台设备上。

    不可变。`for_rank()` 返回新实例。`vendor` 在构造时一次性确定，
    `for_rank()` 不会重新探测厂商。
    """

    device: str          # 'cuda:N' 或 'cpu'
    vendor: VendorLit    # 'cuda' (NVIDIA) | 'rocm' (AMD) | 'cpu'

    # ----- factories -----

    @classmethod
    def from_arg(cls, device: Union[str, torch.device]) -> "DeviceCtx":
        """根据用户显式传入的 device 字符串构造（如 argparse --device 的结果）。

        - 'cpu' / torch.device('cpu')             → CPU ctx
        - 'cuda'                                   → 归一化为 'cuda:0'（避免 bare-cuda 在 DDP 外露依赖于 current_device()）
        - 'cuda:N' / torch.device('cuda', N)       → GPU ctx
        - 其他字符串                               → ValueError（避免静默把 'mps'、'xpu'、笔误等当 CPU 处理）
        """
        s = str(device)
        if s == "cpu":
            return cls(device="cpu", vendor="cpu")
        if s == "cuda" or s.startswith("cuda:"):
            v: VendorLit = "rocm" if is_rocm() else "cuda"
            return cls(device="cuda:0" if s == "cuda" else s, vendor=v)
        raise ValueError(
            f"Unsupported device string {s!r}; expected 'cpu', 'cuda' or 'cuda:N'."
        )

    @classmethod
    def auto(cls, local_rank: int = 0) -> "DeviceCtx":
        """根据当前环境自动选择设备。"""
        return cls.from_arg(default_device_str(local_rank))

    # ----- derived properties -----

    @property
    def amp_type(self) -> str:
        """torch.amp.autocast(device_type=...) 期望的字符串。"""
        return "cuda" if self.device.startswith("cuda") else "cpu"

    @property
    def is_gpu(self) -> bool:
        """True 当且仅当本 ctx 会派发到 GPU（即 amp_type == 'cuda'）。"""
        return self.amp_type == "cuda"

    # ----- behaviour -----

    def for_rank(self, local_rank: int) -> "DeviceCtx":
        """返回一个绑定到 local_rank 的新 DeviceCtx。CPU 路径下是 no-op。"""
        if not self.is_gpu:
            return self
        return DeviceCtx(device=f"cuda:{local_rank}", vendor=self.vendor)

    def autocast(self, dtype: Union[str, torch.dtype]) -> ContextManager:
        """混合精度上下文管理器。CPU 路径下返回 nullcontext。

        dtype 接受 'bfloat16' / 'float16' / 'float32' 字符串或 torch.dtype。
        未知字符串抛 KeyError（不论 CPU 还是 GPU，立即报错而非静默忽略）。
        """
        if isinstance(dtype, str):
            dtype = _DTYPE_ALIASES[dtype]
        if not self.is_gpu:
            return nullcontext()
        return torch.amp.autocast(device_type=self.amp_type, dtype=dtype)

    def grad_scaler(self, *, enabled: bool) -> torch.amp.GradScaler:
        """构造 GradScaler。CPU 路径下永远 disabled，避免 fp16+CPU 抛错。"""
        return torch.amp.GradScaler(self.amp_type, enabled=enabled and self.is_gpu)
