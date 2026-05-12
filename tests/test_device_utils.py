"""Unit tests for trainer.device_utils. No GPU required."""
import contextlib
from contextlib import nullcontext

import pytest
import torch

from trainer.device_utils import (
    DeviceCtx,
    default_device_str,
    dist_backend,
    is_cuda_native,
    is_rocm,
    print_device_info,
    reset_device_info_printed,
    set_current_device,
    vendor_name,
)


@pytest.fixture(autouse=True)
def _reset_print_device_info():
    """每个测试前重置 print_device_info 的幂等标志，避免测试间干扰。"""
    reset_device_info_printed()
    yield
    reset_device_info_printed()


# ---------- module-level free functions ----------

def test_is_rocm_returns_bool():
    assert isinstance(is_rocm(), bool)


def test_is_cuda_native_returns_bool():
    assert isinstance(is_cuda_native(), bool)


def test_is_rocm_and_is_cuda_native_mutually_exclusive():
    """A wheel cannot be both NVIDIA-CUDA and ROCm at the same time."""
    if torch.cuda.is_available():
        assert not (is_rocm() and is_cuda_native())


def test_vendor_name_one_of_known():
    assert vendor_name() in {"AMD ROCm", "NVIDIA CUDA", "CPU"}


def test_dist_backend_known():
    assert dist_backend() in {"nccl", "gloo"}


def test_default_device_str_format():
    s = default_device_str()
    assert s == "cpu" or s.startswith("cuda:")


def test_default_device_str_local_rank_threaded_through():
    if torch.cuda.is_available():
        assert default_device_str(local_rank=3) == "cuda:3"
    else:
        # local_rank is meaningless on CPU; still returns 'cpu'
        assert default_device_str(local_rank=3) == "cpu"


def test_print_device_info_prints_something(capsys):
    print_device_info()
    captured = capsys.readouterr()
    assert "[device]" in captured.out


def test_print_device_info_includes_vendor_and_backend(capsys):
    """The startup banner must surface vendor + dist_backend so users can spot
    misconfigured (e.g. accidentally CPU) environments at a glance."""
    print_device_info()
    out = capsys.readouterr().out
    assert vendor_name() in out
    if torch.cuda.is_available():
        # On GPU we also expect the version string and backend name to be visible.
        assert ("CUDA" in out) or ("ROCm" in out)
        assert dist_backend() in out


def test_print_device_info_is_idempotent(capsys):
    """init_model() runs in distillation/PPO/GRPO multiple times per process;
    print_device_info() must only emit the banner on the first call so the log
    isn't dominated by duplicate banners."""
    print_device_info()
    first = capsys.readouterr().out
    print_device_info()
    second = capsys.readouterr().out
    assert "[device]" in first
    assert second == ""


def test_reset_device_info_printed_re_enables_banner(capsys):
    print_device_info()
    capsys.readouterr()  # drain
    reset_device_info_printed()
    print_device_info()
    out = capsys.readouterr().out
    assert "[device]" in out


def test_set_current_device_is_callable_on_gpu():
    """set_current_device() must actually bind the current CUDA device when a GPU
    is present. CPU-only boxes are skipped — there's nothing to bind."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU to bind")
    set_current_device(0)
    assert torch.cuda.current_device() == 0


# ---------- DeviceCtx.from_arg + properties ----------

def test_from_arg_cpu():
    d = DeviceCtx.from_arg("cpu")
    assert d.device == "cpu"
    assert d.vendor == "cpu"
    assert d.is_gpu is False
    assert d.amp_type == "cpu"


def test_from_arg_cuda_string_preserves_index():
    d = DeviceCtx.from_arg("cuda:3")
    assert d.device == "cuda:3"
    assert d.amp_type == "cuda"
    assert d.is_gpu is True
    # vendor depends on the wheel actually running
    assert d.vendor in {"cuda", "rocm"}


def test_from_arg_bare_cuda_normalizes_to_index_zero():
    """'cuda' (no index) should be normalized to 'cuda:0' to avoid downstream code
    that does .device.split(':')[1] or relies on torch.cuda.current_device()."""
    d = DeviceCtx.from_arg("cuda")
    assert d.device == "cuda:0"
    assert d.amp_type == "cuda"
    assert d.is_gpu is True


def test_from_arg_rejects_unknown_device_string():
    """Unsupported device strings must raise ValueError, not silently fall through."""
    for bad in ("mps", "xpu", "my-cuda-device", "acuda", "cuda0", "CUDA:0", ""):
        with pytest.raises(ValueError):
            DeviceCtx.from_arg(bad)


def test_from_arg_accepts_torch_device():
    """torch.device(...) input should be accepted (commonly returned by torch APIs)."""
    d_cpu = DeviceCtx.from_arg(torch.device("cpu"))
    assert d_cpu.device == "cpu"
    assert d_cpu.vendor == "cpu"

    d_gpu = DeviceCtx.from_arg(torch.device("cuda", 1))
    assert d_gpu.device == "cuda:1"
    assert d_gpu.amp_type == "cuda"


def test_deviceCtx_is_frozen():
    d = DeviceCtx.from_arg("cpu")
    with pytest.raises((AttributeError, Exception)):
        d.device = "cuda:0"  # type: ignore


# ---------- DeviceCtx.for_rank ----------

def test_for_rank_changes_index():
    d = DeviceCtx.from_arg("cuda:0")
    d2 = d.for_rank(2)
    assert d2.device == "cuda:2"
    # original unchanged
    assert d.device == "cuda:0"
    # vendor copied over (no re-detection)
    assert d2.vendor == d.vendor


def test_for_rank_noop_on_cpu():
    d = DeviceCtx.from_arg("cpu")
    d2 = d.for_rank(7)
    assert d2.device == "cpu"
    assert d2.vendor == "cpu"


def test_for_rank_works_on_normalized_bare_cuda():
    """After bare 'cuda' is normalized to 'cuda:0', for_rank() should still re-bind."""
    d = DeviceCtx.from_arg("cuda")
    assert d.device == "cuda:0"
    d2 = d.for_rank(5)
    assert d2.device == "cuda:5"
    assert d2.vendor == d.vendor


# ---------- DeviceCtx.autocast ----------

def test_autocast_on_cpu_returns_nullcontext():
    d = DeviceCtx.from_arg("cpu")
    ctx = d.autocast("bfloat16")
    # nullcontext is the documented CPU return
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_accepts_string_dtypes():
    d = DeviceCtx.from_arg("cpu")
    for s in ("bfloat16", "float16", "float32"):
        ctx = d.autocast(s)
        assert ctx is not None


def test_autocast_accepts_torch_dtype():
    d = DeviceCtx.from_arg("cpu")
    ctx = d.autocast(torch.bfloat16)
    assert ctx is not None


def test_autocast_rejects_unknown_dtype_string():
    d = DeviceCtx.from_arg("cpu")
    with pytest.raises(KeyError) as excinfo:
        d.autocast("typo_dtype")
    assert "typo_dtype" in str(excinfo.value)


def test_autocast_on_gpu_returns_amp_object():
    """When dispatch is cuda, autocast() returns torch.amp.autocast_mode.autocast."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU available on this box")
    d = DeviceCtx.from_arg("cuda:0")
    ctx = d.autocast("bfloat16")
    # Not nullcontext; not strict identity check, just confirm it isn't the CPU stub
    assert not isinstance(ctx, contextlib.nullcontext)


# ---------- DeviceCtx.grad_scaler ----------

def test_grad_scaler_disabled_on_cpu_even_when_enabled_arg_true():
    d = DeviceCtx.from_arg("cpu")
    s = d.grad_scaler(enabled=True)
    assert s.is_enabled() is False


def test_grad_scaler_disabled_when_enabled_false():
    d = DeviceCtx.from_arg("cpu")
    s = d.grad_scaler(enabled=False)
    assert s.is_enabled() is False


def test_grad_scaler_enabled_on_gpu_when_requested():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    d = DeviceCtx.from_arg("cuda:0")
    s = d.grad_scaler(enabled=True)
    assert s.is_enabled() is True


# ---------- DeviceCtx.auto ----------

def test_auto_returns_valid_ctx():
    d = DeviceCtx.auto()
    assert d.device == "cpu" or d.device.startswith("cuda:")


def test_auto_local_rank_propagates():
    if torch.cuda.is_available():
        assert DeviceCtx.auto(local_rank=2).device == "cuda:2"


# ---------- vendor switch via monkeypatch (the only NVIDIA-path coverage) ----------

def test_vendor_detected_as_cuda_when_hip_absent(monkeypatch):
    """Simulate an NVIDIA-CUDA wheel: torch.version.hip is None."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("monkeypatching env doesn't change cuda.is_available")
    d = DeviceCtx.from_arg("cuda:0")
    assert d.vendor == "cuda"


def test_vendor_detected_as_rocm_when_hip_present(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.1.fake", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("can't validate vendor without a GPU device visible")
    d = DeviceCtx.from_arg("cuda:0")
    assert d.vendor == "rocm"
