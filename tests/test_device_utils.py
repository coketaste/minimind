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
    set_current_device,
    vendor_name,
)


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


def test_from_arg_bare_cuda():
    d = DeviceCtx.from_arg("cuda")
    assert d.device == "cuda"
    assert d.amp_type == "cuda"


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
