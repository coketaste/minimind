# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MiniMind — an end-to-end "from scratch" pipeline for training a small (~64M dense / ~198M-A64M MoE) Qwen3-compatible LLM in pure PyTorch. Covers tokenizer training → pretraining → SFT → LoRA → DPO → distillation → RLAIF (PPO/GRPO) → Agent RL, plus serving (OpenAI-compatible API, Streamlit demo) and HF-format conversion.

Core algorithms are implemented directly in PyTorch — avoid replacing them with `trl`/`peft`/`transformers.Trainer` wrappers.

## Layout

- `model/model_minimind.py` — `MiniMindConfig`, `MiniMindForCausalLM`. Llama-family block (RMSNorm + RoPE + GQA + SwiGLU), with an optional MoE FFN gated by `config.use_moe`. Returns `MoeCausalLMOutputWithPast` so `loss + aux_loss` works for both dense and MoE (aux_loss is 0 for dense).
- `model/model_lora.py` — `apply_lora` / `load_lora` / `save_lora` / `merge_lora`. LoRA adapters live as a separate `.pth` keyed by layer name (not via `peft`).
- `model/tokenizer.json`, `model/tokenizer_config.json` — vocab 6400, BPE. Includes special tokens for `<tool_call>`, `<tool_response>`, `<think>` and a Jinja chat template that supports `open_thinking` / tool use.
- `dataset/lm_dataset.py` — one `Dataset` class per stage: `PretrainDataset`, `SFTDataset`, `DPODataset`, `RLAIFDataset`, `AgentRLDataset`. SFT/DPO use `tokenizer.apply_chat_template` and mask everything except assistant turns (everything else → label `-100`).
- `trainer/` — one script per training stage (see commands). Shared helpers in `trainer/trainer_utils.py`: `init_model`, `lm_checkpoint`, `init_distributed_mode`, `get_lr` (cosine, floored at 0.1), `SkipBatchSampler`, `LMForRewardModel`.
- `trainer/rollout_engine.py` — sampling/rollout used by PPO/GRPO/agent trainers.
- `eval_llm.py` — CLI inference (the only entry point at the repo root).
- `scripts/serve_openai_api.py` — FastAPI OpenAI-compatible server with streaming, `reasoning_content`, and `tool_calls`.
- `scripts/web_demo.py` — Streamlit chat UI.
- `scripts/convert_model.py` — converts native `.pth` → HF `transformers` format. Two flavors: `convert_torch2transformers_minimind` (custom `MiniMindForCausalLM`, registers auto-class) and `convert_torch2transformers` (remaps weights to `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM` for ecosystem compat — `llama.cpp`, `vllm`, `ollama`, etc.).
- `dataset/` — runtime location for downloaded `.jsonl` training files (gitignored). `out/` and `checkpoints/` are also gitignored.

## Working directory convention

**All `trainer/*.py` scripts and `scripts/*.py` use `../` paths** (`../out`, `../dataset`, `../checkpoints`, `../model`). Run them from inside their own directory:

```bash
cd trainer && python train_pretrain.py ...
cd scripts && python serve_openai_api.py ...
```

`eval_llm.py` is the exception — run it from the repo root (uses `./out`, `./model`).

## Common commands

Standard training pipeline (run from `trainer/`):

```bash
python train_tokenizer.py                                     # only if rebuilding the tokenizer
python train_pretrain.py    --data_path ../dataset/pretrain_t2t_mini.jsonl
python train_full_sft.py    --from_weight pretrain   --data_path ../dataset/sft_t2t_mini.jsonl
python train_dpo.py         --from_weight full_sft   --data_path ../dataset/dpo.jsonl
python train_distillation.py                                  # student/teacher both default to full_sft
python train_lora.py        --from_weight full_sft   --lora_name lora_medical
python train_grpo.py / train_ppo.py / train_agent.py          # RLAIF / agent stages
```

Multi-GPU (DDP) — every trainer is DDP-ready via `init_distributed_mode()`:

```bash
torchrun --nproc_per_node=N train_pretrain.py ...
```

Inference / serving:

```bash
python eval_llm.py --weight full_sft                          # from repo root
cd scripts && python serve_openai_api.py --weight full_sft    # OpenAI-compatible API
cd scripts && streamlit run web_demo.py
cd scripts && python convert_model.py                         # .pth → HF format
```

Tests (pytest, currently scoped to `trainer/device_utils.py`):

```bash
python -m pytest tests/                                       # from repo root, all tests
python -m pytest tests/test_device_utils.py::test_from_arg_cpu -v   # single test
HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" python -m pytest tests/   # force CPU-only path (GPU tests skip)
```

There is no linter config and no build step. Beyond the pytest suite, end-to-end validation = run `eval_llm.py` against the new weights, or run third-party benchmarks externally (C-Eval, C-MMLU, etc.).

## Cross-cutting conventions

- **Weight file naming is load-bearing.** Format: `{save_weight}_{hidden_size}{_moe?}.pth` under `out/` (e.g. `full_sft_768.pth`, `pretrain_768_moe.pth`). `init_model` and every trainer build paths from `--from_weight`, `--hidden_size`, and `--use_moe` — change one and the lookup changes. Resume state lives separately in `checkpoints/{name}_{hidden}{_moe?}_resume.pth` (full optim/scaler/scheduler/wandb-id).
- **`--from_weight none`** = train from scratch (only `train_pretrain.py` defaults to this). Every other stage chains from a previous stage's `save_weight`.
- **`--from_resume 1`** auto-detects the `_resume.pth` and continues epoch+step+optimizer+wandb run id. World-size changes are auto-rescaled in `lm_checkpoint`. Don't try to resume by reloading the half-precision `.pth` — it has no optimizer state.
- **Checkpoints are saved in `.half().cpu()`**, then trained in bf16 (default) or fp16 via the device-agnostic `torch.amp.autocast(device_type=...)` + `torch.amp.GradScaler(device_type, ...)`. The model is held in fp32 master weights.
- **GPU portability (CUDA / ROCm).** Device handling is centralized in `trainer/device_utils.py`, which is split into two layers — never hardcode `"cuda"` strings or use the deprecated `torch.cuda.amp` namespace.
  - **Free functions** for environment facts: `default_device_str()` (argparse default), `dist_backend()` (passed to `dist.init_process_group`), `set_current_device()` (binds local rank, prefers `torch.accelerator.set_device_index`), `vendor_name()`, and `is_rocm()` / `is_cuda_native()` for the rare cases that must branch by vendor.
  - **`DeviceCtx` (frozen dataclass)** for per-trainer state: build it once with `DeviceCtx.from_arg(args.device)`, rebind under DDP via `dev.for_rank(local_rank)`, then call `dev.autocast(args.dtype)` and `dev.grad_scaler(enabled=...)` instead of repeating the `device_type/dtype/nullcontext` boilerplate. CPU paths automatically return `nullcontext()` and a disabled scaler. Unknown dtype strings raise `KeyError` regardless of device.
  - PyTorch-ROCm aliases the `torch.cuda.*` API surface and exposes RCCL as the `"nccl"` backend, so most code works on AMD GPUs unchanged; the wrappers exist so future divergences live in one file. `print_device_info()` runs once from `init_model` to log vendor / device / version at startup.
- **Loss is always `res.loss + res.aux_loss`** even on dense models (aux_loss is 0). Don't strip aux_loss — it carries the MoE router load-balancing term when `use_moe=1`.
- **`MiniMindConfig` derives `intermediate_size` from `hidden_size`** (`ceil(hidden_size * π / 64) * 64`) unless explicitly overridden — bumping `hidden_size` automatically rescales the FFN.
- **Tokenizer path is `model/`**, not HuggingFace Hub. Anything calling `AutoTokenizer.from_pretrained` uses this local directory.
- **Logging uses `swanlab` aliased as `wandb`** (`import swanlab as wandb`). The `--use_wandb` flag toggles both.
- **`init_model` only logs trainable params and total params** — it doesn't freeze anything. LoRA freezing is done explicitly by `apply_lora` in `train_lora.py`.
- Comments and CLI `help=` strings are predominantly Chinese — preserve the existing language when editing nearby code.
