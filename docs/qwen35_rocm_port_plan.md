# Qwen3.5 ROCm Port Plan

This branch is the ROCm/HIP starting point for the existing Qwen3.5 hybrid-attention work.
The goal is to port the current Metal fast paths to ROCm while preserving the current Metal path,
the existing CUDA branch shape, and the DotCache acceptance/benchmark harness.

## What is algorithmic vs backend-specific

### Algorithmic

These pieces should stay the same across Metal, CUDA, and ROCm:

- Dense-control long-context path selection in `candle-transformers/src/models/qwen3_5.rs`
- Packed linear-prefill convolution over the mixed QKV stream
- Full-attention prefill row update with online max / denominator accumulation
- DeltaNet recurrent update equations
- DeltaNet chunk-step, chunk-windowed, chunk-scan, state-scan, chunk-fused, and full-scan math
- Acceptance/control comparison semantics in DotCache

### Backend-specific

These pieces must be reimplemented per backend:

- `Device` / `Storage` / `CustomOp*` dispatch in `candle-core`
- Kernel module build and symbol loading
- Buffer allocation / copies / launch configuration / stream semantics
- Backend-specific gating in `qwen3_5.rs`
- Quantized or safetensor fast paths that currently branch on `Cuda` or `Metal`

## Current ROCm milestone

The first milestone on this machine is intentionally smaller than the CUDA branch:

1. Add a first-class `Hip` seam to `candle-core`
   - `DeviceLocation::Hip`
   - `Device::Hip`
   - `Storage::Hip`
   - `HipDevice` / `HipStorage`
   - `hip_fwd` hooks on `CustomOp*` and `InplaceOp*`
   - explicit `NotCompiledWithHipSupport` errors

2. Add a HIP kernel source that mirrors the Qwen3.5 CUDA entry-point surface
   - `linear_prefill_conv_pack_*`
   - `full_attention_prefill_*`
   - all DeltaNet kernel names preserved as ROCm placeholders for now

3. Keep runtime behavior unchanged
   - Metal path remains the default mature implementation
   - CUDA branch remains separate
   - no DotCache ROCm runtime changes until Candle has a real HIP backend

## Kernel plan

### Full-attention prefill megakernel

- Reuse the CUDA algorithm directly.
- The current HIP file already contains a compileable version of the dense-control prefill kernel.
- Next step: wire this through a real HIP custom-op path in `qwen3_5.rs`.

### Packed linear-prefill path

- Reuse the CUDA loop structure directly.
- The current HIP file already contains a compileable version of the packed linear-prefill kernel.
- Next step: add HIP launch glue and tensor layout validation alongside the future HIP backend.

### DeltaNet chunk-step / long-context path

- Preserve the CUDA/Metal surface and gating names.
- The current HIP file reserves the full DeltaNet entry surface with placeholder kernels.
- Next step: lift the DeltaNet implementations from `qwen35_delta.cu` into HIP in the following order:
  1. recurrent prefill
  2. chunk-step raw
  3. chunk-windowed raw
  4. state-scan / chunk-fused / full-scan

## Files to change first

These are the exact first files for the ROCm branch:

- `candle-core/src/device.rs`
- `candle-core/src/storage.rs`
- `candle-core/src/custom_op.rs`
- `candle-core/src/error.rs`
- `candle-core/src/dummy_hip_backend.rs`
- `candle-core/src/quantized/dummy_hip.rs`
- `candle-kernels/src/qwen35_delta.hip`
- `candle-kernels/scripts/qwen35_hip_smoke.sh`

The first runtime integration files after that should be:

- `candle-transformers/src/models/qwen3_5.rs`
- a future `candle-core/src/hip_backend/*`
- then DotCache benchmark / acceptance entry points

## Local validation on this ROCm host

Current local validation is:

- `cargo check -p candle-core`
- `cargo check -p candle-transformers`
- `candle-kernels/scripts/qwen35_hip_smoke.sh`

This is enough to keep the branch honest while the real HIP backend is still missing.
