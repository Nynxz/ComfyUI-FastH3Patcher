"""Grafting FastH3 weights onto a MiniMax-H3 fl2va model.

FastVideo's FastH3 checkpoint is, weight for weight, the MiniMax-H3 fl2va model. Comparing the
two bf16 releases tensor by tensor: all 532 shared tensors match in shape *and* dtype, and the
entire transformer trunk (attn qkv/out + mlp fc1/fc2, ~21.5 B of the 22 B parameters) differs by
at most **2.4e-4 in absolute terms** — below one bf16 step for any weight above 0.0625, so
85-95% of those elements are bit-identical. Reconstructing FastH3 as "fl2va + the parts that
actually changed" lands within 1.08e-4 relative Frobenius error of the real checkpoint.

What genuinely differs is only two things:

* **The adaLN timestep path** (`adaln_proj.*`, `adaln_t_table`) — completely reparameterised:
  relative Frobenius delta ~3.1 with *negative* cosine similarity. This is the 8-step
  distillation. ~149 MB.
* **`attn.to_gate_compress`** — 50 layers that exist only in FastH3. ~3.85 GB.

## Why this is not a LoRA

Three independent reasons, any one of which is fatal:

1. The trunk delta is nil — there is nothing to extract.
2. `adaln_proj.linear.weight` is `[96768, 8]`, so it is **already rank <= 8**. A rank-r
   factorisation of it is larger than the tensor.
3. `to_gate_compress` has no base tensor to diff against, and worse, no *module*:
   `comfy/model_detection.py` sets `gate_compress` from whether the key is in the checkpoint,
   so an fl2va-built model has `Attention.to_gate_compress = None`. `ModelPatcher.add_patches`
   only touches keys already in `model_state_dict()`, so it can never reach a layer that does
   not exist. The layer has to be grafted as an object, which is what this module does.

## Why object patches rather than the weight-patch path

`add_object_patch` hands ComfyUI a finished module/tensor instead of a delta to fold in, which
matters here for correctness, not just convenience:

* With a curve table present, `comfy/ldm/minimax/model.py` builds the adaLN linears in
  **float32** while the checkpoint stores them as **fp16**. We cast to the module's dtype.
* The ordinary weight-patch path would be worse than a no-op for `adaln_t_table`:
  `model_management.lora_compute_dtype()` returns **fp16** on most CUDA cards, so a float32
  time-embedding basis would round-trip through fp16 and lose precision.
* `ModelPatcher.patch_model()` applies object patches *before* `load()`, and `load()` walks
  `named_modules()` — so a grafted gate still gets normal device placement and lowvram offload,
  and `unpatch_model()` restores `to_gate_compress` to `None` cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import comfy.ops
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model

#: Suffix identifying the VSA gate weights, which need a module built for them.
GATE_SUFFIX = ".to_gate_compress.weight"


@dataclass
class GraftReport:
    """What a single patch application actually changed."""

    params: int = 0
    buffers: int = 0
    gates: int = 0
    added_bytes: int = 0
    unmatched: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.params + self.buffers + self.gates

    def summary(self) -> str:
        parts = []
        if self.params:
            parts.append(f"{self.params} params")
        if self.buffers:
            parts.append(f"{self.buffers} buffers")
        if self.gates:
            parts.append(f"{self.gates} VSA gate layers ({self.added_bytes / 1e9:.2f} GB)")
        return ", ".join(parts) if parts else "nothing"


def _graft_gate(patcher, block, index: int, weight: torch.Tensor) -> int:
    """Build the Linear that fl2va never instantiated, and hang it on the block's attention.

    The ops class is taken from the block's own `qkv_proj` rather than hardcoded, so the gate
    gets whatever `comfy.ops` variant the model was built with instead of a plain
    `torch.nn.Linear` that would ignore ComfyUI's casting rules -- except on a quantized
    checkpoint, where that class cannot hold a plain weight at all (see below).

    Built on the meta device: `nn.Linear.__init__` would otherwise allocate a real 77 MB tensor
    per block, 3.85 GB of it across the model, purely to throw away.
    """
    attn = block.attn
    reference = attn.qkv_proj
    out_features, in_features = weight.shape

    linear_cls = type(reference)
    reference_weight = getattr(reference, "weight", None)
    dtype = weight.dtype if reference_weight is None else reference_weight.dtype
    if not issubclass(linear_cls, torch.nn.Linear):
        # A quantized checkpoint's Linear (comfy.ops MixedPrecisionOps) is not an nn.Linear and
        # never creates a plain `.weight` -- it expects a quantized tensor with scales, loaded
        # through its own `_load_from_state_dict`. The gate weights ship as plain bf16 and there
        # is nothing to quantize them against, so build an ordinary cast-aware Linear instead of
        # an empty quantized shell. It still casts to the input's device/dtype at forward time.
        linear_cls = comfy.ops.manual_cast.Linear
        dtype = weight.dtype

    gate = linear_cls(
        in_features,
        out_features,
        bias=False,
        dtype=dtype,
        device=torch.device("meta"),
    )
    gate.weight = torch.nn.Parameter(weight.to(dtype), requires_grad=False)
    patcher.add_object_patch(f"diffusion_model.blocks.{index}.attn.to_gate_compress", gate)
    return gate.weight.numel() * gate.weight.element_size()


def apply_patch(model, state_dict: dict[str, torch.Tensor], name: str = "patch"):
    """Return a clone of `model` with `state_dict` grafted onto its diffusion model.

    Accepts either half of the split patch (adaLN core or VSA gates) or the combined file —
    every tensor is routed by what it is, so chaining two applications is the same as one
    combined application.
    """
    patcher = model.clone()
    diffusion_model = patcher.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError(
            "A FastH3 patch only applies to a MiniMax-H3 model, but this MODEL holds "
            f"{type(diffusion_model).__name__}. Load minimax_h3_fl2va_pruned_bf16.safetensors "
            "with a diffusion-model loader and patch that."
        )

    params = dict(diffusion_model.named_parameters(recurse=True))
    buffers = dict(diffusion_model.named_buffers(recurse=True))
    report = GraftReport()

    for key, value in state_dict.items():
        if key.endswith(GATE_SUFFIX):
            continue  # needs a module built for it; handled below
        target = params.get(key)
        is_param = target is not None
        if target is None:
            target = buffers.get(key)
        if target is None:
            report.unmatched.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            report.unmatched.append(f"{key} {tuple(value.shape)} != {tuple(target.shape)}")
            continue
        # Cast to the dtype the module was BUILT with, not the dtype the file happens to store:
        # with a curve table the adaln linears are float32 while the checkpoint ships fp16.
        cast = value.to(target.dtype)
        patcher.add_object_patch(
            "diffusion_model." + key,
            torch.nn.Parameter(cast, requires_grad=False) if is_param else cast,
        )
        if is_param:
            report.params += 1
        else:
            report.buffers += 1

    for index, block in enumerate(getattr(diffusion_model, "blocks", [])):
        weight = state_dict.get(f"blocks.{index}.attn{GATE_SUFFIX}")
        if weight is None:
            continue
        report.added_bytes += _graft_gate(patcher, block, index, weight)
        report.gates += 1

    if report.gates:
        # These modules do not exist when ComfyUI sizes the model for VRAM, so fold them in or
        # the gates are 3.85 GB the loader never budgeted for.
        patcher.size = patcher.model_size() + report.added_bytes

    if report.unmatched:
        logging.warning(
            "FastH3 patch '%s': %d tensors did not match the model, e.g. %s",
            name,
            len(report.unmatched),
            report.unmatched[:4],
        )
    if report.total == 0:
        raise ValueError(
            f"FastH3 patch '{name}' matched nothing in this model. Either it is not a FastH3 "
            "patch file, or it has already been applied to a different architecture."
        )
    logging.info("FastH3 patch '%s': grafted %s", name, report.summary())
    return patcher, report
