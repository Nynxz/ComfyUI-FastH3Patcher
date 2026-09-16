# ComfyUI-FastH3Patcher

Run FastVideo's **FastH3** on top of the **MiniMax-H3 fl2va** model you already have, instead of
downloading a second 44 GB checkpoint.

| you download | instead of |
|---|---|
| **4.0 GB** — the full patch, VSA included | 44 GB |
| **149 MB** — adaLN only, if you never use VSA sparse attention | 44 GB |

---

## Comparison


### Full FastH3 checkpoint vs. fl2va + patch



https://github.com/user-attachments/assets/7e60cec1-61e1-494b-80d5-dd5acd611eff


▶ [example_1.mp4](https://github.com/Nynxz/ComfyUI-FastH3Patcher/raw/main/.github/assets/example_1.mp4) · 🧩 [example workflow](https://github.com/Nynxz/ComfyUI-FastH3Patcher/raw/main/.github/assets/video_fastvideo_fasth3_t2v_patched.json) — drag onto the canvas

## Why this works

FastH3 is, weight for weight, the fl2va model. Comparing the two bf16 releases tensor by tensor:

* All 532 shared tensors match in shape **and** dtype. Zero mismatches.
* The whole transformer trunk — attention qkv/out and both MLP matrices, ~21.5 B of the 22 B
  parameters — differs by at most **2.4e-4 in absolute terms**. That is below one bf16 step for
  any weight above 0.0625, so 85–95% of those elements are literally bit-identical. Rebuilding
  FastH3 as "fl2va + the parts that changed" lands within **1.08e-4** relative Frobenius error.

Only two things genuinely differ:

1. **The adaLN timestep path** (`adaln_proj.*`, `adaln_t_table`) is completely reparameterised —
   relative Frobenius delta ~3.1, with *negative* cosine similarity. This is the 8-step
   distillation. **149 MB.**
2. **`attn.to_gate_compress`** — 50 layers that exist only in FastH3. **3.85 GB.**

### The gates are optional

From `comfy/ldm/minimax/model.py`:

```python
self.to_gate_compress = None
if gate_compress:
    # VSA gate, unused by the dense forward; consumed by sparse attention patches
```

They are read **only** by **Model Sparse Attention** with `method=vsa`, and even then a missing
gate just degrades gracefully ("running the fine stage without the coarse branch"). If you are
not using VSA, that 3.85 GB does nothing and you can drop to the 149 MB adaLN file. The
default recommendation is the full patch only because the example workflow uses VSA.

### Why it is not a LoRA

Three independent reasons, any one of them fatal:

1. The trunk delta is nil — there is nothing to extract.
2. `adaln_proj.linear.weight` is `[96768, 8]`, so it is **already rank ≤ 8**. Any rank-r
   factorisation of it is bigger than the tensor itself.
3. `to_gate_compress` has no base tensor to diff against — and no *module*.
   `comfy/model_detection.py` decides `gate_compress` from whether the key is in the checkpoint,
   so an fl2va-built model has `Attention.to_gate_compress = None`. `add_patches` only touches
   keys already in `model_state_dict()`, so a LoRA loader can never reach a layer that does not
   exist. This pack grafts it with `add_object_patch` instead.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-FastH3Patcher
```

No dependencies.

### Download the patch files

Put them in `ComfyUI/models/model_patches/`.

**Start with the full patch** — it is what the [example workflow](#use) expects, because that
workflow runs Model Sparse Attention with `method=vsa`, and VSA is the one thing that reads the
gate layers.

| file | size | get this one if |
|---|---|---|
| [`..._patch_full.safetensors`](https://huggingface.co/nynxz/H3_Loras/resolve/main/model_patches/fasth3_8step_v2_pruned_patch_full.safetensors) | **4.0 GB** | **default.** Anything using VSA, including the example workflow |
| [`..._patch_adaln.safetensors`](https://huggingface.co/nynxz/H3_Loras/resolve/main/model_patches/separated/fasth3_8step_v2_pruned_patch_adaln.safetensors) | 149 MB | you only run dense / `sol-attn` / `sla` and want the small download |
| [`..._patch_vsa.safetensors`](https://huggingface.co/nynxz/H3_Loras/resolve/main/model_patches/separated/fasth3_8step_v2_pruned_patch_vsa.safetensors) | 3.85 GB | you took the adaLN file and later want VSA after all |

Hosted on [huggingface.co/nynxz/H3_Loras](https://huggingface.co/nynxz/H3_Loras/tree/main/model_patches).
The full file is exactly the other two combined, so take the full one **or** the pair — never
all three. The nodes route tensors by what they are, not by filename, so they can be renamed
freely.

You also need the base model, [MiniMax-H3 fl2va](https://huggingface.co/) <!-- FILL: fl2va link -->,
as `minimax_h3_fl2va_pruned_bf16.safetensors` in `ComfyUI/models/diffusion_models/`.

## Use

```
Load Diffusion Model (minimax_h3_fl2va_pruned_bf16)
        │
        ▼
Apply FastH3 Patch ◄── Load FastH3 Patch (..._patch_full.safetensors)
        │
        ▼
Model Sparse Attention (method=vsa, keep_percent=10)
```

That is the whole graph with the full patch. If you took the two separate files instead, chain a
second **Apply** for the gates:

```
Apply FastH3 Patch ◄── Load FastH3 Patch (..._patch_adaln.safetensors)
        │
        ▼
Apply FastH3 Patch ◄── Load FastH3 Patch (..._patch_vsa.safetensors)
```

Order does not matter, and chaining the pair is equivalent to applying the full file. Drop the
VSA half and the **Model Sparse Attention** node and you have a dense 8-step setup on 149 MB.

A ready-made text-to-video workflow:
[`video_fastvideo_fasth3_t2v_patched.json`](https://github.com/Nynxz/ComfyUI-FastH3Patcher/raw/main/.github/assets/video_fastvideo_fasth3_t2v_patched.json)
([view on GitHub](https://github.com/Nynxz/ComfyUI-FastH3Patcher/blob/main/.github/assets/video_fastvideo_fasth3_t2v_patched.json)) — download and drag it onto the canvas.

The base weights are never modified on disk, and the patch is undone when ComfyUI unpatches the
model — so an unpatched fl2va stays a clean fl2va.

## How the patch files were made

All three files are plain safetensors holding FastH3's own tensors verbatim — no quantisation, no
factorisation, nothing reconstructed. The adaLN file is every tensor except the 50 gates; the
VSA file is the 50 gates; the full file is both. Splitting and recombining is
lossless, and applying both in either order equals applying the combined file.

## Notes on the implementation

`graft.py` uses `add_object_patch` rather than the ordinary weight-patch path, for correctness
rather than convenience:

* With a curve table present, `model.py` builds the adaLN linears in **float32** while the
  checkpoint stores them as **fp16**. The patch casts to the module's dtype.
* The weight-patch path would be actively worse for `adaln_t_table`:
  `model_management.lora_compute_dtype()` returns **fp16** on most CUDA cards, which would
  round-trip a float32 time-embedding basis through fp16.
* `patch_model()` applies object patches *before* `load()`, and `load()` walks
  `named_modules()`, so grafted gates still get normal device placement and lowvram offload.
  The pack also corrects `ModelPatcher.size`, since the gates do not exist when ComfyUI budgets
  VRAM.

## Credits

FastH3 by [FastVideo](https://github.com/hao-ai-lab/FastVideo). MiniMax-H3 by MiniMax.
This pack only repackages the difference between the two.
