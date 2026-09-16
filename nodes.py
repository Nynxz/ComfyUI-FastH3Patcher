"""The pack's two nodes: read a patch file, graft it onto a model.

They are separate so the file is read once even when a workflow patches two model branches,
and so applying stays pure MODEL-in/MODEL-out.
"""

from __future__ import annotations

import comfy.utils
import folder_paths
from comfy_api.latest import io

from .graft import apply_patch
from .pack import FastH3PatchData, PackNode

#: Patches live in ComfyUI/models/model_patches/ — core already registers this folder.
FOLDER = "model_patches"


class FastH3LoadPatch(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="LoadPatch",
            display_name="Load FastH3 Patch",
            description=(
                "Loads a FastH3 patch from models/model_patches. The adaLN patch carries the "
                "8-step distillation and is what makes an fl2va model behave like FastH3; the "
                "VSA patch carries the to_gate_compress layers, which are only read by Model "
                "Sparse Attention with method=vsa and are dead weight otherwise. A combined "
                "file holding both works too — tensors are routed by what they are, not by "
                "filename, so any of them can be renamed."
            ),
            inputs=[
                io.Combo.Input(
                    "name",
                    options=folder_paths.get_filename_list(FOLDER),
                    tooltip="The patch file to read.",
                ),
            ],
            outputs=[FastH3PatchData.Output()],
        )

    @classmethod
    def execute(cls, name: str) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise(FOLDER, name)
        state_dict, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True
        )
        return io.NodeOutput({"sd": state_dict, "metadata": metadata or {}, "name": name})


class FastH3ApplyPatch(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="ApplyPatch",
            display_name="Apply FastH3 Patch",
            description=(
                "Turns a MiniMax-H3 fl2va model into FastH3 by replacing the adaLN timestep "
                "path and, optionally, adding the VSA gate layers. The transformer trunk is "
                "already identical between the two checkpoints, so nothing there is touched.\n\n"
                "Chain two of these to apply both files: the adaLN patch is always needed, the "
                "VSA gates only if you follow with Model Sparse Attention (method=vsa). Order "
                "does not matter. The base weights are untouched on disk and the patch is "
                "undone when ComfyUI unpatches the model."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="A MiniMax-H3 model, normally minimax_h3_fl2va_pruned_bf16.",
                ),
                FastH3PatchData.Input("fasth3_patch"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, fasth3_patch) -> io.NodeOutput:
        patcher, _report = apply_patch(model, fasth3_patch["sd"], fasth3_patch.get("name", "patch"))
        return io.NodeOutput(patcher)
