"""ComfyUI-FastH3Patcher — run FastVideo's FastH3 as a patch on MiniMax-H3 fl2va.

FastH3 ships as a 44 GB checkpoint, but it is the fl2va model with two things changed: a
reparameterised adaLN timestep path (~149 MB, this is the 8-step distillation) and 50 new
`to_gate_compress` layers (~3.85 GB, read only by VSA sparse attention). The trunk is close
enough that reconstructing it from fl2va lands within 1.08e-4 relative error. So users keep the
model they already have and download 149 MB instead of 44 GB.

    pack.json   the pack's identity: namespace, display name, menu category.
    pack.py     that identity, the wire type, and the node base class.
    graft.py    the grafting logic, and why this cannot be a LoRA.
    nodes.py    the two nodes.

There is no `web/`: both nodes are plain MODEL-in/MODEL-out patchers with nothing to render.
"""

from comfy_api.latest import ComfyExtension, io

from .nodes import FastH3ApplyPatch, FastH3LoadPatch


class FastH3PatcherExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FastH3LoadPatch, FastH3ApplyPatch]


async def comfy_entrypoint() -> ComfyExtension:
    return FastH3PatcherExtension()


__all__ = ["FastH3PatcherExtension", "comfy_entrypoint"]
