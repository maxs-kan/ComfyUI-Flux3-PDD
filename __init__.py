"""ComfyUI node for a FLUX.3 klein PDD student (better_trainer Flux3PDDTrainer).

Clone this repo into ComfyUI/custom_nodes/ (ComfyUI with FLUX.3 support, >= 0.42) and convert the export with
scripts/convert_pdd_to_comfy.py into ComfyUI/models/loras/.

FLUX.3 PDD Loader (MODEL + file + steps -> MODEL, SIGMAS):
  * applies the LoRA (ComfyUI's LoRA loader, strength `lora_strength`);
  * replaces final_layer.image.linear (the diffusers proj_out) with the multi-step head: for a jump n -> m on the
    training grid the dt-weighted mean of heads n..m-1, so the sampler's Euler step sigma_n -> sigma_m equals the
    training update x_n + sum_{n<=t<m} dt_t * v_t. The jump is picked from the sigma of each model call (nearest jump
    start), not from a step counter, so CFG, batching and restarts stay in sync;
  * returns SIGMAS: the jump starts on the training grid (already shifted, pdd_config `sigmas`) and the final 0.

Sample with SamplerCustomAdvanced: KSamplerSelect `euler`, these SIGMAS, and CFGGuider with cfg 1 (the teacher's CFG,
pdd_config `guidance_scale`, is distilled into the student) or BasicGuider. The model's own shift does not apply to
custom sigmas.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.sd
import comfy.utils
import folder_paths

HEAD_MODULE = "diffusion_model.final_layer.image.linear"
STEPS = ["8", "4", "2"]  # 128 grid steps in jumps of 16 (L_min) or 32 (L_max)


def jump_schedule(steps: int, trajectory_steps: int, L_min: int, L_max: int) -> list[int]:
    """Grid indices of `steps` equal jumps; a jump is a multiple of L_min (training block starts) and at most L_max."""
    block = trajectory_steps // steps if steps > 0 else 0
    if steps < 1 or trajectory_steps % steps or block % L_min or not L_min <= block:
        raise ValueError(f"{steps} steps do not fit {trajectory_steps} grid steps with jumps of {L_min}..{L_max} "
                         f"(multiples of {L_min})")
    return list(range(0, trajectory_steps + 1, block))


class PDDHead(nn.Module):
    """final_layer.image.linear with one fused head per jump; `index` is set per model call by the wrapper."""

    def __init__(self, heads: torch.Tensor):
        super().__init__()
        self.heads = heads  # (jumps, out, in) fp32 on the CPU; a plain attribute, not a parameter ComfyUI would load
        self.index = 0
        self._cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._cache:
            self._cache = {key: self.heads.to(device=x.device, dtype=x.dtype)}
        return F.linear(x, self._cache[key][self.index])


class Flux3PDDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pdd_name": (folder_paths.get_filename_list("loras"),),
                "steps": (STEPS, {"default": STEPS[0], "tooltip": "jumps on the 128-step training grid; only 8 and 4 were trained"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "SIGMAS")
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = "FLUX.3 PDD student: LoRA + multi-step head, and the sigmas of `steps` jumps on its training grid."

    def load(self, model, pdd_name, steps, lora_strength):
        path = folder_paths.get_full_path_or_raise("loras", pdd_name)
        state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if not metadata or "pdd_config" not in metadata or "pdd.heads" not in state_dict:
            raise ValueError(f"{pdd_name} is not a FLUX.3 PDD file (scripts/convert_pdd_to_comfy.py)")
        config = json.loads(metadata["pdd_config"])
        grid = torch.tensor(config["sigmas"], dtype=torch.float64)  # (trajectory_steps + 1,), 1 -> 0, shifted
        heads = state_dict.pop("pdd.heads").float()                   # (trajectory_steps, out, in)

        boundaries = jump_schedule(int(steps), config["trajectory_steps"], config["L_min"], config["L_max"])
        dt = grid.diff()
        fused = []
        for n, m in zip(boundaries[:-1], boundaries[1:]):
            mix = (dt[n:m] / dt[n:m].sum()).float()
            fused.append(torch.einsum("t,toi->oi", mix, heads[n:m]))
        head = PDDHead(torch.stack(fused))
        starts = grid[boundaries[:-1]].float()

        patched, _ = comfy.sd.load_lora_for_models(model, None, state_dict, lora_strength, 0)
        base = patched.get_model_object(HEAD_MODULE)
        if tuple(base.weight.shape) != tuple(heads.shape[1:]):
            raise ValueError(f"head {tuple(heads.shape)} does not match {HEAD_MODULE} {tuple(base.weight.shape)}")
        patched.add_object_patch(HEAD_MODULE, head)

        previous = patched.model_options.get("model_function_wrapper")

        def pick_head(apply_model, args):
            sigma = float(args["timestep"].flatten()[0])
            head.index = int((starts - sigma).abs().argmin())
            if previous is not None:
                return previous(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        patched.set_model_unet_function_wrapper(pick_head)
        sigmas = grid[boundaries].float()  # jump starts + the grid's final 0
        return patched, sigmas


NODE_CLASS_MAPPINGS = {"Flux3PDDLoader": Flux3PDDLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Flux3PDDLoader": "FLUX.3 PDD Loader"}
