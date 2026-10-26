# ComfyUI-Flux3-PDD

ComfyUI node for [FLUX.3 klein turbo PDD](https://huggingface.co/NoiseToNova/Flux-3-Klein-turbo-pdd-4_8-steps), a
few-step (8 / 4) LoRA for FLUX.3 klein base: the LoRA plus the multi-step output head, sampled in a few jumps along
the student's training grid.

Needs ComfyUI with FLUX.3 support (0.42 or newer).

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/maxs-kan/ComfyUI-Flux3-PDD.git
```


## Workflow

Drag [workflows/flux3_pdd_t2i.json](workflows/flux3_pdd_t2i.json) onto the canvas. It is BFL's base text-to-image
workflow (prompt, bounding boxes, FLUX.3 prompt upsampler) with the 50-step CFG sampling replaced by the PDD loader;


1. Load Diffusion Model (FLUX.3 klein base), Load CLIP (type `flux3`), Load VAE, as in the FLUX.3 klein template.
2. **FLUX.3 PDD Loader**: model -> `model`, the converted file -> `pdd_name`, `steps` 8 or 4. Outputs MODEL and
   SIGMAS.
3. SamplerCustomAdvanced with:
   - guider: CFGGuider with cfg 1 (CFG is distilled into the student), or BasicGuider;
   - sampler: KSamplerSelect `euler`;
   - sigmas: from the loader;
   - noise: RandomNoise; latent: the template's empty latent node.

`steps` is 8 or 4: the student was trained to jump 16 to 32 steps of its 128-step grid. `2` (two jumps of 64) is also offered for a very fast preview: it was never trained, so the quality is bad, but it shows the overall composition.
