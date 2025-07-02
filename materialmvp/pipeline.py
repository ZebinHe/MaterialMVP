from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy
import torch
import torch.utils.checkpoint
import torch.distributed
import numpy as np
import transformers
from PIL import Image
from einops import rearrange
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import diffusers
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
)

from diffusers.utils import deprecate
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from .modules import UNet2p5DConditionModel
from .attn_processor import SelfAttnProcessor2_0, RefAttnProcessor2_0, PoseRoPEAttnProcessor2_0

__all__ = [
    "MaterialMVPPipeline",
    "UNet2p5DConditionModel",
    "SelfAttnProcessor2_0",
    "RefAttnProcessor2_0",
    "PoseRoPEAttnProcessor2_0",
]


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class MaterialMVPPipeline(StableDiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        safety_checker=None,
        use_torch_compile=False,
    ):
        DiffusionPipeline.__init__(self)

        safety_checker = None
        self.register_modules(
            vae=torch.compile(vae) if use_torch_compile else vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=torch.compile(feature_extractor) if use_torch_compile else feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = UNet2p5DConditionModel(self.unet, None, self.scheduler)

    def eval(self):
        self.unet.eval()
        self.vae.eval()

    def set_pbr_settings(self, pbr_settings: List[str]):
        self.pbr_settings = pbr_settings

    def set_learned_parameters(self):

        freezed_names = ["attn1", "unet_dual"]
        added_learned_names = ["albedo", "mr", "dino"]

        for name, params in self.unet.named_parameters():
            if any(freeze_name in name for freeze_name in freezed_names) and all(
                learned_name not in name for learned_name in added_learned_names
            ):
                params.requires_grad = False
            else:
                params.requires_grad = True

    def prepare(self):
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = UNet2p5DConditionModel(self.unet, None, self.scheduler).eval()

    @torch.no_grad()
    def encode_images(self, images):

        B = images.shape[0]
        images = rearrange(images, "b n c h w -> (b n) c h w")

        dtype = next(self.vae.parameters()).dtype
        images = (images - 0.5) * 2.0
        posterior = self.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        latents = rearrange(latents, "(b n) c h w -> b n c h w", b=B)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        images=None,
        prompt=None,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=3.0,
        output_type: Optional[str] = "pil",
        width=512,
        height=512,
        num_inference_steps=15,
        return_dict=True,
        sync_condition=None,
        **cached_condition,
    ):

        self.prepare()
        if images is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(images, torch.Tensor)

        if not isinstance(images, List):
            images = [images]

        images = [to_rgb_image(image) for image in images]
        images_vae = [torch.tensor(np.array(image) / 255.0) for image in images]
        images_vae = [image_vae.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0) for image_vae in images_vae]
        images_vae = torch.cat(images_vae, dim=1)
        images_vae = images_vae.to(device=self.vae.device, dtype=self.unet.dtype)

        batch_size = images_vae.shape[0]
        N_ref = images_vae.shape[1]

        assert batch_size == 1
        assert num_images_per_prompt == 1

        if self.unet.use_ra:
            ref_latents = self.encode_images(images_vae)
            cached_condition["ref_latents"] = ref_latents

        def convert_pil_list_to_tensor(images):
            bg_c = [1.0, 1.0, 1.0]
            images_tensor = []
            for batch_imgs in images:
                view_imgs = []
                for pil_img in batch_imgs:
                    img = numpy.asarray(pil_img, dtype=numpy.float32) / 255.0
                    if img.shape[2] > 3:
                        alpha = img[:, :, 3:]
                        img = img[:, :, :3] * alpha + bg_c * (1 - alpha)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().half().to("cuda")
                    view_imgs.append(img)
                view_imgs = torch.cat(view_imgs, dim=0)
                images_tensor.append(view_imgs.unsqueeze(0))

            images_tensor = torch.cat(images_tensor, dim=0)
            return images_tensor

        if "images_normal" in cached_condition:
            if isinstance(cached_condition["images_normal"], List):
                cached_condition["images_normal"] = convert_pil_list_to_tensor(cached_condition["images_normal"])

            cached_condition["embeds_normal"] = self.encode_images(cached_condition["images_normal"])

        if "images_position" in cached_condition:

            if isinstance(cached_condition["images_position"], List):
                cached_condition["images_position"] = convert_pil_list_to_tensor(cached_condition["images_position"])

            cached_condition["position_maps"] = cached_condition["images_position"]
            cached_condition["embeds_position"] = self.encode_images(cached_condition["images_position"])

        if self.unet.use_learned_text_clip:

            all_shading_tokens = []
            for token in self.unet.pbr_setting:
                all_shading_tokens.append(
                    getattr(self.unet, f"learned_text_clip_{token}").unsqueeze(dim=0).repeat(batch_size, 1, 1)
                )
            prompt_embeds = torch.stack(all_shading_tokens, dim=1)
            negative_prompt_embeds = torch.stack(all_shading_tokens, dim=1)
            # negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        else:
            if prompt is None:
                prompt = "high quality"
            if isinstance(prompt, str):
                prompt = [prompt for _ in range(batch_size)]
            device = self._execution_device
            prompt_embeds, _ = self.encode_prompt(
                prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False
            )

            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt for _ in range(batch_size)]
            if negative_prompt is not None:
                negative_prompt_embeds, _ = self.encode_prompt(
                    negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                )
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        if guidance_scale > 1:
            if self.unet.use_ra:
                cached_condition["ref_latents"] = cached_condition["ref_latents"].repeat(
                    3, *([1] * (cached_condition["ref_latents"].dim() - 1))
                )
                cached_condition["ref_scale"] = torch.as_tensor([0.0, 1.0, 1.0]).to(cached_condition["ref_latents"])

            if self.unet.use_dino:
                zero_states = torch.zeros_like(cached_condition["dino_hidden_states"])
                cached_condition["dino_hidden_states"] = torch.cat(
                    [zero_states, zero_states, cached_condition["dino_hidden_states"]]
                )

                del zero_states
            if "embeds_normal" in cached_condition:
                cached_condition["embeds_normal"] = cached_condition["embeds_normal"].repeat(
                    3, *([1] * (cached_condition["embeds_normal"].dim() - 1))
                )

            if "embeds_position" in cached_condition:
                cached_condition["embeds_position"] = cached_condition["embeds_position"].repeat(
                    3, *([1] * (cached_condition["embeds_position"].dim() - 1))
                )

            if "position_maps" in cached_condition:
                cached_condition["position_maps"] = cached_condition["position_maps"].repeat(
                    3, *([1] * (cached_condition["position_maps"].dim() - 1))
                )

        images = self.denoise(
            None,
            *args,
            cross_attention_kwargs=None,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            width=width,
            height=height,
            return_dict=return_dict,
            **cached_condition,
        )

        return images

    def denoise(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # open cache
        kwargs["cache"] = {}

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated,"
                "consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated,"
                "consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        assert num_images_per_prompt == 1
        # 5. Prepare latent variables
        n_pbr = len(self.unet.pbr_setting)
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * kwargs["num_in_batch"] * n_pbr,  # num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latents = rearrange(
                    latents, "(b n_pbr n) c h w -> b n_pbr n c h w", n=kwargs["num_in_batch"], n_pbr=n_pbr
                )
                # latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents
                latent_model_input = latents.repeat(3, 1, 1, 1, 1, 1) if self.do_classifier_free_guidance else latents
                latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = rearrange(
                    latent_model_input, "(b n_pbr n) c h w ->b n_pbr n c h w", n=kwargs["num_in_batch"], n_pbr=n_pbr
                )

                # predict the noise residual

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **kwargs,
                )[0]
                latents = rearrange(latents, "b n_pbr n c h w -> (b n_pbr n) c h w")
                # perform guidance
                if self.do_classifier_free_guidance:
                    # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred_uncond, noise_pred_ref, noise_pred_full = noise_pred.chunk(3)

                    if "camera_azims" in kwargs.keys():
                        camera_azims = kwargs["camera_azims"]
                    else:
                        camera_azims = [0] * kwargs["num_in_batch"]

                    def cam_mapping(azim):
                        if azim < 90 and azim >= 0:
                            return float(azim) / 90.0 + 1
                        elif azim >= 90 and azim < 330:
                            return 2.0
                        else:
                            return -float(azim) / 90.0 + 5.0

                    view_scale_tensor = (
                        torch.from_numpy(np.asarray([cam_mapping(azim) for azim in camera_azims]))
                        .unsqueeze(0)
                        .repeat(n_pbr, 1)
                        .view(-1)
                        .to(noise_pred_uncond)[:, None, None, None]
                    )
                    noise_pred = noise_pred_uncond + self.guidance_scale * view_scale_tensor * (
                        noise_pred_ref - noise_pred_uncond
                    )
                    noise_pred += self.guidance_scale * view_scale_tensor * (noise_pred_full - noise_pred_ref)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_ref, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents[:, :num_channels_latents, :, :], **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
