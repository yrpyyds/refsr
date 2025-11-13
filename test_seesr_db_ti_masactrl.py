'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything
import sys
sys.path.append("/root/autodl-tmp/sr/MasaCtrl")
from diffusers import DDIMScheduler,StableDiffusionPipeline
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControlMaskAuto
from torchvision.transforms.functional import to_pil_image
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
import safetensors
import torch
import torch.utils.checkpoint
import random
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)

@torch.no_grad()
def masactrl_invert(pipe, image: torch.Tensor, prompt, num_inference_steps=50, guidance_scale=7.5):
    """
    pipe: 任何有 self.unet / self.vae / self.scheduler / self.tokenizer / self.text_encoder 的 pipeline
          比如你的 StableDiffusionControlNetPipeline(SeeSR)
    """
    device = image.device
    batch_size = image.shape[0]

    if isinstance(prompt, list):
        if batch_size == 1:
            image = image.expand(len(prompt), -1, -1, -1)
    elif isinstance(prompt, str):
        if batch_size > 1:
            prompt = [prompt] * batch_size

    # 1. text embeddings
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    # 2. image → latent
    latents = pipe.vae.encode(image)['latent_dist'].mean * 0.18215

    # 3. CFG
    if guidance_scale > 1.:
        unconditional_input = pipe.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = pipe.text_encoder(unconditional_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

    # 4. DDIM inversion loop
    pipe.scheduler.set_timesteps(num_inference_steps)
    latents_list = [latents]

    for i, t in enumerate(reversed(pipe.scheduler.timesteps)):
        if guidance_scale > 1.:
            model_inputs = torch.cat([latents] * 2)
        else:
            model_inputs = latents

        noise_pred = pipe.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
        if guidance_scale > 1.:
            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

        # 这里用你原来 next_step 的公式
        next_step = t
        timestep = min(t - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps, 999)
        alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep] if timestep >= 0 else pipe.scheduler.final_alpha_cumprod
        alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        latents, pred_x0 = (
            alpha_prod_t_next**0.5 * ((latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5) +
            (1 - alpha_prod_t_next)**0.5 * noise_pred,
            None,
        )

        latents_list.append(latents)

    return latents, latents_list

@torch.no_grad()
def masactrl_sample(
    pipe,
    prompt,
    latents=None,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    neg_prompt=None,
    ref_intermediate_latents=None,
):
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else (latents.device if latents is not None else "cuda")

    if isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = 1
        prompt = [prompt]

    # 1. 文本 embedding
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    # 2. initial latent
    latents_shape = (batch_size, pipe.unet.in_channels, height // 8, width // 8)
    if latents is None:
        latents = torch.randn(latents_shape, device=device)
    else:
        assert latents.shape == latents_shape

    # 3. CFG
    if guidance_scale > 1.:
        if neg_prompt:
            uc_text = neg_prompt
        else:
            uc_text = ""
        unconditional_input = pipe.tokenizer(
            [uc_text] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = pipe.text_encoder(unconditional_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

    # 4. DDIM 采样
    pipe.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(pipe.scheduler.timesteps):
        if ref_intermediate_latents is not None:
            latents_ref = ref_intermediate_latents[-1 - i]
            _, latents_cur = latents.chunk(2)
            latents = torch.cat([latents_ref, latents_cur])

        if guidance_scale > 1.:
            model_inputs = torch.cat([latents] * 2)
        else:
            model_inputs = latents

        noise_pred = pipe.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
        if guidance_scale > 1.:
            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

        # 用你原来的 step 公式
        prev_timestep = t - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else pipe.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * noise_pred
        latents = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir

    # 5. latent → image
    latents_decode = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents_decode)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    # scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
    unet_raw = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    if args.use_lora:
        print("loading lora...")
        unet.load_attn_procs(args.lora_path)
        unet_raw.load_attn_procs(args.lora_path)
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet_raw.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            unet_raw.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    sd2_base_pipeline = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet_raw, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,)
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_raw.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    print('loading textual inversion...')
    validation_pipeline.load_textual_inversion(
        args.ti_path,
        token=args.placeholder_token,  # 这个 token 必须和训练时一致
    )
    # sd2_base_pipeline.load_textual_inversion(
    #     args.ti_path,
    #     token=args.placeholder_token,  # 这个 token 必须和训练时一致
    # )
    sd2_base_pipeline.tokenizer = validation_pipeline.tokenizer
    sd2_base_pipeline.text_encoder = validation_pipeline.text_encoder

    editor = MutualSelfAttentionControlMaskAuto(
        start_step=4,
        start_layer=10,
        total_steps=args.num_inference_steps,                     
        ref_token_idx=[1],
        cur_token_idx=[1],
        mask_save_dir="/root/autodl-tmp/sr/mask_debug/",
        model_type="SD2",                    
    )
    regiter_attention_editor_diffusers(validation_pipeline, editor)
    return validation_pipeline,sd2_base_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='/root/autodl-tmp/sr/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=False,):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline,sd2_base_pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")

            validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            ref_image_path = os.path.join(args.ref_img_path,random.sample(os.listdir(args.ref_img_path),1)[0])
            ref_image = Image.open(ref_image_path).convert("RGB")
            print(f"Loaded reference image from {ref_image_path} for masactrl")
            ref_image = np.array(ref_image)
            ref_image = torch.from_numpy(ref_image).float() / 127.5 - 1
            ref_image = ref_image.permute(2, 0, 1).unsqueeze(0).to(accelerator.device)
            print("Performing DDIM inversion on reference image...")
            start_code, latents_list = masactrl_invert(
                sd2_base_pipeline,
                ref_image,
                prompt="", 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            start_code = start_code.expand(2, -1, -1, -1)
            # === user_prompt 替换逻辑 ===
            if hasattr(args, "user_prompt") and args.user_prompt:
                class_list = [c.strip() for c in args.user_classes.split(",") if c.strip()]
                found = False

                for cls in class_list:
                    import re
                    # 不区分大小写匹配
                    if re.search(rf"(?i)\b{re.escape(cls)}\b", validation_prompt):
                        validation_prompt = re.sub(rf"(?i)\b{re.escape(cls)}\b", args.user_prompt, validation_prompt)
                        found = True

                # 如果没有匹配到任何类别，则追加 user_prompt
                if not found:
                    if validation_prompt.endswith(","):
                        validation_prompt += f" {args.user_prompt}"
                    else:
                        validation_prompt += f", {args.user_prompt}"
            prompts = [
                "",
                validation_prompt
            ]
            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                file = open(txt_save_path, "w")
                file.write(validation_prompt)
                file.close()
            print(f'{validation_prompt}')

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

            for sample_idx in range(args.sample_times):  

                with torch.autocast("cuda"):
                    # image = pipeline(
                    #         validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                    #         guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                    #         start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                    #         latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                    #         args=args,
                    #     ).images[0]
                    image_masactrl = masactrl_sample(
                        pipeline,
                        prompts,
                        latents=start_code,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                    )
                    out_image = image_masactrl[-1]
                    image = to_pil_image(out_image.cpu().squeeze(0))
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default='/root/autodl-tmp/sr/models/seesr_model/seesr/')
    parser.add_argument("--ram_ft_path", type=str, default='/root/autodl-tmp/sr/models/seesr_model/DAPE.pth')
    parser.add_argument("--pretrained_model_path", type=str, default='/root/autodl-tmp/sr/models/sd2_base')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=r'/root/autodl-tmp/sr/datasets/test/backpage/test_data/')
    parser.add_argument("--output_dir", type=str, default=r'/root/autodl-tmp/sr/datasets/test/backpage3/predict_db_ti_masactrl/')
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=4) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", default=True)
    parser.add_argument("--use_lora",action="store_true")
    # parser.add_argument("--lora_path",default=r'/root/autodl-tmp/sr/models/output/dreambooth/lora-dreambooth-model9-train_text_encoder/')
    parser.add_argument("--lora_path",default=r'/root/autodl-tmp/sr/models/output/dreambooth/lora-dreambooth-model10-textual_inversion/checkpoint-1000/')
    parser.add_argument("--user_prompt",default='sksbackpack')
    parser.add_argument("--user_classes",default='backpack,bag')
    parser.add_argument("--ti_path",default=r'/root/autodl-tmp/sr/models/output/dreambooth/lora-dreambooth-model10-textual_inversion/checkpoint-1000/learned_embeds.safetensors')
    parser.add_argument("--placeholder_token",default='sksbackpack')
    parser.add_argument("--ref_img_path",default='/root/autodl-tmp/sr/datasets/train/dataset/backpack/train/')
    args = parser.parse_args()
    main(args)



