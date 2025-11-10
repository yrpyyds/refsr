import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything
import sys
sys.path.append("/root/autodl-tmp/sr/MasaCtrl")
from diffusers import DDIMScheduler
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControlMaskAuto

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ 加载 SD2 模型
model_path = "/root/autodl-tmp/sr/models/sd2_base"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

# 2️⃣ 定义图像加载函数
def load_image(image_path, device):
    image = read_image(image_path)[:3].unsqueeze(0).float() / 127.5 - 1.0
    image = F.interpolate(image, (512, 512))
    return image.to(device)

# 3️⃣ 设置输入
source_image_path = "/root/autodl-tmp/sr/corgi.png"
source_image = load_image(source_image_path, device)

prompts = [
    "",  # 源 prompt
    "a photo of a running corgi"  # 目标 prompt
]

seed_everything(42)
os.makedirs("/root/autodl-tmp/sr/mask_edit_result", exist_ok=True)

# 4️⃣ 图像 inversion（获得 latent）
start_code, latents_list = model.invert(
    source_image, prompts[0],
    guidance_scale=7.5,
    num_inference_steps=50,
    return_intermediates=True
)
start_code = start_code.expand(len(prompts), -1, -1, -1)

# 5️⃣ 构建 MaskAuto 控制器
editor = MutualSelfAttentionControlMaskAuto(
    start_step=4,
    start_layer=10,   # SD2 的 decoder 层推荐 30~45
    total_steps=50,
    thres=0.1,
    ref_token_idx=[1],
    cur_token_idx=[1],
    mask_save_dir="/root/autodl-tmp/sr/mask_edit_result/masks",
    model_type="SD2"
)
regiter_attention_editor_diffusers(model, editor)

# 6️⃣ 推理编辑
image_masactrl = model(
    prompts,
    latents=start_code,
    guidance_scale=7.5
)

# 7️⃣ 保存结果
out_image = torch.cat([
    source_image * 0.5 + 0.5,   # 原图
    image_masactrl[0:1],        # 重建图
    image_masactrl[-1:]         # 编辑后图
], dim=0)

save_image(out_image, "/root/autodl-tmp/sr/mask_edit_result/all_result.png")
print("✅ MasaCtrl MaskAuto image editing finished! Results saved.")
