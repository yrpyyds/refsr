import sys
sys.path.append("/root/autodl-tmp/sr/MasaCtrl")

import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControlMaskAuto  # 假设你把上面的类放在这个文件里
from masactrl.diffuser_utils import MasaCtrlPipeline

device = "cuda"

# 1️⃣ 加载 Stable Diffusion 2 模型
model_path = "/root/autodl-tmp/sr/models/sd2_base"
pipe = DiffusionPipeline.from_pretrained(
    model_path,
).to(device)

# 2️⃣ 构建 mask-auto 版 MasaCtrl 编辑器
editor = MutualSelfAttentionControlMaskAuto(
    start_step=4,          # 从第4步开始启用 mutual attention
    start_layer=10,        # SD 2 有70层，可以选50之后的decoder层#33层
    total_steps=50,        # 采样总步数
    thres=0.1,             # mask 阈值
    ref_token_idx=[1],     # 主体词 token 索引（参考 cross-attn map 生成）
    cur_token_idx=[1],     # 当前图对应 token 索引
    mask_save_dir="/root/autodl-tmp/sr/mask_debug1/",
    # model_type="SDXL"      # 这里设置 SDXL 是因为SD2层数接近 70
    model_type="SD2"
)

# 3️⃣ 注册到 U-Net attention 层
regiter_attention_editor_diffusers(pipe, editor)

# 4️⃣ 设置两个 prompt
prompts = [
    "a boy sitting on a chair, outdoors",   # 源 prompt
    "a boy standing on the grass, outdoors" # 目标 prompt
]

# 5️⃣ 初始化同一份噪声（保持两图对应）
seed = 42
torch.manual_seed(seed)
start_code = torch.randn([1, 4, 64, 64], device=device)
start_code = start_code.expand(len(prompts), -1, -1, -1)

# 6️⃣ 推理生成
images = pipe(
    prompt=prompts,
    latents=start_code,
    guidance_scale=7.5,
    num_inference_steps=50
).images

# 7️⃣ 保存结果
images[0].save("/root/autodl-tmp/sr/mask_debug1/source_no_mask.png")
images[1].save("/root/autodl-tmp/sr/mask_debug1/target_with_masactrl.png")
print("MasaCtrl MaskAuto applied successfully!")
