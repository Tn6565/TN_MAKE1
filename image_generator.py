"""
image_generator.py
- Stable Diffusion v1.5 を使って画像を生成（CPU/GPU対応）
- Adobe Stock 4MP チェック（幅×高さ >= 4,000,000 pixels）
- 生成画像を JPEG(sRGB) にして返す
"""

import os
from typing import List, Tuple
from PIL import Image, ImageCms
import torch

# diffusers import may be heavy — import on demand
from diffusers import StableDiffusionPipeline

# 出力ディレクトリ
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初期化関数（呼び出し側で一度だけ呼ぶ想定）
def init_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: str | None = None):
    """
    Returns initialized pipeline.
    device: "cuda" or "cpu" or None (auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    # disable NSFW filter raise if not desired: pipe.safety_checker = None (not recommended)
    return pipe

# Validate Adobe Stock size (4MP or above)
def validate_size(width: int, height: int) -> Tuple[bool, str]:
    pixels = width * height
    if pixels < 4_000_000:
        return False, f"Size too small: {width}x{height} = {pixels:,} px (< 4,000,000)."
    return True, "OK"

# Ensure sRGB profile — convert to RGB (Pillow typically uses sRGB)
def ensure_srgb(img: Image.Image) -> Image.Image:
    # If embedded profile exists and is not sRGB, convert — best-effort
    try:
        icc_profile = img.info.get("icc_profile", None)
        if icc_profile:
            src_profile = ImageCms.ImageCmsProfile(io=icc_profile)
            dst_profile = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(img, src_profile, dst_profile, outputMode="RGB")
        else:
            img = img.convert("RGB")
    except Exception:
        img = img.convert("RGB")
    return img

# Generate images with pipeline (pipeline should be initialized once and reused)
def generate_images(
    pipe,
    prompt: str,
    width: int,
    height: int,
    num_images: int = 1,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20
) -> List[Image.Image]:
    """
    Returns list of PIL.Image objects (RGB JPEG-ready).
    pipe: StableDiffusionPipeline instance
    """
    ok, msg = validate_size(width, height)
    if not ok:
        raise ValueError(msg)

    images = []
    # diffusers allows batched prompts; here we call iteratively to control random seeds if needed
    for _ in range(num_images):
        out = pipe(prompt, width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        img = out.images[0]
        # ensure RGB / sRGB
        img = img.convert("RGB")
        images.append(img)
    return images

# Save image to OUTPUT_DIR with given filename (no extension required)
def save_image(img: Image.Image, filename_base: str) -> str:
    fname = f"{filename_base}.jpg" if not filename_base.lower().endswith(".jpg") else filename_base
    path = os.path.join(OUTPUT_DIR, fname)
    img.save(path, format="JPEG", quality=95)
    return path
