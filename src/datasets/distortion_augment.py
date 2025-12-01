import io
import random
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = arr + noise
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def add_gaussian_blur(img: Image.Image, kernel_size: int, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def jpeg_compress(img: Image.Image, quality_factor: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(10, min(quality_factor, 95)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def adjust_brightness_contrast(img: Image.Image, brightness_delta: float, contrast_factor: float) -> Image.Image:
    b = ImageEnhance.Brightness(img).enhance(1.0 + brightness_delta)
    c = ImageEnhance.Contrast(b).enhance(contrast_factor)
    return c

