"""Synthetic distortion augmentation utilities for stage-1 pretraining."""
from __future__ import annotations
import random
from PIL import Image, ImageFilter, ImageEnhance


def random_distortion(img: Image.Image) -> Image.Image:
    ops = []
    if random.random() < 0.5:
        ops.append(lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 2.0))))
    if random.random() < 0.5:
        ops.append(lambda im: ImageEnhance.Sharpness(im).enhance(random.uniform(0.2, 1.8)))
    if random.random() < 0.5:
        ops.append(lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(0.6, 1.4)))
    if random.random() < 0.5:
        ops.append(lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3)))
    for op in ops:
        img = op(img)
    return img

