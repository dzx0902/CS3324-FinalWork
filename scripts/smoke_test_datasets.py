from PIL import Image
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.distortion_augment import add_gaussian_noise, add_gaussian_blur, jpeg_compress, adjust_brightness_contrast

img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype('uint8'))
print('noise', add_gaussian_noise(img, 10.0).size)
print('blur', add_gaussian_blur(img, 3, 1.5).size)
print('jpeg', jpeg_compress(img, 60).size)
print('bc', adjust_brightness_contrast(img, 0.1, 1.1).size)

