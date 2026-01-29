import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color

def calculate_psnr(img1, img2):
    return float(psnr(img1, img2, data_range=1.0))

def calculate_ssim(img1, img2):
    min_side = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_side if min_side % 2 != 0 else min_side - 1)
    
    if win_size < 3:
        return 0.0

    return float(ssim(img1, img2, win_size=win_size, channel_axis=2, data_range=1.0))

def calculate_ciede2000(img1, img2):
    lab1 = color.rgb2lab(img1)
    lab2 = color.rgb2lab(img2)
    
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    return float(np.mean(delta_e))

def tensor_to_numpy(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img
