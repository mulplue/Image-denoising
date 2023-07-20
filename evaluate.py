import argparse
import numpy as np
import pandas as pd

from utils.evaluate import *
from utils.img import get_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./dataset/test/input/')
    parser.add_argument('--restore_path', type=str, default='./dataset/test/restore_jiahe/')
    parser.add_argument('--target_path', type=str, default='./dataset/test/target/')
    parser.add_argument('--author', type=str, default='jiahe')
    args = parser.parse_args()
    
    """Read images"""
    _, noise_imgs = get_imgs(args.input_path)
    _, restore_imgs = get_imgs(args.restore_path)
    img_names, target_imgs = get_imgs(args.target_path)

    """"""
    meric_index = ["l2_noise", "l2_restore", "ssim_noise", "ssim_restore", "csim_noise", "csim_restore"]
    df = pd.DataFrame(columns=meric_index, index=img_names)
    l2_metrics = []
    ssim_metrics = []
    csim_metrics = []
    for i, (img_name, noise_img, restore_img, target_img) in enumerate(zip(img_names, noise_imgs, restore_imgs, target_imgs)):
        l2_noise = calc_l2(target_img, noise_img)
        l2_restore = calc_l2(target_img, restore_img)
        
        ssim_noise = calc_ssim(target_img, noise_img)
        ssim_restore = calc_ssim(target_img, restore_img)
        
        csim_noise = calc_csim(target_img, noise_img)
        csim_restore = calc_csim(target_img, restore_img)
        
        all_merics = [l2_noise, l2_restore, ssim_noise, ssim_restore, csim_noise, csim_restore]
        df.iloc[i] = all_merics
        
        output_path = "results/" + args.author + ".csv"
        df.to_csv(output_path)
        