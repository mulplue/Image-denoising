import glob
import cv2
import argparse
import sys
import os
from tqdm import tqdm, trange
sys.path.append("../")
sys.path.append("E:\Study\Technology\AI\github\Image-denoising")
from utils.add_noise import noise_mask_image
from utils.utils import set_seed
from utils.img import read_image, save_image, plot_image, normalization


def get_imgs(dataset_dir):
    img_names = os.listdir(dataset_dir)
    imgs = [read_image(os.path.join(dataset_dir, x)) for x in img_names]
    return img_names, imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', type=str, default='../dataset/CBSD68/original_png/', help="dataset path")
    parser.add_argument('--output_path', type=str, default='../dataset/CBSD68/noisy/', help="dataset path")
    parser.add_argument('--seed', type=int, default=666, help="random seed")
    parser.add_argument('--n', type=int, default=1, help="generate noisy images for n times")
    parser.add_argument('--r_ratio', type=float, default=0.4, help="noise ratio of R channel")
    parser.add_argument('--g_ratio', type=float, default=0.6, help="noise ratio of G channel")
    parser.add_argument('--b_ratio', type=float, default=0.8, help="noise ratio of B channel")
    args = parser.parse_args()
    
    ## Set random seed
    set_seed(args.seed)
    
    
    ## Read original images
    img_name_list, img_list = get_imgs(args.original_path)
    
    
    ## Generate noisy images
    output_path = os.path.join(args.output_path, str(args.n))
    if not os.path.exists(output_path):
        os.makedirs(output_path) 

    noise_ratio = [args.r_ratio, args.g_ratio, args.b_ratio]
    for n_i in range(args.n):
        print(f"process {n_i}-th round")
        for img_name, img in zip(img_name_list, img_list):
            nor_img = normalization(img)
            noisy_img = noise_mask_image(nor_img, noise_ratio)
            noisy_img_title = output_path + "/" + str(n_i) + "_" + img_name
            print(f"add noise to {noisy_img_title}", end="\r")
            save_image(noisy_img_title, noisy_img)
    print("finish")