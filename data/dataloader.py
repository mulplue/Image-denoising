# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
from os import path
import random
import numpy as np
from PIL import Image
import time
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

np.set_printoptions(suppress=True, precision=4, linewidth=65535)
import matplotlib.pyplot as plt

# import sys
# sys.path.append('.\\..\\')
# print(sys.path)
# from utils.add_noise import noise_mask_image # 不知道为啥报错

def noise_mask_image(img, noise_ratio=[0.8,0.4,0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    noise_img = np.zeros(img.shape, np.double)
    for k in range(3):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if random.random() < noise_ratio[k]:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点
                    noise_img[i][j][k] = 0  # 1 / 0
                elif random.random() > 1-noise_ratio[k]:
                    noise_img[i][j][k] = 255  # 1 / 0
                else:
                    noise_img[i][j][k] = img[i][j][k]
    
    # -----------------------------------------------

    return noise_img

def get_filelist(path):
    files = glob.glob(path + '/*.png')
    # print(files)
    file_names = []
    for file in files:
        # print(file)
        file_name = file.split('/')[-1][-8:-4]
        cv2.waitKey(0)

        file_names.append(file_name)
    file_names.sort()
    return file_names


class ImageDataset(Dataset):
    def __init__(self, dataset_path, eval_mode=False, size=(128, 256)):
        self.eval_mode = eval_mode
        self.size = size

        img_transforms = [
            transforms.Resize(self.size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.img_transforms = transforms.Compose(img_transforms)

        self.dataset_path = dataset_path
        self.filenames = get_filelist(self.dataset_path)

    def __getitem__(self, index):
        filename = self.filenames[index]

        path = self.dataset_path+'/'+filename+'.png'
        raw_image = cv2.imread(path)
        noisy_image = noise_mask_image(raw_image)

        # mirror the inputs
        mirror = True if random.uniform(0.0, 1.0) > 0.5 else False
        if mirror:
            raw_image = raw_image[:, ::-1, :]
            noisy_image = noisy_image[:, ::-1, :]

        raw_image = Image.fromarray(raw_image.astype('uint8')).convert('RGB')
        noisy_image = Image.fromarray(noisy_image.astype('uint8')).convert('RGB')

        label = self.img_transforms(raw_image)
        image = self.img_transforms(noisy_image)

        if not self.eval_mode:
            return {'image': image, 'label': label}
        else:
            return {'image': image, 'label': label, 'filename': filename, 'index': index, 'mirror': mirror}

    def __len__(self):
        return len(self.filenames)
    
if __name__ == '__main__':
    test_loader = DataLoader(ImageDataset(dataset_path='E:/Study/Technology/AI/github/Image-denoising/dataset/CBSD68/original_png',eval_mode=True), batch_size=1, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)
    # 'E:/Study/Technology/AI/github/Image-denoising/dataset/CBSD68/original_png'

    for i, batch in enumerate(test_loader):
        image = batch['image'][0]
        label = batch['label'][0]

        print("filename:", batch['filename'][0])
        print("mirror:", batch['mirror'][0])


        cv2.imshow("Image", image.permute(1, 2, 0).numpy())
        cv2.imshow("Label", label.permute(1, 2, 0).numpy())
        cv2.waitKey(0)

    cv2.destroyAllWindows()

