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

import sys
sys.path.append(r'E:\Study\Technology\AI\github\Image-denoising')
# print(sys.path)
from utils.add_noise import noise_mask_image

def get_filelist(path):
    files = glob.glob(path + '/*.png')
    # print(files)
    file_names = []
    for file in files:
        # print(file)
        file_name = file.split('\\')[-1][:-4]
        cv2.waitKey(0)

        file_names.append(file_name)
    file_names.sort()
    return file_names


class ImageDataset(Dataset):
    def __init__(self, dataset_path, eval_mode=False, size=(256, 512)):
        self.eval_mode = eval_mode
        self.size = size

        img_transforms = [
            transforms.Resize(self.size, Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.img_transforms = transforms.Compose(img_transforms)

        self.dataset_path = dataset_path
        self.filenames = get_filelist(self.dataset_path)

    def __getitem__(self, index):
        if not self.eval_mode:
            filename = random.choice(self.filenames) # 训练则随机取样 防止局部震荡
        else:
            filename = self.filenames[index]

        path = self.dataset_path+'\\'+filename+'.png'
        raw_image = cv2.imread(path)

        # mirror the inputs for data augmentation
        horizontal_mirror = True if random.uniform(0.0, 1.0) > 0.5 else False
        vertical_mirror = True if random.uniform(0.0, 1.0) > 0.5 else False
        if horizontal_mirror:
            raw_image = raw_image[:, ::-1, :]
        if vertical_mirror:
            raw_image = raw_image[::-1, :, :]

        noisy_image = noise_mask_image(raw_image)

        raw_image = Image.fromarray(raw_image.astype('uint8')).convert('RGB')
        noisy_image = Image.fromarray(noisy_image.astype('uint8')).convert('RGB')

        label = self.img_transforms(raw_image)
        image = self.img_transforms(noisy_image)

        if not self.eval_mode:
            return {'image': image, 'label': label}
        else:
            return {'image': image, 'label': label, 'filename': filename, 'index': index, 'mirror': (horizontal_mirror,vertical_mirror)}

    def __len__(self):
        return len(self.filenames)
    
if __name__ == '__main__':
    test_loader = DataLoader(ImageDataset(dataset_path=r'.\dataset\CBSD68\original_png',eval_mode=True), batch_size=1, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)
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

