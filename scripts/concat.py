import copy
import torch
import os
import sys
import argparse
# from os.path import join, dirname
sys.path.append(r'/home/wanghejun/Desktop/wanghejun/Image-Denoising/github/Image-denoising/')
# print(sys.path)
# import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../../../'))


from models.unet import ForwardRemover,ResRemover
import torchvision.transforms as transforms
from PIL import Image
import glob
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--png', type=str, default='A')
args = parser.parse_args()

def get_filelist(path):
    files = glob.glob(path + '/*.png')
    # print(files)
    file_names = []
    for file in files:
        # print(file)
        file_name = file.split('/')[-1]
        file_name = file_name.split('\\')[-1][:-4]

        file_names.append(file_name)
    file_names.sort()
    return file_names

if __name__ == '__main__':

    files = get_filelist(r"./dataset/test/input")

    for png in files:
        noise = cv2.imread("./dataset/test/input/"+png+".png")
        target = cv2.imread("./dataset/test/target/"+png+".png")

        F1 = cv2.imread("./dataset/test/restore_hejunF1/"+png+".png")
        F2 = cv2.imread("./dataset/test/restore_hejunF2/"+png+".png")
        R1 = cv2.imread("./dataset/test/restore_hejunR1/"+png+".png")
        R2 = cv2.imread("./dataset/test/restore_hejunR2/"+png+".png")

        restormer = cv2.imread("./dataset/test/restore_jiahe/"+png+".png")
        # print(noise.shape)

        concated = np.concatenate((noise, F1, F2, R1, R2, restormer, target), axis=1)

        # cv2.imshow("concated", concated)
        os.makedirs("./dataset/test/restore_concat/", exist_ok=True)

        cv2.imwrite("./dataset/test/restore_concat/"+png+".png", concated)
        print(png)

    # cv2.waitKey(0)



