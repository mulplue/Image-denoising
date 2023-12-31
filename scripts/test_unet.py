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
parser.add_argument('--model', type=str, default='F1')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResRemover().to(device)
model.load_state_dict(torch.load("./trained_model/"+args.model+".pth"))



def restore_image(noise_img, size=4):
    img = np.copy(noise_img)
    # print("x")
    # print(img)
    # print(img.shape)


    # cv2.imshow(img)


    img = torch.from_numpy(img).permute(2,0,1).to(device)
    img = img.unsqueeze(0)
    img = transforms.Resize((256,512))(img)
    img = img.float()

    img = model(img)
    img = img.squeeze()

    img = img.permute(1,2,0)

    img = img.cpu().detach().numpy()
    # img = img/np.max(img)
    img = np.maximum(img,0)
    img = np.minimum(img,1)
    # print("y")
    # print(img)
    # cv2.imshow(img)

    # cv2.waitkey(0)

    img = cv2.resize(img,dsize=(noise_img.shape[1],noise_img.shape[0]))
    # print(img.shape)

    # ---------------------------------------------------------------

    return img

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


def read_img(path):
    img = Image.open(path)
    # img = img.resize((150,150))
    img = np.asarray(img, dtype="uint8")
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(img.dtype)
    # 图像数组数据放缩在 0-1 之间
    return img.astype(np.double) / info.max

files = get_filelist(r"./dataset/test/input")


for file in files:
    print(file)
    source = read_img("./dataset/test/input/"+file+".png")
    # target = read_image("./dataset/test/target/"+file+".png")
    result = restore_image(source)

    # cv2.imshow("source",source)
    # cv2.imshow("result",result)
    # cv2.waitKey(0)


    cv2.cvtColor(result, cv2.COLOR_RGB2BGR, result)
    os.makedirs("./dataset/test/restore_hejun"+args.model, exist_ok=True)
    cv2.imwrite("./dataset/test/restore_hejun"+args.model+"/"+file+".png", result*255)

