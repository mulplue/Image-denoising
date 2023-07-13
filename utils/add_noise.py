import random
import cv2
import numpy as np

# def noise_mask_image(img, noise_ratio=[0.8,0.4,0.6]):
#     """
#     根据题目要求生成受损图片
#     :param img: cv2 读取图片,而且通道数顺序为 RGB
#     :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
#                         默认值分别是 [0.8,0.4,0.6]
#     :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
#              数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
#     """
#     # 受损图片初始化
#     noise_img = None
#     # -------------实现受损图像答题区域-----------------
#     noise_img = np.zeros(img.shape, np.double)
#     for k in range(3):
#         for i in range(img.shape[0]):
#             for j in range(img.shape[1]):
#                 if random.random() < noise_ratio[k]:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点
#                     noise_img[i][j][k] = 0  # 1 / 0
#                 else:
#                     noise_img[i][j][k] = img[i][j][k]
    
#     # -----------------------------------------------

#     return noise_img


def get_noisy_mask(ratio, shape):
    mask = []
    H, W = shape[0], shape[1]
    for row in range(H):
        line = np.ones(W)
        line[:int(W*ratio)] = 0
        np.random.shuffle(line)
        mask.append(line)
    return np.array(mask)


def noise_mask_image(img, noise_ratio=[0.4,0.6,0.8]):
    """
    根据题目要求生成受损图片
    :param img: cv2.imread(), the order is GBR
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    img_shape = img.shape
    r_ratio, g_ratio, b_ratio = noise_ratio[0], noise_ratio[1], noise_ratio[2]
    r_mask = get_noisy_mask(r_ratio, img_shape)
    g_mask = get_noisy_mask(g_ratio, img_shape)
    b_mask = get_noisy_mask(b_ratio, img_shape)

    r_channel = img[:, :, 0] * r_mask
    g_channel = img[:, :, 1] * g_mask
    b_channel = img[:, :, 2] * b_mask
    
    noise_img = cv2.merge([r_channel, g_channel, b_channel])
    # -----------------------------------------------

    return noise_img
