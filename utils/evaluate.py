import numpy as np
from skimage import measure
from scipy import spatial

def calc_ssim(img, img_restore):
    """
    计算图片的结构相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    return measure.compare_ssim(img, img_restore,
                multichannel=True,
                data_range=img_restore.max() - img_restore.min())

def calc_csim(img, img_restore):
    """
    计算图片的 cos 相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    img = img.reshape(-1)
    img_restore = img_restore.reshape(-1)
    return 1 - spatial.distance.cosine(img, img_restore)

def calc_l2(img, img_restore):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像 
    :param img:原始图像 
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0
    
    # 将图像矩阵转换成为np.narray
    img = np.array(img)
    img_restore = np.array(img_restore)
    
    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if img.shape != img_restore.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (img.shape, img_restore.shape))
        return None
    
    total_pixels = img.shape[0] * img.shape[1] * img.shape[2]
    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(img - img_restore, 2))) / total_pixels
    
    return error