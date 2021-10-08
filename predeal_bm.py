import os
import os.path as osp
import cv2
import numpy as np
import operator
from PIL import Image
from math import ceil
from tqdm import tqdm
from osgeo import gdal
from functools import reduce


def _sampleNorm(image, NUMS=65536):
    stretched_r = _stretch(image[:, :, 0], NUMS)
    stretched_g = _stretch(image[:, :, 1], NUMS)
    stretched_b = _stretch(image[:, :, 2], NUMS)
    stretched_img = cv2.merge([
        stretched_r / float(NUMS), 
        stretched_g / float(NUMS), 
        stretched_b / float(NUMS)])
    return np.uint8(stretched_img * 255)


# 计算直方图
def _histogram(ima, NUMS):
    bins = list(range(0, NUMS))
    flat = ima.flat
    n = np.searchsorted(np.sort(flat), bins)
    n = np.concatenate([n, [len(flat)]])
    hist = n[1:] - n[:-1]
    return hist


# 直方图均衡化
def _stretch(ima, NUMS):
    hist = _histogram(ima, NUMS)
    lut = []
    for bt in range(0, len(hist), NUMS):
        # 步长尺寸
        step = reduce(operator.add, hist[bt: bt + NUMS]) / (NUMS - 1)
        # 创建均衡的查找表
        n = 0
        for i in range(NUMS):
            lut.append(n / step)
            n += hist[i + bt]
        np.take(lut, ima, out=ima)
        return ima


# 裁剪
def _createGrids(img, grid_size=512):
    h, w = img.shape[:2]
    grid_row_count = ceil(h / grid_size)
    grid_col_count = ceil(w / grid_size)
    imagesGrid = _slideOut(img, grid_row_count, grid_col_count)
    return imagesGrid


def _slideOut(img, row, col, overlap=0):
    H, W = img.shape[:2]
    shape_len = len(img.shape)
    c_size = [ceil(H / row), ceil(W / col)]
    # 扩展不够的以及重叠部分
    h_new = row * c_size[0] + overlap
    w_new = col * c_size[1] + overlap
    # 新图
    if shape_len == 3:
        tmp = np.zeros((h_new, w_new, img.shape[-1]), dtype="uint8")
        tmp[:img.shape[0], :img.shape[1], :] = img
    else:
        tmp = np.zeros((h_new, w_new), dtype="uint8")
        tmp[:img.shape[0], :img.shape[1]] = img
    H, W = tmp.shape[:2]
    cell_h = c_size[0]
    cell_w = c_size[1]
    # 开始分块
    result = []
    if shape_len == 3:
        for i in range(row):
            for j in range(col):
                start_h = i * cell_h
                end_h = start_h + cell_h + overlap
                start_w = j * cell_w
                end_w = start_w + cell_w + overlap
                result.append(tmp[start_h : end_h, start_w : end_w, :])
    else:
        for i in range(row):
            for j in range(col):
                start_h = i * cell_h
                end_h = start_h + cell_h + overlap
                start_w = j * cell_w
                end_w = start_w + cell_w + overlap
                result.append(tmp[start_h : end_h, start_w : end_w])
    return result


# 伪彩色保存
def _savePalette(label, save_path):
    bin_colormap = np.random.randint(0, 255, (256, 3))
    bin_colormap[-1, :] = [0, 0, 0]
    bin_colormap = bin_colormap.astype(np.uint8)
    visualimg  = Image.fromarray(label, "P")
    palette = bin_colormap
    visualimg.putpalette(palette) 
    visualimg.save(save_path, format="PNG")


dataset_name = "GID5"
dataset_path = "GID Large-scale Classification_5classes"
img_folder = osp.join(dataset_path, "img")
lab_folder = osp.join(dataset_path, "class")
others = [0]

imgs_name = os.listdir(img_folder)
for img_name in tqdm(imgs_name):
    img_path = osp.join(img_folder, img_name)
    lab_path = osp.join(lab_folder, img_name.replace(".tif", "_label.tif"))
    if (osp.exists(img_path) and osp.exists(lab_path)):
        # img = cv2.imread(img_path).astype("uint8")
        # lab = cv2.imread(lab_path).astype("uint8")
        img = gdal.Open(img_path).ReadAsArray().transpose((1, 2, 0))[:, :, :3]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = _sampleNorm(img)
        lab = gdal.Open(lab_path).ReadAsArray().astype("uint8").transpose((1, 2, 0))
        # 处理标签
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        # -- test --
        # print(img.shape, lab.shape)
        # print(np.unique(lab))
        # cv2.imshow("lab", lab)
        # cv2.waitKey(0)
        # break
        # -- ---- --
        imgs = _createGrids(img)
        labs = _createGrids(lab)
        for idx, (g_img, g_lab) in enumerate(zip(imgs, labs)):
            label_list = np.unique(g_lab)
            tmp = g_lab.copy()
            # 背景过大
            sumer = 0
            for oth in others:
                sumer += np.sum(g_lab[tmp != oth])
            if (len(label_list) == 1 and label_list[0] in others) or (sumer < 64):
                pass
            else:
                for id, label in enumerate(label_list):
                    if label not in others:
                        g_lab[tmp == label] = id
                    else:
                        for oth in others:
                            g_lab[tmp == oth] = 255
                # 处理尺寸
                g_img = cv2.resize(g_img, (512, 512))
                g_lab = cv2.resize(g_lab, (512, 512), interpolation=cv2.INTER_NEAREST)
                name = img_path.split("\\")[-1].split(".")[0]
                img_save_path = osp.join("img", (dataset_name + "_G" + str(idx) + "_" + 
                                                name + ".jpg"))
                lab_save_path = osp.join("gt", (dataset_name + "_G" + str(idx) + "_" + 
                                                name + ".png"))
                cv2.imwrite(img_save_path, g_img)
                # g_lab = Image.fromarray(g_lab)
                # g_lab.save(lab_save_path)
                _savePalette(g_lab, lab_save_path)