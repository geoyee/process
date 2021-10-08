import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


img_folder = r"rsai\img"
lab_folder = r"rsai\gt"
rm_txt = "rm.txt"

mps = (512 * 0.75) ** 2
imgs_name = os.listdir(img_folder)
with open(rm_txt, "w") as f:
    for img_name in tqdm(imgs_name):
        name = img_name.split(".")[0]
        img_path = osp.join(img_folder, (name + ".jpg"))
        lab_path = osp.join(lab_folder, (name + ".png"))
        if osp.exists(img_path) and osp.exists(lab_path):
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sum_255 = np.sum(gray_img == 255)
            sum_0 = np.sum(gray_img == 0)
            # print(sum_255, sum_0)
            lab = np.asarray(Image.open(lab_path))
            lab_unique = np.unique(lab)
            # 只有一个标签
            if len(lab_unique) == 1:
                f.write(img_path + "\n" + lab_path + "\n")
                continue
            # 大面积无内容
            elif sum_255 > mps or sum_0 > mps:
                f.write(img_path + "\n" + lab_path + "\n")
                continue
        else:
            print("不存在：", lab_path)
            f.write(img_path + "\n")