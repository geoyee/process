import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


img_folder = r"trainData\img"
lab_folder = r"trainData\gt"


mps = (512 * 0.75) ** 2
imgs_name = os.listdir(img_folder)
for img_name in tqdm(imgs_name):
    name = img_name.split(".")[0]
    img_path = osp.join(img_folder, (name + ".png"))
    lab_path = osp.join(lab_folder, (name + "_label.png"))
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
            os.remove(img_path)
            os.remove(lab_path)
            continue
        # 大面积无内容
        elif sum_255 > mps or sum_0 > mps:
            os.remove(img_path)
            os.remove(lab_path)
            continue
        else:
            cv2.imwrite(img_path.replace(".png", ".jpg"), img)
            os.remove(img_path)
            os.rename(lab_path, lab_path.replace("_label", ""))
    else:
        print("不存在：", lab_path)
        os.remove(img_path)