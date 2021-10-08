import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# 伪彩色保存
def _savePalette(label, save_path):
    bin_colormap = np.random.randint(0, 255, (256, 3))
    bin_colormap[-1, :] = [0, 0, 0]
    bin_colormap = bin_colormap.astype(np.uint8)
    visualimg  = Image.fromarray(label, "P")
    palette = bin_colormap
    visualimg.putpalette(palette) 
    visualimg.save(save_path, format='PNG')


dataset_name = "AIplus"
dataset_path = "train"
img_folder = osp.join(dataset_path, "img")
lab_folder = osp.join(dataset_path, "class")
other = 8

# clases = os.listdir(imgs_folder)  # --
# for clas in clases:  # --
#     img_folder = osp.join(imgs_folder, clas)  # --
imgs_name = os.listdir(img_folder)
for img_name in tqdm(imgs_name):
    img_path = osp.join(img_folder, img_name)
    lab_path = osp.join(lab_folder, img_name.replace(".tif", ".png"))
    if (osp.exists(img_path) and osp.exists(lab_path)):
        # img = cv2.imread(img_path)
        img = np.asarray(Image.open(img_path))
        lab = cv2.imread(lab_path)
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
        label_list = np.unique(lab)
        if (len(label_list) == 1 and label_list[0] == other) or (np.sum(lab != other) < 64):
            pass
        else:
            tmp = lab.copy()
            for id, label in enumerate(label_list):
                if label != other:
                    lab[tmp == label] = id
                else:
                    if other != 255:
                        lab[tmp == other] = 255
            # 处理尺寸
            img = cv2.resize(img, (512, 512))
            lab = cv2.resize(lab, (512, 512), interpolation=cv2.INTER_NEAREST)
            name = img_path.split("\\")[-1].split(".")[0]
            img_save_path = osp.join("img", (dataset_name + "_" + name + ".jpg"))
            lab_save_path = osp.join("gt", (dataset_name + "_" + name + ".png"))
            cv2.imwrite(img_save_path, img)
            # lab = Image.fromarray(lab)
            # lab.save(lab_save_path)
            _savePalette(lab, lab_save_path)