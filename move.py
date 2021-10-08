import os
import os.path as osp
import shutil
from tqdm import tqdm


rm_txt = "rm.txt"
datas = []
with open(rm_txt, "r") as f:
    datas = f.readlines()
for data in tqdm(datas):
    path = data.strip()
    save_path = path.replace("rsai", "rsai2")
    if osp.exists(path):
        shutil.copy(path, save_path)
        os.remove(path)


# imgs_folder = r"rsai2\img"
# labs_folder = r"rsai2\gt"
# labs_name = os.listdir(labs_folder)
# for lab_name in tqdm(labs_name):
#     name = lab_name.split(".")[0]
#     img_path = osp.join(imgs_folder, (name + ".jpg"))
#     save_path = img_path.replace("img", "img2")
#     if osp.exists(img_path):
#         shutil.copy(img_path, save_path)