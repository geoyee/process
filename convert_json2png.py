import os
import os.path as osp
import shutil

import cv2
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict


def _mkdir_p(path):
    if not osp.exists(path):
        os.makedirs(path)


def _get_mask(annotation, image):
    if isinstance(image, str):
        image = cv2.imread(image)
    mask = np.zeros(image.shape[:2], dtype=np.int32)
    for contour_points in annotation:
        contour_points = np.array(contour_points).reshape((-1, 2))
        contour_points = np.round(contour_points).astype(np.int32)[np.newaxis, :]
        cv2.fillPoly(mask, contour_points, 255)
    return mask


def _read_geojson(json_path):
    with open(json_path, 'r') as f:
        jsoner = json.load(f)
        imgs = jsoner['images']
        images = defaultdict(list)
        for img in imgs:
            images[img['id']] = img['file_name']
        anns = jsoner['annotations']
        annotations = defaultdict(list)
        for ann in anns:
            annotations[images[ann['image_id']]].append(ann['segmentation'])
        return annotations


def convertData(raw_folder, end_folder):
    # 初始化
    print('-- 开始初始化 --')
    img_folder = osp.join(raw_folder, 'images')
    save_img_folder = osp.join(end_folder, 'img')
    save_lab_folder = osp.join(end_folder, 'gt')
    _mkdir_p(save_img_folder)
    _mkdir_p(save_lab_folder)
    # 标签
    print('-- 开始加载标签 --')
    json_file_1 = osp.join(raw_folder, 'annotation.json')
    ann_1 = _read_geojson(json_file_1)
    json_file_2 = osp.join(raw_folder, 'annotation-small.json')
    ann_2 = _read_geojson(json_file_2)
    anns = {**ann_1, **ann_2}  # 合并字典
    print('-- 开始转换数据 --')
    for k in tqdm(anns.keys()):
        img_path = osp.join(img_folder, k)
        ann = anns[k]
        mask = _get_mask(ann, img_path)
        img_save_path = osp.join(save_img_folder, k)
        lab_save_path = osp.join(save_lab_folder, k.replace(".jpg", ".png"))
        shutil.copy(img_path, img_save_path)
        cv2.imwrite(lab_save_path, mask)


if __name__ == '__main__':
    raw_folder = r'datas\raw'
    end_folder = r'datas\end'
    convertData(raw_folder, end_folder)