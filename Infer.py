from tqdm.auto import tqdm

from skimage import io, segmentation, morphology, exposure
import tifffile as tif
import json
import shutil
import numpy as np
import os
import cv2
import torch
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from pycocotools import mask as mutils
from mmdet.apis import init_detector, inference_detector, single_gpu_test
from mmcv import Config
import traceback
from mmdet.utils import AvoidCUDAOOM
config = Config.fromfile(f'config_780-1100.py')
ckpt_path = 'weights/epoch_30.pth'

import os
import sys

model = init_detector(config, ckpt_path, device='cuda')

ROOT_FOLDER = '/app/'
INPUT_FOLDER = '/workspace/inputs/'

SAVED_FOLDER = f'/workspace/outputs' # folder in drive to save prediction zipped file
VERSION = 'CBNetV2_mask_rcnn'
MAX_DETS = 1500
NOTE = f'_maxdet{MAX_DETS}_sliding'


# Predict on the whole tuning set
MIN_SIDE_FOR_SLIDING = 4000
MIN_REQUIRED_INST_NUM = 5
import time
import os
import pandas as pd
import numpy as np

class TimeLogger(object):
    def __init__(self, model_name, save_path, image_name):
        self.model_name = model_name
        self.save_path = save_path
        self.image_name = image_name
        self.data: pd.DataFrame
        if os.path.exists(save_path):
            self.data = pd.read_csv(save_path)
        else:
            self.data = pd.DataFrame({'image_name': [], 'time': []})
        self.flag = False

    def setDNF(self):
        self.flag = True

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        new_record = pd.DataFrame({'image_name':[ self.image_name], 'time': [self.end - self.start if not self.flag else -1]})
        self.data = self.data.append(new_record)
        self.data = self.data.sort_values('image_name')
        self.data.to_csv(self.save_path, index= False)

def read_image(img_path):
    if img_path.endswith('.tif') or img_path.endswith('.tiff'):
        img_data = tif.imread(img_path)
    else:
        img_data = io.imread(img_path)
    return img_data

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def process_image(img_data):
    # normalize image data
    if len(img_data.shape) == 2:
        img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        img_data = img_data[:,:, :3]
    else:
        pass
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:,:,i]
        if len(img_channel_i[np.nonzero(img_channel_i)])>0:
            pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
    return pre_img_data


def sliding_window_prediction(im, window_size = 1024):
    H, W = im.shape[:2]
    n_rows = int(np.ceil(H / window_size))
    n_cols = int(np.ceil(W / window_size))

    pred_instance_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)

    num_existed_inst = 0
    for i in tqdm(range(n_cols)):
        for j in range(n_rows):
            start_x, end_x = window_size*i, np.minimum(window_size*(i+1), W)
            start_y, end_y = window_size*j, np.minimum(window_size*(j+1), H)
            patch = im[start_y:end_y, start_x:end_x]

            outputs = inference_detector(model, patch)

            # for num, mask in enumerate(outputs[1][0]):

            #   ys, xs = np.where(mask==1)
            #   ys += start_y
            #   xs += start_x
            #   pred_instance_mask[ys, xs] = inst_id
            #   inst_id += 1
            output = outputs[1]
            pred_instance_mask[start_y:end_y, start_x:end_x] = np.where(output > 0, output + num_existed_inst, 0).reshape(output.shape)
            num_existed_inst += outputs[1].max()


    return pred_instance_mask

def get_patch_size(size):
    if size >= 2000 and size < 3000:
        return 256
    if size < 4000:
        return 512
    if size < 15000:
        return 1024
    if size >= 15000:
        return 2048
    return 1024
import cv2

def shortest_edge_resize(img, shortest_edge_length, max_size):
    h,w  = img.shape
    size = shortest_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale= max_size * 1.0  / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int (neww + 0.5)
    newh = int (newh + 0.5)
    return cv2.resize(img, (neww, newh))

# Predict tuning
os.makedirs(SAVED_FOLDER, exist_ok=True)
for fname in tqdm(sorted(os.listdir(INPUT_FOLDER))):

# for fname in sorted(os.listdir(TUNING_SET_DIR))[:4]:
        img_path = os.path.join(INPUT_FOLDER, fname)
        # print(img_path)
        im = read_image(img_path)
        im = process_image(im)

        # # convert to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        shortest_edge = np.min(im.shape[:2])
        torch.cuda.empty_cache()

        outputs = inference_detector(model, im)
        torch.cuda.empty_cache()
        # print(outputs)
        # outputs = predictor(im)

        if type(outputs[1]) == list or outputs[1].max() <= MIN_REQUIRED_INST_NUM or shortest_edge > MIN_SIDE_FOR_SLIDING:
            patch_size = get_patch_size(shortest_edge)
            # print('Image', fname, 'has predicted inst num =', 0 if type(outputs[1]) == list else outputs[1].max(),
            #     'And size =', shortest_edge,
            #     '. Use sliding window infer with patch size:', patch_size)
            pred_instance_mask = sliding_window_prediction(im, patch_size)
            # print('After using sliding window, Image', fname, 'has predicted inst num =', pred_instance_mask.max())
        else:
            # pred_instance_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)
            # for i, mask in enumerate(outputs[1][0]):
            #     inst_id = i+1
            #     pred_instance_mask[mask] = inst_id
            pred_instance_mask = outputs[1]
        # if not len(np.unique(pred_instance_mask)) > 5:
        #     print(fname)
        torch.cuda.empty_cache()
        output_path = os.path.join(SAVED_FOLDER, fname.split('.')[0] +'_label.tiff')
        tif.imwrite(output_path, pred_instance_mask, compression='zlib')