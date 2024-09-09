import os, sys
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchinfo import summary
import clip
from PIL import Image
import cv2
import time
import multiprocessing

from going_modular.going_modular import data_setup, engine

import logging
import datetime

cwd = Path(__file__).parent

# DEBUG = True

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG if locals().get("DEBUG") else logging.INFO)

file_handler = logging.FileHandler(f"{cwd}/logs/{datetime.datetime.now()}.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

info = logging.info
debug = logging.debug
error = logging.error

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = None, None

# 图像输入尺寸
input_shape = (224, 224)
batch_size = 64
# num_workers = 4
num_workers = 0

seed = 42


def makeFrames(video_path, parent_path, count=1):
    vidcap = cv2.VideoCapture(str(video_path))

    success, image = vidcap.read()

    # count 为 1 就是连续读取 4 帧
    # count 为 2 就是连续读取 4 帧（_0），再连续读取 4 帧（_1）
    for itr in range(0, count):
        path = Path(f"{str(parent_path)}_{itr}")

        if os.path.isdir(path) == False:
            os.makedirs(path)

        for frame in range(0, 4):
            frame_path = Path(path, f"frame{frame}.jpg")
            cv2.imwrite(str(frame_path), image)
            success, image = vidcap.read()
