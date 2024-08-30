import os
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

from going_modular.going_modular import data_setup, engine

import logging
import datetime

cwd = Path(__file__).parent

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f"{cwd}/logs/{datetime.datetime.now()}.log")
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# 打印到控制台并保存到日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("log.txt")],
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

info = logging.info

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = None, None

# 图像输入尺寸
input_shape = (224, 224)
batch_size = 64
num_workers = 32

seed = 42

# 图像预处理转换
transform = transforms.Compose(
    [
        transforms.Resize(input_shape),
    ]
)


def clip_feature(img: Image) -> torch.Tensor:
    """提取图像的 CLIP 特征"""

    # 按需加载 CLIP 模型
    global clip_model, clip_preprocess
    if clip_model is None or clip_preprocess is None:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # 图像预处理后用 CLIP 提取特征
    img = clip_preprocess(transform(img)).unsqueeze(0).to(device)
    features = clip_model.encode_image(img)
    return features  # torch.Size([1, 512])
