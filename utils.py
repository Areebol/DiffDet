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
import random
import multiprocessing
import json
import logging
import datetime
from typing import Optional, List, Tuple


DEBUG = True


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cwd = Path(__file__).parent


# 优化 GPU 运算速度
"""当设置为 True 时，cuDNN 会自动寻找最适合当前硬件的卷积算法。
系统会先花费一些时间找到最优算法，然后在接下来的运行中一直使用这个最优算法。
第一次运行时会略微变慢（因为要寻找最优算法），之后的运行会明显加速，并会占用更多的显存。"""
torch.backends.cudnn.benchmark = True


# 日志
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG if locals().get("DEBUG") else logging.INFO)

file_handler = logging.FileHandler(f"{cwd}/logs/{datetime.datetime.now()}.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

info = logging.info
debug = logging.debug
error = logging.error


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # 设置 PyTorch 的 CPU 随机种子
    if torch.cuda.is_available():  # 如果 GPU 可用，设置 PyTorch 的 GPU 随机种子
        torch.cuda.manual_seed_all(random_seed)


def tensor_detail(x: torch.Tensor):
    return (
        f"{x.shape} [{x.device}] ({x.min():.3f} {x.mean():.3f} {x.max():.3f}) {x.dtype}"
    )


class Timer:
    """简易计时器，用于测量代码执行时间"""

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def tick(self):
        """返回从创建计时器到现在经过的秒数"""
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        return f"{elapsed:.2f} s"

    def total(self):
        """返回从创建计时器到现在的总用时"""
        return f"{time.time() - self.start_time:.2f} s"
