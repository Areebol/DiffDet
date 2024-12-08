from data_setup import *

from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 读取图像
img = Image.open("tmp/adm.png")

img: torch.Tensor = diffusion_preprocess(img)

print(tensor_detail(img))

dnf_ = dnf_feature(img)

print(tensor_detail(dnf_))

# 将 tensor 转换为图像并保存
save_image(dnf_.squeeze(0), "out/dnf_feature_test.png")
