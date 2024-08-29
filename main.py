#!/usr/bin/env python

import opendatasets as od


# od.download(
#     "https://www.kaggle.com/datasets/mohammadsarfrazalam/realfake-video-dataset"
# )

import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from torchinfo import summary

from going_modular.going_modular import data_setup, engine
from utils import download_data, set_seeds, plot_loss_curves


device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data import Dataset


input_shape = (224, 224)


import pandas as pd
from glob import glob
import numpy as np
from pathlib import Path


cwd = Path(__file__).parent

real_paths = glob(f"{cwd}/realfake-video-dataset/real/*/*")
fake_paths = glob(f"{cwd}/realfake-video-dataset/fake/*/*")


len(fake_paths), len(real_paths)


paths = np.concatenate((real_paths, fake_paths))
fake = np.concatenate((np.zeros(len(real_paths)), np.ones(len(fake_paths))))


import pandas as pd
from glob import glob
import numpy as np

import torch
import torchvision
from torchvision import transforms

from torch import nn


import os


from torch.utils.data import Dataset


class VideoDetectionDatasetV1(Dataset):
    def __init__(self, paths, labels):
        super()
        self.paths = paths
        self.labels = labels
        pass

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames = glob(os.path.join(self.paths[idx], "*"))
        label = self.labels[idx]

        sample = {"frames": frames, "fake": label, "path": self.paths[idx]}

        return sample


video_detection_dataset_v1 = VideoDetectionDatasetV1(paths, fake)

print(video_detection_dataset_v1.paths)

video_detection_dataset_v1[0]


# pip install git+https://github.com/openai/CLIP.git


import clip

model, preprocess = clip.load("ViT-B/32", device=device)


from PIL import Image


image = (
    preprocess(Image.open(f"{cwd}/realfake-video-dataset/real/MSVD/0_0/frame0.jpg"))
    .unsqueeze(0)
    .to(device)
)


with torch.no_grad():
    image_features = model.encode_image(image)


from tqdm import tqdm


resize_transform = transforms.Resize(input_shape)


def getFeatures(path, transform=resize_transform):
    img = Image.open(path)
    transformed_img = img
    if transform is not None:
        transformed_img = transform(img)

    image = preprocess(transformed_img).unsqueeze(0).to(device)
    image_features = model.encode_image(image)

    return image_features


getFeatures(
    f"{cwd}/realfake-video-dataset/real/MSVD/0_0/frame0.jpg", resize_transform
).shape


features_data = []
features_label = []

for i in tqdm(range(len(video_detection_dataset_v1))):
    sample = video_detection_dataset_v1[i]
    feature_set = []
    for i in range(0, 4):
        frame = os.path.join(sample["path"], f"frame{i}.jpg")
        frame_featureset = getFeatures(frame, resize_transform)
        feature_set.append(frame_featureset.detach().cpu().numpy())

    if sample["fake"] == 0.0:
        features_label.append(0)
    else:
        features_label.append(1)

    features_data.append(np.array(feature_set))
    pass


features_label


len(features_data), len(video_detection_dataset_v1)


features_data[180].shape


features_data[80]


np.array(features_data)


file = open("clip_input", "wb")

np.save(file, features_data)


file = open("clip_output", "wb")

np.save(file, features_label)


features_data[0].shape


input = np.load("clip_input")
labels = np.load("clip_output")
print(f"输入的尺寸：{input.shape}")

input = input.reshape((len(input), 1, 2048))
print(f"输入的尺寸（reshape 后）：{input.shape}")


# 建数据集和数据加载器


class VideoDetectionDatasetV2(Dataset):
    def __init__(self, input, labels):
        super()
        self.inputs = input
        self.labels = labels
        pass

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.inputs[idx], self.labels[idx]


video_detection_dataset_v2 = VideoDetectionDatasetV2(input, labels)


from torch.utils.data import DataLoader


print(f"数据集的长度：{len(video_detection_dataset_v2)}")


train_size = int(0.8 * len(video_detection_dataset_v2))
test_size = len(video_detection_dataset_v2) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    video_detection_dataset_v2, [train_size, test_size]
)


print(f"训练数据集的长度：{len(train_dataset)}")
print(f"训练数据集 [0]：{train_dataset[0]}")

encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(4, 1, 512).flatten().reshape((1, 1, 2048))
out = transformer_encoder(src)

print(f"Encode 前输入的尺寸：{src.shape}")
print(f"Encode 后输出的尺寸：{out.shape}")


src.flatten().shape


classifier = nn.Sequential(
    nn.LayerNorm(normalized_shape=2048), nn.Linear(in_features=2048, out_features=2)
)


out[:, 0].shape


classifier(out)


class ViT(nn.Module):
    def __init__(
        self, d_model: int = 2048, num_heads: int = 12, num_classes: int = 1000
    ):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            nn.Linear(in_features=d_model, out_features=num_classes),
        )

    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.classifier(x[:, 0])

        return x


modelV1 = ViT(d_model=2048, num_heads=8, num_classes=2)


summary(
    model=modelV1,
    input_size=(4, 1, 2048),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)


optimizer = torch.optim.Adam(
    params=modelV1.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
)

loss_fn = torch.nn.CrossEntropyLoss()


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)


for batch, (x, y) in enumerate(train_dataloader):
    print(batch, (x.shape, y.shape))


# 开始训练

from going_modular.going_modular import engine

set_seeds()

results = engine.train(
    model=modelV1,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
)


# ## Testing Models


# get_ipython().system("rm -rf test")
# get_ipython().system("unzip test.zip")
# get_ipython().system("ls")
# get_ipython().system("rm -rf test.zip")
path = f"{cwd}/kaggle/working/test/sora/tokyo-walk_1"


test_data = np.array(
    [
        getFeatures(f"{path}/frame0.jpg").cpu().detach().numpy(),
        getFeatures(f"{path}/frame1.jpg").cpu().detach().numpy(),
        getFeatures(f"{path}/frame2.jpg").cpu().detach().numpy(),
        getFeatures(f"{path}/frame3.jpg").cpu().detach().numpy(),
    ]
)


print(f"测试数据的尺寸：{test_data.shape}")


test_data = test_data.reshape(1, 1, 2048)


modelV1.eval()

data = modelV1(torch.tensor(test_data).to(device).to(torch.float32))


data

print(np.argmax(data.cpu().detach().numpy()))
