#!/usr/bin/env python

import opendatasets as od
import pandas


# od.download(
#     "https://www.kaggle.com/datasets/mohammadsarfrazalam/realfake-video-dataset"
# )


# In[18]:


# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from torchinfo import summary

from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves


# In[19]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[20]:


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn


# In[21]:


input_shape = (224, 224)


# ## Building the DataFrame

# In[ ]:


import pandas as pd
from glob import glob
import numpy as np


# In[ ]:


real_paths = glob("/content/realfake-video-dataset/real/*/*")
fake_paths = glob("/content/realfake-video-dataset/fake/*/*")


# In[ ]:


len(fake_paths), len(real_paths)


# In[ ]:


paths = np.concatenate((real_paths, fake_paths))
fake = np.concatenate((np.zeros(len(real_paths)), np.ones(len(fake_paths))))


# In[ ]:


import pandas as pd
from glob import glob
import numpy as np

import torch
import torchvision
from torchvision import transforms

from torch import nn


# ## Building the Dataset

# In[22]:


import os


# In[23]:


from torch.utils.data import Dataset


# In[24]:


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


# In[25]:


video_detection_dataset_v1 = VideoDetectionDatasetV1(paths, fake)


# In[26]:


video_detection_dataset_v1[0]


# In[ ]:


get_ipython().system("pip install git+https://github.com/openai/CLIP.git")


# In[ ]:


import clip

model, preprocess = clip.load("ViT-B/32", device=device)


# # Setting up ClipVIT

# In[ ]:


from PIL import Image


# In[ ]:


image = (
    preprocess(Image.open("/content/realfake-video-dataset/real/MSVD/0_0/frame0.jpg"))
    .unsqueeze(0)
    .to(device)
)


# In[ ]:


with torch.no_grad():
    image_features = model.encode_image(image)


# In[ ]:


from tqdm import tqdm


# In[ ]:


resize_transform = transforms.Resize(input_shape)


# In[ ]:


def getFeatures(path, transform=resize_transform):
    img = Image.open(path)
    transformed_img = img
    if transform is not None:
        transformed_img = transform(img)

    image = preprocess(transformed_img).unsqueeze(0).to(device)
    image_features = model.encode_image(image)

    return image_features


# In[ ]:


getFeatures(
    "/content/realfake-video-dataset/real/MSVD/0_0/frame0.jpg", resize_transform
).shape


# In[ ]:


features_data = []
features_label = []

for i in tqdm(range(len(video_detection_dataset_v1))):
    sample = video_detection_dataset_v1[i]
    feature_set = []
    for i in range(0, 4):
        frame = os.path.join(sample["path"], f"frame{i}.jpg")
        frame_featureset = getFeatures(frame, resize_transform)
        feature_set.append(frame_featureset.detach().numpy())

    if sample["fake"] == 0.0:
        features_label.append(0)
    else:
        features_label.append(1)

    features_data.append(np.array(feature_set))
    pass


# In[ ]:


features_label


# In[ ]:


len(features_data), len(video_detection_dataset_v1)


# In[ ]:


features_data[180].shape


# In[ ]:


features_data[80]


# In[ ]:


np.array(features_data)


# In[ ]:


file = open("clip_input", "wb")

np.save(file, features_data)


# In[ ]:


file = open("clip_output", "wb")

np.save(file, features_label)


# In[ ]:


features_data[0].shape


# # Building the model

# ## Get Input and Labels

# In[ ]:


# In[ ]:


input = np.load("clip_input")
labels = np.load("clip_output")


# In[ ]:


input.shape


# In[ ]:


# In[ ]:


input = input.reshape((len(input), 1, 2048))
input.shape


# In[ ]:


labels.shape


# ## Building the Dataset and Dataloader

# In[ ]:


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


# In[ ]:


video_detection_dataset_v2 = VideoDetectionDatasetV2(input, labels)


# In[ ]:


from torch.utils.data import DataLoader


# In[ ]:


len(video_detection_dataset_v2)


# In[ ]:


train_size = int(0.8 * len(video_detection_dataset_v2))
test_size = len(video_detection_dataset_v2) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    video_detection_dataset_v2, [train_size, test_size]
)


# In[ ]:


len(train_dataset)


# In[ ]:


# In[ ]:


encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(4, 1, 512).flatten().reshape((1, 1, 2048))
out = transformer_encoder(src)

out.shape


# In[ ]:


src.flatten().shape


# In[ ]:


classifier = nn.Sequential(
    nn.LayerNorm(normalized_shape=2048), nn.Linear(in_features=2048, out_features=2)
)


# In[ ]:


out[:, 0].shape


# In[ ]:


classifier(out)


# In[ ]:


# In[ ]:


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


# In[ ]:


modelV1 = ViT(d_model=2048, num_heads=8, num_classes=2)


# In[ ]:


summary(
    model=modelV1,
    input_size=(4, 1, 2048),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)


# In[ ]:


optimizer = torch.optim.Adam(
    params=modelV1.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
)

loss_fn = torch.nn.CrossEntropyLoss()


# In[ ]:


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)


# In[ ]:


# In[ ]:


for batch, (x, y) in enumerate(train_dataloader):
    print(batch, (x.shape, y.shape))


# In[ ]:


# In[ ]:


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

# In[ ]:


get_ipython().system("rm -rf test")


# In[ ]:


get_ipython().system("unzip test.zip")


# In[ ]:


get_ipython().system("ls")


# In[ ]:


get_ipython().system("rm -rf test.zip")


# In[ ]:


path = "/content/kaggle/working/test/sora/tokyo-walk_1"


# In[ ]:


test_data = np.array(
    [
        getFeatures(f"{path}/frame0.jpg").cpu().detach().numpy(),
        getFeatures(f"{path}/frame1.jpg").cpu().detach().numpy(),
        getFeatures(f"{path}/frame2.jpg").cpu().detach().numpy(),
        getFeatures(f"{path}/frame3.jpg").cpu().detach().numpy(),
    ]
)


# In[ ]:


test_data.shape


# In[ ]:


test_data = test_data.reshape(1, 1, 2048)


# In[ ]:


modelV1.eval()

data = modelV1(torch.tensor(test_data).to(device).to(torch.float32))


# In[ ]:


data


# In[ ]:


np.argmax(data.cpu().detach().numpy())


# In[ ]:
