from utils import *
from data_setup import VideoDetectionDatasetV2


# 数据集和数据加载器
input = np.load("clip_input")
labels = np.load("clip_output")
input = input.reshape((len(input), 1, 2048))
video_detection_dataset_v2 = VideoDetectionDatasetV2(input, labels)

# 划分训练集和测试集
train_size = int(0.8 * len(video_detection_dataset_v2))
test_size = len(video_detection_dataset_v2) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    video_detection_dataset_v2, [train_size, test_size]
)


# 定义模型
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

# 模型总结
summary(
    model=modelV1,
    input_size=(4, 1, 2048),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# 优化器和损失函数
optimizer = torch.optim.Adam(
    params=modelV1.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
)
loss_fn = torch.nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

# 开始训练
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
results = engine.train(
    model=modelV1,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
)

# 测试模型
path = f"{cwd}/kaggle/working/test/sora/tokyo-walk_1"
test_data = np.array(
    [
        clip_feature(Image.open(f"{path}/frame0.jpg")).cpu().detach().numpy(),
        clip_feature(Image.open(f"{path}/frame1.jpg")).cpu().detach().numpy(),
        clip_feature(Image.open(f"{path}/frame2.jpg")).cpu().detach().numpy(),
        clip_feature(Image.open(f"{path}/frame3.jpg")).cpu().detach().numpy(),
    ]
)

test_data = test_data.reshape(1, 1, 2048)
modelV1.eval()
data = modelV1(torch.tensor(test_data).to(device).to(torch.float32))
print(np.argmax(data.cpu().detach().numpy()))
