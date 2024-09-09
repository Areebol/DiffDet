from utils import *
from data_setup import *
from model import *


# 训练模型
vit_model = ViT(d_model=2048, num_heads=8, num_classes=2)
# # 模型小结
# summary(
#     model=vit_model,
#     # input_size=(4, 1, 2048),
#     # col_names=["input_size", "output_size", "num_params", "trainable"],
#     # col_width=20,
#     # row_settings=["var_names"],
# )

# 优化器
optimizer = torch.optim.Adam(
    params=vit_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()


def train_on(train_features_fake_path: str, test_features_path: str):
    info(f"[DoCoF] 训练集：{train_features_fake_path}")
    info(f"[DoCoF] 测试集：{test_features_path}")

    train_features_fake = np.load(train_features_fake_path)
    train_features_real = np.load(f"{cwd}/out/clip_feature/real/MSVD.npy")
    test_features = np.load(test_features_path)

    train_dataset_fake = VideoDetectionDatasetV3(train_features_fake, label=1)
    train_dataset_real = VideoDetectionDatasetV3(train_features_real, label=0)
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_fake, train_dataset_real]
    )
    test_label = 1 if "/fake/" in test_features_path else 0
    info(f"[DoCoF] 测试集标签：{test_label}")
    test_dataset = VideoDetectionDatasetV3(test_features, label=test_label)

    train_dataloader = dataloader(train_dataset)
    test_dataloader = dataloader(test_dataset)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = engine.train(
        model=vit_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device,
    )

    # torch.save(vit_model.state_dict(), f"{cwd}/models/vit.pth")


def __archive():
    # 加载数据集特征
    video_detection_dataset = load_features(f"{cwd}/clip_input", f"{cwd}/clip_output")

    # 划分数据集
    train_dataset, test_dataset = split_dataset(video_detection_dataset)

    # 模型小结
    summary(
        model=vit_model,
        # input_size=(4, 1, 2048),
        # col_names=["input_size", "output_size", "num_params", "trainable"],
        # col_width=20,
        # row_settings=["var_names"],
    )

    # 数据加载器
    train_dataloader, test_dataloader = dataloader(train_dataset, test_dataset)

    # 开始训练
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = engine.train(
        model=vit_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device,
    )

    # 保存模型
    torch.save(vit_model.state_dict(), f"{cwd}/models/vit.pth")


if __name__ == "__main__":
    """
    输入：
        out/clip_feature/fake/Sarfraz_sora.npy（训练集）
        out/clip_feature/fake/Sarfraz_text2video-zero.npy（测试集）

    中间输出：
        训练集数据加载器
        测试集数据加载器
    """

    feature_dir = f"{cwd}/out/clip_feature"
    train_features = [
        f"{feature_dir}/fake/Sarfraz_text2video-zero.npy",
        f"{feature_dir}/fake/Sarfraz_zeroscope.npy",
    ]
    test_features = [
        f"{feature_dir}/fake/Sarfraz_text2video-zero.npy",
        f"{feature_dir}/fake/Sarfraz_zeroscope.npy",
        f"{feature_dir}/fake/Sarfraz_sora.npy",
        f"{feature_dir}/real/MSVD.npy",
    ]

    for train_feature in train_features:
        for test_feature in test_features:
            if train_feature != test_feature:
                train_on(train_feature, test_feature)
