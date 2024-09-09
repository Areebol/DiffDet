from utils import *
from data_setup import *
from model import *

# # 模型小结
# summary(
#     model=ViT(d_model=4096, num_heads=8, num_classes=2),
#     # input_size=(4, 1, 2048),
#     # col_names=["input_size", "output_size", "num_params", "trainable"],
#     # col_width=20,
#     # row_settings=["var_names"],
# )


def _train(train_dataset: Dataset, test_dataset: Dataset):
    # 初始化模型
    vit_model = ViT(d_model=4096, num_heads=8, num_classes=2)
    vit_model = vit_model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        params=vit_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
    )

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    engine.train(
        model=vit_model,
        train_dataloader=dataloader(train_dataset),
        test_dataloader=dataloader(test_dataset),
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device,
    )

    return vit_model


def train(dataset: str, num_samples: int = 1000):
    info(f"[DoCoF] 训练集：{dataset}")

    dataset_fake = SubsetVideoFeatureDataset(
        VideoFeatureDataset(dataset_paths[dataset]), list(range(num_samples))
    )
    dataset_real = SubsetVideoFeatureDataset(
        VideoFeatureDataset(dataset_paths["MSR-VTT"]), list(range(num_samples))
    )
    # 合并数据集
    dataset = torch.utils.data.ConcatDataset([dataset_fake, dataset_real])

    # 分割数据集
    train_dataset, val_dataset = split_dataset(dataset)

    model = _train(train_dataset, val_dataset)

    # 保存模型
    torch.save(model.state_dict(), f"{cwd}/models/{dataset}.pth")


# def train_on(train_features_fake_path: str, test_features_path: str):
#     info(f"[DoCoF] 训练集：{train_features_fake_path}")
#     info(f"[DoCoF] 测试集：{test_features_path}")

#     train_features_fake = np.load(train_features_fake_path)
#     train_features_real = np.load(f"{cwd}/out/clip_feature/real/MSVD.npy")
#     test_features = np.load(test_features_path)

#     train_dataset_fake = VideoDetectionDatasetV3(train_features_fake, label=1)
#     train_dataset_real = VideoDetectionDatasetV3(train_features_real, label=0)
#     train_dataset = torch.utils.data.ConcatDataset(
#         [train_dataset_fake, train_dataset_real]
#     )
#     test_label = 1 if "/fake/" in test_features_path else 0
#     info(f"[DoCoF] 测试集标签：{test_label}")
#     test_dataset = VideoDetectionDatasetV3(test_features, label=test_label)

#     train_dataloader = dataloader(train_dataset)
#     test_dataloader = dataloader(test_dataset)

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     model = engine.train(
#         model=vit_model,
#         train_dataloader=train_dataloader,
#         test_dataloader=test_dataloader,
#         optimizer=optimizer,
#         loss_fn=loss_fn,
#         epochs=10,
#         device=device,
#     )

#     # torch.save(vit_model.state_dict(), f"{cwd}/models/vit.pth")


if __name__ == "__main__":
    """"""

    # 训练数据集为所有的 fake 数据集（除了 Sora）
    train_dataset = [
        dataset_name
        for dataset_name, dataset_path in dataset_paths.items()
        if "/fake/" in dataset_path and "Sora" not in dataset_name
    ]

    info(f"[DoCoF] 训练数据集：{train_dataset}")

    for train_dataset_name in train_dataset:
        train(train_dataset_name)
