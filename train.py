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
        VideoFeatureDataset(dataset), list(range(num_samples))
    )
    dataset_real = SubsetVideoFeatureDataset(
        VideoFeatureDataset("MSR-VTT"), list(range(num_samples))
    )
    # 合并数据集
    dataset = ConcatDataset([dataset_fake, dataset_real])

    # 分割数据集
    train_dataset, val_dataset = split_dataset(dataset)

    # 训练模型
    model = _train(train_dataset, val_dataset)

    # 保存模型
    torch.save(model.state_dict(), f"{cwd}/models/{dataset}.pth")


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
