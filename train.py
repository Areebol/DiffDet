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


def train(dataset_name: str, feature: str = "dnf", num_samples: int = 1000):
    info(f"[DoCoF] 训练集：{dataset_name}")

    dataset_fake = SubsetVideoFeatureDataset(
        VideoFeatureDataset(dataset_name, feature), list(range(num_samples))
    )
    dataset_real = SubsetVideoFeatureDataset(
        VideoFeatureDataset("MSR-VTT", feature), list(range(num_samples))
    )
    # 合并数据集
    dataset = ConcatDataset([dataset_fake, dataset_real])

    # 分割数据集
    train_dataset, val_dataset = split_dataset(dataset)

    # 训练模型
    model = _train(train_dataset, val_dataset)

    # 保存模型
    torch.save(model.state_dict(), f"{cwd}/models/{feature}/{dataset_name}.pth")


def evaluate(
    train_dataset_name: str,
    test_dataset_name: str,
    feature: str = "dnf",
    num_samples: int = 1000,
):
    info(f"[评估模式] 训练集：{train_dataset_name}，测试集：{test_dataset_name}")

    # 加载测试数据集
    test_dataset = SubsetVideoFeatureDataset(
        VideoFeatureDataset(test_dataset_name), list(range(num_samples))
    )

    # 加载模型
    model = ViT(d_model=4096, num_heads=8, num_classes=2)
    model.load_state_dict(
        torch.load(f"{cwd}/models/{feature}/{train_dataset_name}.pth")
    )
    model = model.to(device)
    model.eval()

    # 评估模型
    with torch.no_grad():
        correct = 0
        total = 0
        for data in dataloader(test_dataset):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        info(f"[DoCoF] 准确率：{acc:.2f}%")
        return acc


if __name__ == "__main__":
    """"""
    experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 训练数据集为所有的 fake 数据集（除了 Sora）
    train_dataset = [
        dataset_name
        for dataset_name, dataset_path in dataset_paths.items()
        if "/fake/" in dataset_path and "Sora" not in dataset_name
    ]

    info(f"[DoCoF] 训练数据集：{train_dataset}")

    # for train_dataset_name in train_dataset:
    #     train(train_dataset_name)

    for train_dataset_name in train_dataset:
        for test_dataset_name in dataset_paths.keys():
            acc = evaluate(train_dataset_name, test_dataset_name)
            # 写入文件
            open(f"{cwd}/results/{experiment_name}.csv", "a").write(
                f"{train_dataset_name},{test_dataset_name},{acc:.2f}\n"
            )
