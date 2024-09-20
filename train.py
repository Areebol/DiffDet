from utils import *
from data_setup import *

# from model import *
from vit_pytorch.vivit import ViT

# # 模型小结
# summary(
#     model=ViT(d_model=4096, num_heads=8, num_classes=2),
#     # input_size=(4, 1, 2048),
#     # col_names=["input_size", "output_size", "num_params", "trainable"],
#     # col_width=20,
#     # row_settings=["var_names"],
# )

# model = MLPViT(
#     mlp_input_size=3 * 8 * 224 * 224,
#     mlp_output_size=672,
#     vit_d_model=672,
#     vit_num_heads=1,
#     num_classes=2,
# ).to(device)

model = ViT(
    image_size=224,  # image size
    frames=8,  # number of frames
    image_patch_size=16,  # image patch size
    frame_patch_size=2,  # frame patch size
    num_classes=2,
    dim=4096,
    spatial_depth=6,  # depth of the spatial transformer
    temporal_depth=6,  # depth of the temporal transformer
    heads=8,
    mlp_dim=2048,
    variant="factorized_encoder",  # or 'factorized_self_attention'
)

summary(model=model)
# exit()


def _train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
):

    model = model.to(device)

    for epoch in range(epochs):

        # 训练模型
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        info(
            f"[DoCoF] Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.3f}, Acc: {train_acc:.2f}%"
        )

        # 测试模型
        model.eval()

        with torch.inference_mode():
            correct = 0
            total = 0
            for data in tqdm(test_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_acc = 100 * correct / total
            info(f"[DoCoF] 准确率：{test_acc:.2f}%")

    return model


def train_on(dataset_name: str, feature: str = "dnf", num_samples: int = 1000):
    info(f"[DoCoF] 训练集：{dataset_name}")

    global model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
    )

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
    model = _train(
        model=model,
        train_dataloader=dataloader(train_dataset),
        test_dataloader=dataloader(val_dataset),
        criterion=criterion,
        optimizer=optimizer,
        epochs=10,
        device=device,
    )

    # 保存模型
    torch.save(model.state_dict(), f"{cwd}/models/{feature}/{dataset_name}.pth")


def evaluate_on(
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

    global model
    model.load_state_dict(
        torch.load(f"{cwd}/models/{feature}/{train_dataset_name}.pth")
    )
    model = model.to(device)

    # 评估模型
    with torch.inference_mode():
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


# def evaluate(
#     train_dataset_name: str,
#     test_dataset_name: str,
#     feature: str = "dnf",
#     num_samples: int = 1000,
# ):
#     info(f"[评估模式] 训练集：{train_dataset_name}，测试集：{test_dataset_name}")

#     # 加载测试数据集
#     test_dataset = SubsetVideoFeatureDataset(
#         VideoFeatureDataset(test_dataset_name), list(range(num_samples))
#     )

#     # 加载模型
#     model = ViT(d_model=4096, num_heads=8, num_classes=2)
#     model.load_state_dict(
#         torch.load(f"{cwd}/models/{feature}/{train_dataset_name}.pth")
#     )
#     model = model.to(device)
#     model.eval()

#     # 评估模型
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for data in dataloader(test_dataset):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         acc = 100 * correct / total
#         info(f"[DoCoF] 准确率：{acc:.2f}%")
#         return acc


if __name__ == "__main__":
    """"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 训练数据集为所有的 fake 数据集（除了 Sora）
    train_dataset = [
        dataset_name
        for dataset_name, dataset_path in dataset_paths.items()
        if "/fake/" in dataset_path and "Sora" not in dataset_name
    ]

    info(f"[DoCoF] 训练数据集：{train_dataset}")

    for train_dataset_name in train_dataset:
        train_on(train_dataset_name)

    for train_dataset_name in train_dataset:
        for test_dataset_name in dataset_paths.keys():
            acc = evaluate_on(train_dataset_name, test_dataset_name)
            # 写入文件
            open(f"{cwd}/results/{experiment_name}.csv", "a").write(
                f"{train_dataset_name},{test_dataset_name},{acc:.2f}\n"
            )

    # # 训练模型
    # train_on(dataset_name="DynamicCrafter", num_samples=1000)

    # for test_dataset_name in dataset_paths.keys():
    #     acc = evaluate_on("DynamicCrafter", test_dataset_name, num_samples=1000)
