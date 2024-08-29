from utils import *
from data_setup import *
from model import *


if __name__ == "__main__":
    # 加载数据集特征
    video_detection_dataset = load_features(f"{cwd}/clip_input", f"{cwd}/clip_output")

    # 划分数据集
    train_dataset, test_dataset = split_dataset(video_detection_dataset)

    # 加载训练模型
    vit_model = ViT(d_model=2048, num_heads=8, num_classes=2)

    # 模型小结
    summary(
        model=vit_model,
        input_size=(4, 1, 2048),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    # 优化器
    optimizer = torch.optim.Adam(
        params=vit_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
    )

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 数据加载器
    train_dataloader, test_dataloader = get_dataloader(train_dataset, test_dataset)

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
