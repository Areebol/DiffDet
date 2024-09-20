from utils import *


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        return x


class ViT(nn.Module):
    def __init__(
        self, d_model: int = 2048, num_heads: int = 12, num_classes: int = 1000
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads
        )
        """
        nn.TransformerEncoderLayer 的输出形状通常是 (sequence_length, batch_size, d_model)，也就是 (T, N, D)
        其中 T 是序列长度，N 是批次大小，D 是模型维度（在你的例子中是 2048）。
        """
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            nn.Linear(in_features=d_model, out_features=num_classes),
        )

    def forward(self, x):
        x = self.encoder_layer(x)
        # print(f"x: {tensor_detail(x)}")
        x = self.classifier(x[0])
        # print(f"x: {tensor_detail(x)}")
        return x


class MLPViT(nn.Module):
    def __init__(
        self,
        mlp_input_size,
        mlp_output_size,
        vit_d_model,
        vit_num_heads,
        num_classes,
    ):
        super().__init__()
        self.mlp = MLP(input_size=mlp_input_size, output_size=mlp_output_size)
        self.vit = ViT(
            d_model=vit_d_model, num_heads=vit_num_heads, num_classes=num_classes
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.unsqueeze(0)
        x = self.vit(x)
        return x
