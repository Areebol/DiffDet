from utils import *


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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


class MLPViT(nn.Module):
    def __init__(
        self,
        mlp_input_size,
        mlp_hidden_size,
        mlp_output_size,
        vit_d_model,
        vit_num_heads,
        num_classes,
    ):
        super().__init__()
        self.mlp = MLP(mlp_input_size, mlp_hidden_size, mlp_output_size)
        self.vit = ViT(vit_d_model, vit_num_heads, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        x = self.vit(x)
        return x
