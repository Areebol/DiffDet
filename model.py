from utils import *


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


