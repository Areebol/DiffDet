import torch

file = "features/dnf/genvideo/train/fake/DynamicCrafter/DynamicCrafter_008.pt"

feature = torch.load(file)
feature.cuda()
print(f"feature: {feature.shape}")
exit()

# from model import MLPViT

# 进行一次前馈和反向传播

# model = MLPViT(
#     mlp_input_size=3 * 8 * 224 * 224,
#     mlp_hidden_size=2048,
#     mlp_output_size=4096,
#     vit_d_model=4096,
#     vit_num_heads=8,
#     num_classes=2,
# )

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(
#     params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
# )

import torch
from vit_pytorch.vivit import ViT

v = ViT(
    image_size=224,  # image size
    frames=8,  # number of frames
    image_patch_size=16,  # image patch size
    frame_patch_size=2,  # frame patch size
    num_classes=2,
    dim=1024,
    spatial_depth=6,  # depth of the spatial transformer
    temporal_depth=6,  # depth of the temporal transformer
    heads=8,
    mlp_dim=2048,
    variant="factorized_encoder",  # or 'factorized_self_attention'
)
# v = ViT(
#     image_size=128,  # image size
#     frames=16,  # number of frames
#     image_patch_size=16,  # image patch size
#     frame_patch_size=2,  # frame patch size
#     num_classes=1000,
#     dim=1024,
#     spatial_depth=6,  # depth of the spatial transformer
#     temporal_depth=6,  # depth of the temporal transformer
#     heads=8,
#     mlp_dim=2048,
#     variant="factorized_encoder",  # or 'factorized_self_attention'
# )

video = torch.randn(64, 3, 8, 224, 224)  # (batch, channels, frames, height, width)
# video = torch.randn(4, 3, 16, 128, 128)  # (batch, channels, frames, height, width)

preds = v(video)  # (4, 1000)
print(f"preds: {preds.shape}")
# print(f"preds: {preds}")
