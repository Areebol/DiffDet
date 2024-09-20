import torch

file = "features/dnf/genvideo/train/fake/DynamicCrafter/DynamicCrafter_008.pt"

feature = torch.load(file)
feature.cuda()
feature = feature.flatten()
print(f"feature: {feature.shape}")
feature = feature.unsqueeze(0)

from model import MLPViT

# 进行一次前馈和反向传播

model = MLPViT(
    mlp_input_size=3 * 8 * 224 * 224,
    mlp_hidden_size=2048,
    mlp_output_size=4096,
    vit_d_model=4096,
    vit_num_heads=8,
    num_classes=2,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1
)

model.cuda()

y = model(feature)
y.backward()
