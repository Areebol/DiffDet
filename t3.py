import torch

file = "features/dnf/genvideo/train/fake/DynamicCrafter/DynamicCrafter_008.pt"

feature = torch.load(file)
print(f"feature: {feature.shape}")
