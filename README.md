CLIP 预处理之后的图像：（关键看这个）
clip_preprocess_img: torch.Size([1, 3, 224, 224]) [cuda:0] (-1.792 0.456 2.146) torch.float32

CLIP 提取的特征：
clip_feature: torch.Size([1, 512]) [cuda:0] (-6.816 -0.015 1.158) torch.float16

两者完全相等
feature_without_batch_dim = feature.squeeze(0) # torch.Size([3, 256, 256]) [cuda:0] (-5.319 -0.012 4.882) torch.float32
grid = make_grid(feature) # torch.Size([3, 256, 256]) [cuda:0] (-5.319 -0.012 4.882) torch.float32
torch.equal(feature_without_batch_dim, grid) # True


dnf_feature: torch.Size([1, 3, 256, 256]) [cuda:0] (-5.319 -0.012 4.882) torch.float32
dnf_feature_without_batch_dim: torch.Size([3, 256, 256]) [cuda:0] (-5.319 -0.012 4.882) torch.float32
dnf_feature_rz: torch.Size([1, 3, 224, 224]) [cuda:0] (-4.276 -0.012 3.774) torch.float32
dnf_feature_rz: torch.Size([1, 3, 224, 224]) [cuda:0] (-3.500 -0.004 3.734) torch.float32
ndarr: torch.Size([256, 256, 3]) [cuda:0] (0.000 53.099 255.000) torch.float32
feature_256: torch.Size([256, 256, 3]) [cuda:0] (0.000 53.099 255.000) torch.float32
clip_feature: torch.Size([1, 512]) [cuda:0] (-7.953 -0.037 2.381) torch.float16
clip_feature: torch.Size([1, 512]) [cuda:0] (-6.965 -0.027 1.167) torch.float16