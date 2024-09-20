from utils import *
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    random_split,
    default_collate,
)
from diffusion import Diffusion
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms.functional import InterpolationMode

NUM_FRAMES = 8
FEATURE_LEN = NUM_FRAMES * 512
FEATURE = "dnf"

load_size = 256
crop_size = 224

DATASETS_DIR = "/root/jyj/proj/decof/datasets"
FEATURES_DIR = "/root/jyj/proj/decof/features"

dataset_paths = {
    "DynamicCrafter": f"{DATASETS_DIR}/genvideo/train/fake/DynamicCrafter",
    "Latte": f"{DATASETS_DIR}/genvideo/train/fake/Latte",
    "OpenSora": f"{DATASETS_DIR}/genvideo/train/fake/OpenSora",
    "Pika": f"{DATASETS_DIR}/genvideo/train/fake/Pika",
    "SEINE": f"{DATASETS_DIR}/genvideo/train/fake/SEINE",
    "Crafter": f"{DATASETS_DIR}/genvideo/val/fake/Crafter",
    "Gen2": f"{DATASETS_DIR}/genvideo/val/fake/Gen2",
    "HotShot": f"{DATASETS_DIR}/genvideo/val/fake/HotShot",
    "Lavie": f"{DATASETS_DIR}/genvideo/val/fake/Lavie",
    "ModelScope": f"{DATASETS_DIR}/genvideo/val/fake/ModelScope",
    "MoonValley": f"{DATASETS_DIR}/genvideo/val/fake/MoonValley",
    "MorphStudio": f"{DATASETS_DIR}/genvideo/val/fake/MorphStudio",
    "Show_1": f"{DATASETS_DIR}/genvideo/val/fake/Show_1",
    "WildScrape": f"{DATASETS_DIR}/genvideo/val/fake/WildScrape",
    "Sora": f"{DATASETS_DIR}/genvideo/val/fake/Sora",
    "MSR-VTT": f"{DATASETS_DIR}/genvideo/val/real/MSR-VTT",
}


# 特征提取 CLIP 模型
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
"""
clip_preprocess 自带预处理
Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,  # image.convert("RGB")
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
"""


def clip_feature(img: Image) -> torch.Tensor:
    """提取图像的 CLIP 特征"""
    # 对图像进行预处理
    img = clip_preprocess(img).unsqueeze(0).to(device)
    # 用 CLIP 提取特征
    features = clip_model.encode_image(img)
    return features  # torch.Size([1, 512])


# 特征提取 Diffusion 模型
diffusion_model = Diffusion()
diffusion_ckpt = f"{cwd}/models/dnf_diffusion.ckpt"  # Diffusion trained on LSUN Bedroom
diffusion_model.load_state_dict(torch.load(diffusion_ckpt, weights_only=True))
diffusion_model = diffusion_model.to(device)
diffusion_model.eval()
diffusion_preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ]
)


def dnf_feature(x: torch.Tensor) -> torch.Tensor:
    total_timesteps = 1000
    step = 20
    seq = list(map(int, np.linspace(0, total_timesteps, step + 1)))

    # 若输入为单张图像，而非一个图像批量，则添加第一个维度
    if x.dim() == 3:
        x = x.unsqueeze(0)

    x = x.to(device)
    with torch.no_grad():
        n = x.size(0)  # 获取输入张量 x 的第一个维度的大小，即批量大小
        t = (torch.ones(n) * seq[0]).to(device)
        et = diffusion_model(x, t)
    return et


class VideoFeatureDataset(Dataset):
    def __init__(self, dataset_name: str, feature=FEATURE):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_paths[dataset_name]
        self.files = list(Path(self.dataset_path).glob("*"))
        self.fake = 1 if "/fake/" in str(self.dataset_path) else 0
        self.feature = feature
        info(
            f"[数据集] 初始化数据集 {self.dataset_name}，特征：{self.feature}，总样本数量：{len(self)}，标签：{self.fake}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # try:
        t = time.time()

        # 构建特征文件路径
        feature_path = Path(
            str(self.files[idx]).replace(DATASETS_DIR, f"{FEATURES_DIR}/{self.feature}")
        )

        # 如果已有特征文件则直接加载（速率：4s/万）
        if self.feature == "clip" and feature_path.with_suffix(".npy").exists():
            debug(f"[数据集] 加载特征：{feature_path}")
            features_array = np.load(feature_path.with_suffix(".npy"))
            features_tensor = torch.tensor(
                features_array, dtype=torch.float32, device=device
            )
            # 如果形状不是 (1, -1)，则 reshape
            try:
                debug(f"[数据集] 原特征形状：{features_tensor.shape}")
                features_tensor = features_tensor.reshape(1, FEATURE_LEN)
                debug(f"[数据集] 重塑后特征形状：{features_tensor.shape}")
            except Exception as e:
                error(f"[数据集] 重塑特征出错：{e}")
                error(f"[数据集] 原特征形状：{features_tensor.shape}")
                error(f"[数据集] 原特征：{features_tensor}")
                # 删除视频和特征文件
                info(f"[数据集] 删除视频和特征文件：{feature_path.with_suffix('.npy')}")
                self.files[idx].unlink()
                feature_path.with_suffix(".npy").unlink()
                return None

            debug(f"[数据集] 加载特征耗时：{time.time() - t:.2f} 秒")
            return features_tensor.to(device), self.fake

        elif self.feature == "dnf" and feature_path.with_suffix(".pt").exists():
            debug(f"[数据集] 加载特征：{feature_path}")
            features = torch.load(feature_path.with_suffix(".pt"), weights_only=True)
            features: torch.Tensor = features.to(device)
            debug(f"[数据集] 加载特征耗时：{time.time() - t:.2f} 秒")
            return features, self.fake

        # 读入视频文件
        try:
            cap = cv2.VideoCapture(self.files[idx])
            debug(f"[数据集] 读取视频耗时：{time.time() - t:.2f} 秒")

            # 用 cv2 读取视频帧
            frames = []
            for i in range(NUM_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        except:
            error(f"[数据集] 读取视频帧出错：{self.files[idx]}")
            # 删除视频文件
            info(f"[数据集] 删除视频文件：{self.files[idx]}")
            self.files[idx].unlink()
            return None
        debug(f"[数据集] 读取视频帧耗时：{time.time() - t:.2f} 秒")

        # 用 CLIP 提取特征（速率：1500s/万个）
        features = []
        if self.feature == "clip":
            for frame in frames:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # CLIP 中自带图像预处理
                features.append(clip_feature(img).detach().cpu().numpy())

        elif self.feature == "dnf":
            for frame in frames:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 对图像进行 DNF 预处理
                img: torch.Tensor = diffusion_preprocess(img)
                # 提取图像的 DNF 特征
                feature_dnf = dnf_feature(img)  # (1, 3, 256, 256)

                # 将 DNF 特征中心裁剪成 224*224
                feature_dnf = feature_dnf.reshape(3, 256, 256)[:, 16:240, 16:240]

                # # 将 DNF 特征缩放成 224*224 以输入 CLIP 模型
                # feature_dnf_rz = nn.functional.interpolate(
                #     feature_dnf, (224, 224), mode="bicubic"
                # )
                # # 使用 CLIP 提取特征
                # feature_clip: torch.Tensor = clip_model.encode_image(feature_dnf_rz)

                features.append(feature_dnf.detach())

        features = torch.stack(features)
        # 交换维度 ->（通道，帧数，...）
        features = features.permute(1, 0, 2, 3).type(torch.float32)
        debug(f"[数据集] 提取视频特征耗时：{time.time() - t:.2f} 秒")

        # # 将特征列表转换为单一的 NumPy 数组
        # try:
        #     features_array = np.array(features).reshape(1, FEATURE_LEN)
        # except Exception as e:
        #     error(f"[数据集] 转换特征为 NumPy 数组出错：{e}")
        #     error(f"[数据集] 提取特征：{features}")
        #     # 删除视频和特征文件
        #     info(f"[数据集] 删除视频和特征文件：{feature_path.with_suffix('.npy')}")
        #     self.files[idx].unlink()
        #     feature_path.with_suffix(".npy").unlink()
        #     return None
        # debug(f"[数据集] 转换视频特征耗时：{time.time() - t:.2f} 秒")

        # # 保存特征，直接保存 NumPy 数组
        # feature_path.parent.mkdir(parents=True, exist_ok=True)
        # np.save(feature_path.with_suffix(".npy"), features_array)
        # debug(f"[数据集] 保存特征耗时：{time.time() - t:.2f} 秒")

        # # 转换为张量
        # features_tensor = torch.tensor(features_array, dtype=torch.float32)
        # debug(f"[数据集] 转换视频特征为张量耗时：{time.time() - t:.2f} 秒")

        # 保存特征
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, feature_path.with_suffix(".pt"))
        debug(f"[数据集] 保存特征耗时：{time.time() - t:.2f} 秒")

        return features, self.fake

    # except Exception as e:
    #     error(f"[数据集] 加载数据集 {self.dataset_path}[{idx}] 出错：{e}")
    #     return None

    def __repr__(self) -> str:
        return f"{self.dataset_name} ({len(self)})"


# 切片子数据集
class SubsetVideoFeatureDataset(Dataset):
    def __init__(self, dataset: VideoFeatureDataset, indices: list):
        super().__init__()
        self.dataset = dataset
        # 有效 indices
        self.indices = sorted(list(set(indices) & set(range(len(dataset)))))
        info(
            f"[数据集] 初始化子数据集 {self.dataset.dataset_path}，{self.dataset.feature}，样本数量：{len(self)}，标签：{self.dataset.fake}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __repr__(self) -> str:
        return f"{self.dataset.dataset_name} ({len(self)})"


def split_dataset(dataset: Dataset):
    # 划分训练集和测试集
    train_ratio = 0.8
    train_size = int(len(dataset) * train_ratio)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return train_dataset, val_dataset


# 使用 collate_fn 过滤无效数据（数据值为 None）
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch) if batch else None


def dataloader(dataset: Dataset):
    """传入数据集构建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    """"""

    # 加载所有数据集
    for dataset_name, dataset_path in dataset_paths.items():
        dataset = VideoFeatureDataset(dataset_name)
        sub_dataset = SubsetVideoFeatureDataset(dataset, list(range(3000)))
        for data in sub_dataset:
            data

    # dataset = VideoFeatureDataset(dataset_paths["DynamicCrafter"])

    # dataset[:100]

    # _dataloader = dataloader(dataset)
    # data = next(iter(_dataloader))
    # print(data, len(data[0]), len(data[1]))
