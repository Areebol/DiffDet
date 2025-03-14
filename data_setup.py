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
from score import extract_video_score
import threading
from queue import Queue

FEATURE = "score"
# FEATURE = "dnf"

# 根据特征类型设置帧数
NUM_FRAMES = 10 if FEATURE == "score" else 8
FEATURE_LEN = NUM_FRAMES * 512

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
clip_model, clip_preprocess = None, None
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
diffusion_model = None
diffusion_preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ]
)


def get_clip_model():
    global clip_model, clip_preprocess
    if clip_model is None or clip_preprocess is None:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
    return clip_model, clip_preprocess


def get_diffusion_model():
    global diffusion_model
    if diffusion_model is None:
        diffusion_model = Diffusion()
        diffusion_ckpt = (
            f"{cwd}/models/dnf_diffusion.ckpt"  # Diffusion trained on LSUN Bedroom
        )
        diffusion_model.load_state_dict(
            torch.load(diffusion_ckpt, weights_only=True, map_location=device)
        )
        diffusion_model.eval()
    return diffusion_model


def clip_feature(img: Image) -> torch.Tensor:
    """提取图像的 CLIP 特征"""
    clip_model, clip_preprocess = get_clip_model()
    # 对图像进行预处理
    img = clip_preprocess(img).unsqueeze(0).to(device)
    # 用 CLIP 提取特征
    features = clip_model.encode_image(img)
    return features  # torch.Size([1, 512])


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
        t = (torch.ones(n) * seq[0]).to(device)  # t=[0. ... 0.]
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
            f"[数据集] 初始化数据集 {self.dataset_name}，特征：{self.feature}，"
            f"总样本数量：{len(self)}，标签：{self.fake}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Optional[Tuple[torch.Tensor, int]]:
        timer = Timer()
        video_path = self.files[idx]
        feature_path = self._get_feature_path(video_path)

        cached_features = self._load_cached_feature(feature_path)
        if cached_features is not None:
            return cached_features, self.fake

        # 读取并处理视频帧
        if frames := self._read_video_frames(video_path):
            features = self._extract_features(frames)
            self._save_features(features, feature_path)
            debug(f"[数据集] 样本处理完成，总耗时：{timer.tick()}")
            return features, self.fake

        # 处理失败，删除损坏的视频文件
        video_path.unlink(missing_ok=True)
        return None

    def __repr__(self) -> str:
        return f"{self.dataset_name} ({len(self)})"

    def _get_feature_path(self, video_path: Path) -> Path:
        """根据视频路径生成特征文件路径"""
        return Path(
            str(video_path).replace(DATASETS_DIR, f"{FEATURES_DIR}/{self.feature}")
        )

    def _load_clip_feature(self, feature_path: Path) -> Optional[torch.Tensor]:
        """加载CLIP特征文件"""
        if not feature_path.with_suffix(".npy").exists():
            return None

        timer = Timer()
        debug(f"[数据集] 加载CLIP特征：{feature_path}")

        try:
            features_array = np.load(feature_path.with_suffix(".npy"))
            features_tensor = torch.tensor(
                features_array, dtype=torch.float32, device=device
            )
            features_tensor = features_tensor.reshape(1, FEATURE_LEN)
            debug(f"[数据集] 加载CLIP特征耗时：{timer.tick()}")
            return features_tensor.to(device)
        except Exception as e:
            error(f"[数据集] 加载CLIP特征失败：{e}")
            return None

    def _load_dnf_feature(self, feature_path: Path) -> Optional[torch.Tensor]:
        """加载DNF特征文件"""
        if not feature_path.with_suffix(".pt").exists():
            return None

        timer = Timer()
        debug(f"[数据集] 加载DNF特征：{feature_path}")

        try:
            features = torch.load(feature_path.with_suffix(".pt"), weights_only=True)
            debug(f"[数据集] 加载DNF特征耗时：{timer.tick()}")
            return features.to(device)
        except Exception as e:
            error(f"[数据集] 加载DNF特征失败：{e}")
            return None

    def _load_score_feature(self, feature_path: Path) -> Optional[torch.Tensor]:
        """加载score特征文件"""
        if not feature_path.with_suffix(".pt").exists():
            return None

        timer = Timer()
        debug(f"[数据集] 加载score特征：{feature_path}")

        try:
            features = torch.load(feature_path.with_suffix(".pt"), weights_only=True)
            debug(f"[数据集] 加载score特征耗时：{timer.tick()}")
            return features.to(device)
        except Exception as e:
            error(f"[数据集] 加载score特征失败：{e}")
            return None

    def _load_cached_feature(self, feature_path: Path) -> Optional[torch.Tensor]:
        """加载缓存的特征文件"""
        if self.feature == "clip":
            return self._load_clip_feature(feature_path)
        elif self.feature == "dnf":
            return self._load_dnf_feature(feature_path)
        elif self.feature == "score":
            return self._load_score_feature(feature_path)
        return None

    def _read_video_frames(self, video_path: Path) -> Optional[List]:
        """读取视频帧"""
        timer = Timer()
        debug(f"[数据集] 开始读取视频：{video_path}")

        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            for _ in range(NUM_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) < NUM_FRAMES:
                error(f"[数据集] 视频帧数不足：{video_path}")
                return None

            debug(f"[数据集] 读取视频帧完成，耗时：{timer.tick()}")
            return frames
        except Exception as e:
            error(f"[数据集] 读取视频失败：{video_path}, 错误：{e}")
            return None

    def _extract_features(self, frames: List) -> torch.Tensor:
        timer = Timer()

        if self.feature == "score":
            score_transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # 将 0-255 归一化到 0-1
                    transforms.Resize(
                        (256, 256), interpolation=InterpolationMode.BICUBIC
                    ),
                ]
            )

            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            frames = [score_transform(frame) for frame in frames]
            frames_tensor = torch.stack(frames)
            frames_tensor = torch.clamp(frames_tensor, 0, 1)
            debug(f"[数据集] 视频帧转换为 tensor，形状：{frames_tensor.shape}")

            features = extract_video_score(frames_tensor)
            debug(f"[数据集] Score 特征提取完成，形状：{features.shape}")
            return features

        else:
            features = []

            for frame in frames:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.feature == "clip":
                    feature = clip_feature(img).detach().cpu().numpy()
                elif self.feature == "dnf":  # dnf
                    img = diffusion_preprocess(img)
                    feature_dnf = dnf_feature(img)
                    feature = feature_dnf.reshape(3, 256, 256)[
                        :, 16:240, 16:240
                    ].detach()
                features.append(feature)

            features = torch.stack(features)
            features = features.permute(1, 0, 2, 3).type(torch.float32)
            debug(f"[数据集] 特征提取完成，耗时：{timer.tick()}")
            return features

    def _save_features(self, features: torch.Tensor, feature_path: Path) -> None:
        """保存特征"""
        timer = Timer()
        try:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, feature_path.with_suffix(".pt"))
            debug(f"[数据集] 特征保存完成，耗时：{timer.tick()}")
        except Exception as e:
            error(f"[数据集] 保存特征失败：{e}")


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
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )


def process_gpu_samples(device, dataset, start_idx, end_idx):
    """在指定 GPU 上处理数据样本的线程函数"""
    torch.cuda.set_device(device)
    for idx in range(start_idx, end_idx):
        data = dataset[idx]


if __name__ == "__main__":
    """"""

    # 加载数据集
    fake_datasets = {
        "DynamicCrafter": VideoFeatureDataset("DynamicCrafter"),
        "Latte": VideoFeatureDataset("Latte"),
        "OpenSora": VideoFeatureDataset("OpenSora"),
        "Pika": VideoFeatureDataset("Pika"),
        "SEINE": VideoFeatureDataset("SEINE"),
        "Crafter": VideoFeatureDataset("Crafter"),
        "Gen2": VideoFeatureDataset("Gen2"),
        "HotShot": VideoFeatureDataset("HotShot"),
        "Lavie": VideoFeatureDataset("Lavie"),
        "ModelScope": VideoFeatureDataset("ModelScope"),
        "MoonValley": VideoFeatureDataset("MoonValley"),
        "MorphStudio": VideoFeatureDataset("MorphStudio"),
        "Show_1": VideoFeatureDataset("Show_1"),
        "WildScrape": VideoFeatureDataset("WildScrape"),
        "Sora": VideoFeatureDataset("Sora"),
    }

    # 对假视频数据集取前50个样本并遍历
    for name, dataset in fake_datasets.items():
        dataset = SubsetVideoFeatureDataset(dataset, list(range(50)))
        for data in dataset:
            data

    # 加载真实视频数据集并取前500个样本并遍历
    real_dataset = VideoFeatureDataset("MSR-VTT")
    real_dataset = SubsetVideoFeatureDataset(real_dataset, list(range(500)))
    for data in real_dataset:
        data
