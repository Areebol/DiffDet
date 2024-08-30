from utils import *


# 从视频帧中构建数据集
class VideoDetectionDatasetV1(Dataset):
    def __init__(self, paths, labels):
        super().__init__()
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frames = glob(os.path.join(self.paths[idx], "*"))
        label = self.labels[idx]
        return {"frames": frames, "fake": label, "path": self.paths[idx]}


# 从 NumPy 对象中加载数据集
class VideoDetectionDatasetV2(Dataset):
    def __init__(self, input, labels):
        super().__init__()
        self.inputs = input
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 返回数据类型为 torch.float32 的 input
        # (1, 2048), ()
        # tensor([[ 0.3582,  0.1869,  0.0369,  ...,  0.6187, -0.3901,  0.2335]]), 0
        return torch.tensor(self.inputs[idx], dtype=torch.float32), self.labels[idx]


# 从 NumPy 对象中加载数据集（传入标签）
class VideoDetectionDatasetV3(Dataset):
    def __init__(self, input, label):
        super().__init__()
        self.inputs = input
        self.label = label  # 0 (real) / 1 (fake)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), self.label


def batch_get_features_v1(video_detection_dataset_v1: VideoDetectionDatasetV1):
    clip_features = []
    for i in tqdm(range(len(video_detection_dataset_v1))):
        # 载入单个样本
        sample = video_detection_dataset_v1[i]

        # 计算单个样本的 CLIP 特征
        sample_features = []
        for i in range(0, 4):
            frame_path = os.path.join(sample["path"], f"frame{i}.jpg")
            frame = Image.open(frame_path)
            frame_features = clip_feature(frame)
            sample_features.append(frame_features.detach().cpu().numpy())

        clip_features.append(np.array(sample_features))

    labels = [0 if d["fake"] == 0 else 1 for d in video_detection_dataset_v1]
    return clip_features, labels


def clip_features_from_dataset(dataset_dir: str) -> np.ndarray:
    info(f"[CLIP 特征] 从数据集中提取特征：{dataset_dir}")
    samples_path = glob(f"{dataset_dir}/*")
    info(f"[CLIP 特征] 数据集样本数量：{len(samples_path)}")
    clip_features = []
    for i, sample_path in enumerate(tqdm(samples_path)):
        frames_path = sorted(glob(f"{sample_path}/*"))
        sample_features = []
        for frame_path in frames_path:
            frame = Image.open(frame_path)
            frame_feature = clip_feature(frame)
            sample_features.append(frame_feature.detach().cpu().numpy())
        clip_features.append(np.array(sample_features))

    clip_features = np.array(clip_features)
    info(f"[CLIP 特征] 数据集特征尺寸：{clip_features.shape}")
    return clip_features


def load_features(input_path: str, output_path: str) -> VideoDetectionDatasetV2:
    input = np.load(input_path)
    labels = np.load(output_path)
    input = input.reshape((len(input), 1, 2048))  # NUM_SAMPLES, 1, 2048
    video_detection_dataset_v2 = VideoDetectionDatasetV2(input, labels)
    return video_detection_dataset_v2


def split_dataset(video_detection_dataset_v2: VideoDetectionDatasetV2):
    # 划分训练集和测试集
    train_size = int(0.8 * len(video_detection_dataset_v2))
    test_size = len(video_detection_dataset_v2) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        video_detection_dataset_v2, [train_size, test_size]
    )
    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset):
    """根据训练集和测试集构建数据加载器"""
    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_dataloader, test_dataloader


def __archive():
    real_paths = glob(f"{cwd}/realfake-video-dataset/real/*/*")
    fake_paths = glob(f"{cwd}/realfake-video-dataset/fake/*/*")
    paths = np.concatenate((real_paths, fake_paths))
    info(f"[数据集路径] 真实视频数量：{len(real_paths)}，假视频数量：{len(fake_paths)}")
    info(f"[数据集路径] 路径展示:{paths[:5]}")
    labels = np.concatenate((np.zeros(len(real_paths)), np.ones(len(fake_paths))))
    info(f"[数据集路径] 标签数量：{len(labels)}")

    video_detection_dataset_v1 = VideoDetectionDatasetV1(paths, labels)
    clip_features, labels = batch_get_features_v1(video_detection_dataset_v1)
    np.save("clip_input", clip_features)
    np.save("clip_output", labels)


if __name__ == "__main__":
    for dataset_path in glob(f"{cwd}/datasets/*/*"):
        info(f"[CLIP 特征] 为 {dataset_path} 提取特征")
        clip_features = clip_features_from_dataset(dataset_path)

        # CLIP 特征保存路径
        out_features_path = dataset_path.replace("/datasets/", "/out/clip_feature/")
        out_features_path = Path(out_features_path).with_suffix(".npy")
        out_features_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 CLIP 特征
        info(f"[CLIP 特征] 保存特征到：{out_features_path}")
        np.save(out_features_path, clip_features)
