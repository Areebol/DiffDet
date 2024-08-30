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
        return torch.tensor(self.inputs[idx], dtype=torch.float32), self.labels[idx]
        # # input 数据类型转换后为 torch.float16 (Half)
        # return (self.inputs[idx], self.labels[idx])


def batch_get_features(video_detection_dataset_v1: VideoDetectionDatasetV1):
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


def load_features(input_path: str, output_path: str) -> VideoDetectionDatasetV2:
    input = np.load(input_path)
    labels = np.load(output_path)
    input = input.reshape((len(input), 1, 2048))
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


if __name__ == "__main__":
    # 数据集路径
    real_paths = glob(f"{cwd}/realfake-video-dataset/real/*/*")
    fake_paths = glob(f"{cwd}/realfake-video-dataset/fake/*/*")
    paths = np.concatenate((real_paths, fake_paths))
    info(f"[数据集路径] 真实视频数量：{len(real_paths)}，假视频数量：{len(fake_paths)}")
    info(f"[数据集路径] 路径展示:{paths[:5]}")
    labels = np.concatenate((np.zeros(len(real_paths)), np.ones(len(fake_paths))))
    info(f"[数据集路径] 标签数量：{len(labels)}")

    video_detection_dataset_v1 = VideoDetectionDatasetV1(paths, labels)
    clip_features, labels = batch_get_features(video_detection_dataset_v1)
    np.save("clip_input", clip_features)
    np.save("clip_output", labels)
