from utils import *
from data_setup import *
from model import ViT


def load_vit_model():
    vit_model_path = f"{cwd}/models/vit.pth"
    vit_model = ViT(d_model=2048, num_heads=8, num_classes=2)
    vit_model.load_state_dict(
        torch.load(vit_model_path, map_location=device, weights_only=True)
    )
    vit_model.to(device)
    return vit_model


def eval_from_video_frames(video_frames_dir):
    video_frames_paths = sorted(glob(f"{video_frames_dir}/*"))

    # 获取测试数据的 CLIP 特征
    test_data = np.array(
        [
            clip_feature(Image.open(video_frames_paths[i])).cpu().detach().numpy()
            for i in range(4)
        ]
    ).reshape(1, 1, 2048)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

    # 加载模型
    info("加载 ViT 模型 ...")
    vit_model = load_vit_model()

    # 测试模型
    vit_model.eval()
    with torch.no_grad():
        result = vit_model(test_data)

    info(f"模型输出: {result.cpu().detach().numpy()}")
    info(f"预测结果: {np.argmax(result.cpu().detach().numpy())}")


if __name__ == "__main__":
    eval_from_video_frames(f"{cwd}/realfake-video-dataset/test_0")
