from utils import *
from model import ViT

if __name__ == "__main__":

    # 加载模型
    vit_model_path = f"{cwd}/models/vit.pth"

    vit_model = ViT(d_model=2048, num_heads=8, num_classes=2)
    vit_model.load_state_dict(torch.load(vit_model_path))

    # 测试模型
    path = f"{cwd}/kaggle/working/test/sora/tokyo-walk_1"
    test_data = np.array(
        [
            clip_feature(Image.open(f"{path}/frame0.jpg")).cpu().detach().numpy(),
            clip_feature(Image.open(f"{path}/frame1.jpg")).cpu().detach().numpy(),
            clip_feature(Image.open(f"{path}/frame2.jpg")).cpu().detach().numpy(),
            clip_feature(Image.open(f"{path}/frame3.jpg")).cpu().detach().numpy(),
        ]
    )

    test_data = test_data.reshape(1, 1, 2048)
    vit_model.eval()
    data = vit_model(torch.tensor(test_data).to(device).to(torch.float32))
    print(np.argmax(data.cpu().detach().numpy()))
