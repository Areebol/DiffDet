# %%
import matplotlib.pyplot as plt
from data_setup import *
from score import *
from scipy.stats import mannwhitneyu


def save_scores(
    datasets,
    calc_fn=calc_video_score_by_timestep_v2,
    out_data_file="12091722_scores.json",
):
    """将分数和标签保存到 json 文件"""

    # 创建输出目录
    os.makedirs("out", exist_ok=True)

    results = []

    for name, dataset in datasets.items():
        # 取前50个样本
        dataset = SubsetVideoFeatureDataset(dataset, list(range(50)))

        for data in dataset:
            if data is None:
                continue

            features, label = data
            # 计算每个时间步的分数
            scores = calc_fn(features)

            # 将 tensor 转换为 float
            scores = {k: float(v.cpu()) for k, v in scores.items()}

            results.append({"label": int(label), "scores": scores})

    # 保存到 json 文件
    with open(f"out/{out_data_file}", "w") as f:
        json.dump(results, f, indent=2)

    info(f"保存分数到 out/{out_data_file}，共 {len(results)} 条记录")


def load_scores(data_file="12091722_scores.json"):
    """从 json 文件加载分数和标签"""
    with open(f"out/{data_file}", "r") as f:
        results = json.load(f)
    info(f"从 out/{data_file} 加载分数，共 {len(results)} 条记录")
    return results


def plot_score_v1(datasets):
    plt.figure(figsize=(10, 6))

    for name, dataset in datasets.items():
        # 取前50个样本
        dataset = SubsetVideoFeatureDataset(dataset, list(range(50)))

        for data in dataset:
            if data is None:
                continue

            features, label = data
            # 计算每个时间步的分数
            scores = calc_video_score_by_timestep(features)

            # 提取x和y值
            x = list(scores.keys())
            y = [scores[k].cpu().numpy() for k in x]

            # 根据标签选择颜色
            color = "blue" if label == 0 else "red"

            # 画折线
            plt.plot(x, y, color=color, alpha=0.5)

    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.title("Video Scores by Timestep")
    plt.grid(True)
    plt.show()


def plot_score_v2():
    """第二个版本的画图，使用归一化后的分数"""
    # 加载数据
    results = load_scores()

    # 将数据按标签分组
    real_data = []
    fake_data = []
    for result in results:
        if result["label"] == 0:
            real_data.append(result["scores"])
        else:
            fake_data.append(result["scores"])

    plt.figure(figsize=(10, 6))

    # 获取所有时间步
    timesteps = list(results[0]["scores"].keys())

    # 对每个时间步进行归一化并画图
    for data, color, label in [(real_data, "blue", "Real"), (fake_data, "red", "Fake")]:
        normalized_scores = []

        # 对每个时间步计算所有样本在该时间步的最大最小值
        for t_idx, t in enumerate(timesteps):
            scores_at_t = [float(scores[str(t)]) for scores in data]
            y_min = min(scores_at_t)
            y_max = max(scores_at_t)

            # 如果是第一个时间步，为每个样本创建一个空列表
            if t_idx == 0:
                normalized_scores = [[] for _ in range(len(data))]

            # 对每个样本在该时间步进行归一化
            for sample_idx, score in enumerate(scores_at_t):
                if y_max > y_min:
                    norm_score = (score - y_min) / (y_max - y_min)
                else:
                    norm_score = score
                normalized_scores[sample_idx].append(norm_score)

        # 画每个样本的折线
        for y in normalized_scores:
            plt.plot(timesteps, y, color=color, alpha=0.2)

        # 计算并画出平均值
        mean_scores = []
        for i in range(len(timesteps)):
            mean_scores.append(np.mean([s[i] for s in normalized_scores]))
        plt.plot(
            timesteps, mean_scores, color=color, linewidth=2, label=f"{label} Mean"
        )

    plt.xlabel("Timestep")
    plt.ylabel("Normalized Score")
    plt.title("Video Score Distribution at Different Timesteps")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_auc_v3(data_file="12091722_scores.json"):
    """第三个版本的 AUC 曲线绘制函数,使用 mannwhitneyu 计算每个时间步的 AUC"""
    # 加载数据
    results = load_scores(data_file)

    # 分离真假样本数据
    real_data = []
    fake_data = []
    for result in results:
        if result["label"] == 0:
            real_data.append(result["scores"])
        else:
            fake_data.append(result["scores"])

    # 获取所有时间步
    timesteps = list(results[0]["scores"].keys())

    # 计算每个时间步的 AUC
    auc_scores = []
    for t in timesteps:
        # 收集该时间步的所有分数
        real_scores = [float(scores[str(t)]) for scores in real_data]
        fake_scores = [float(scores[str(t)]) for scores in fake_data]

        # 计算 Mann-Whitney U 统计量和 AUC,将真假样本反过来比较
        u_statistic, _ = mannwhitneyu(fake_scores, real_scores, alternative="greater")
        auc = u_statistic / (len(real_scores) * len(fake_scores))
        auc_scores.append(auc)

    # 绘制 AUC 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, auc_scores, "b-", linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("AUC Score")
    plt.title("AUC Scores at Different Timesteps")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 加载数据集
    datasets = {
        "DynamicCrafter": VideoFeatureDataset("DynamicCrafter"),
        "MSR-VTT": VideoFeatureDataset("MSR-VTT"),
    }

    save_scores(
        datasets,
        calc_fn=calc_video_score_by_timestep_v3,
        out_data_file="12101515_scores.json",
    )
    plot_auc_v3(data_file="12101515_scores.json")
