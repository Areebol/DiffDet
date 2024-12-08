from data_setup import DATASETS_DIR, FEATURES_DIR, FEATURE, dataset_paths
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
from utils import debug, device


selected_dataset = {
    "DynamicCrafter": {},
    "Latte": {},
    "OpenSora": {},
    "Pika": {},
    "MSR-VTT": {},
}


# Calculate the norms of each feature in the specified dataset, and draw a histogram based on density
def clac_norms(dataset_name):
    # 检查是否已经存在对应的 json 文件
    save_to = Path(f"norms/{dataset_name}.json")
    if save_to.exists():
        debug(f"[Calculating norms] {dataset_name} norms already exists, skipping...")
        return

    _feature_dir = Path(
        dataset_paths[dataset_name].replace(DATASETS_DIR, f"{FEATURES_DIR}/{FEATURE}")
    )
    debug(f"[Calculating norms] Feature path: {_feature_dir}")

    features_path = list(_feature_dir.glob("*.pt"))
    debug(f"[Calculating norms] Number of features: {len(features_path)}")

    norms = []
    for feature_path in tqdm(features_path):
        features = torch.load(feature_path)
        features = features.to(device)
        _norm = torch.norm(features, p=2)
        norms.append(_norm.item())

    # Save to JSON file
    save_to.parent.mkdir(parents=True, exist_ok=True)
    with open(save_to, "w") as f:
        json.dump(norms, f)


def plot_norms():
    plt.figure(figsize=(12, 7))

    colors = {
        "DynamicCrafter": "#FF9999",  # light red
        "Latte": "#FF99FF",  # light purple
        "OpenSora": "#99FF99",  # light green
        "Pika": "#FFFF99",  # light yellow
        "MSR-VTT": "#66B2FF",  # light blue
    }

    for dataset_name in selected_dataset:
        with open(f"norms/{dataset_name}.json", "r") as f:
            norms = json.load(f)

        norms_np = np.array(norms)

        sns.histplot(
            data=norms_np,
            bins=50,
            stat="density",
            color=colors[dataset_name],
            label=dataset_name,
            alpha=0.6,
        )

        plt.axvline(
            np.mean(norms_np),
            color=colors[dataset_name],
            linestyle="--",
            alpha=0.8,
            label=f"{dataset_name} mean: {np.mean(norms_np):.2f}",
        )

    # Modify the title and labels to English
    plt.title("Feature Norm Distribution Comparison")
    plt.xlabel("Norm Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the image
    save_to = Path("norms/distribution.png")
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close()

    # Print statistics in English
    for dataset_name in selected_dataset:
        with open(f"norms/{dataset_name}.json", "r") as f:
            norms = np.array(json.load(f))

        print(f"\n{dataset_name} norm statistics:")
        print(f"Min: {np.min(norms):.2f}")
        print(f"Max: {np.max(norms):.2f}")
        print(f"Mean: {np.mean(norms):.2f}")
        print(f"Median: {np.median(norms):.2f}")
        print(f"Std: {np.std(norms):.2f}")


if __name__ == "__main__":
    for dataset_name in selected_dataset:
        clac_norms(dataset_name)
    plot_norms()
