import pandas as pd


def show(csv_path: str):

    df = pd.read_csv(csv_path, header=None, names=["Train", "Test", "Acc"])

    REAL_DATASET = "MSR-VTT"

    # 分别计算 fake 和 real 的平均准确率
    df_real = df[df["Test"] == REAL_DATASET]
    real_avg = df_real.groupby("Train")["Acc"]
    # 给列加上列名 Real
    print(real_avg)

    df_fake = df[df["Test"] != REAL_DATASET]
    fake_avg = df_fake.groupby("Train")["Acc"]
    print(fake_avg)

    exit()

    # 将两个 Series 合并
    avg = pd.concat([fake_avg, real_avg], axis=1)
    avg.columns = ["Fake", "Real"]
    # print(avg)

    # 再加上一列平均值
    avg["Avg"] = avg.mean(axis=1)
    # print(avg)

    # 按照平均值降序排序
    avg = avg.sort_values(by="Avg", ascending=False)
    print(avg)

    # # 创建透视表
    # pivot_table = df.pivot(index="Train", columns="Test", values="Acc")
    # print(pivot_table)


show("results/20240910_190011.csv")
show("results/20240909_200635.csv")
