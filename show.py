import pandas as pd


result_path = "results/20240909_200635.csv"

df = pd.read_csv(result_path, header=None, names=["Train", "Test", "Acc"])

# 对于每行，添加一个新列，计算该行的平均值
df["Mean"] = df.mean(axis=1)

# 创建透视表
pivot_table = df.pivot(index="Train", columns="Test", values="Acc")

# 输出表格
print(pivot_table)
