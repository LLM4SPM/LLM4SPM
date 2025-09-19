import pandas as pd

# 要检查的 CSV 文件路径
csv_path = "test_0.gold_with_cve.csv"

# 读取文件
df = pd.read_csv(csv_path, dtype=str)

# 只保留有 cve 的记录
df = df.dropna(subset=["cve"])

# 找出重复的 cve id
dup_cve_counts = df["cve"].value_counts()
dup_cve_counts = dup_cve_counts[dup_cve_counts > 1]  # 只保留重复的

if not dup_cve_counts.empty:
    print("发现重复的 CVE ID 及出现次数：")
    print(dup_cve_counts)
else:
    print("没有重复的 CVE ID。")
