import pandas as pd

file_path = "/home/linxw/yzu-rza/code-xyf/result/code_msg_ commitbart/test_0.gold.csv"       # 原始文件
out_path = "/home/linxw/yzu-rza/code-xyf/result/code_msg_ commitbart/news_test_0.gold.csv"

# 读取时告诉 pandas 这文件没有表头
df = pd.read_csv(file_path, header=None)

# 添加表头
df.columns = ["id", "desc"]

# 保存回去
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"完成！已生成带表头文件：{out_path}")
