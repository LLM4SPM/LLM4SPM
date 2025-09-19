import pandas as pd
import re
from pathlib import Path

# ====== 配置区 ======
file1 = "/home/linxw/yzu-rza/code-xyf/result/code_msg_ commitbart/test_0.gold.csv"         # 第一个CSV
file2 = "/home/linxw/yzu-rza/code-xyf/data/merged_data.csv"     # 第二个CSV（有desc和cve列）
out_file = "news_with_cve.csv" 
# ====================

df1 = pd.read_csv(file1)

# 若上次已跑过，去掉旧的 matched_cve，避免“第二列变成表头/错列”
if "matched_cve" in df1.columns:
    df1 = df1.drop(columns=["matched_cve"])

# 用“第一个CSV的第二列”当键
col_key = df1.columns[1]

df2 = pd.read_csv(file2)
df2 = df2.rename(columns={c: c.lower() for c in df2.columns})
if not {"desc","cve"}.issubset(df2.columns):
    raise ValueError("第二个CSV需要包含 desc 和 cve 列")

df2["desc"] = df2["desc"].astype(str)
df2["cve"]  = df2["cve"].astype(str)

# 逐个键去 desc 里找，命中取第一个 CVE（大小写不敏感）
keys = (df1[col_key].dropna().astype(str).str.strip().unique())
mapping = {}
for k in keys:
    if not k:
        continue
    pat = re.escape(k)
    mask = df2["desc"].str.contains(pat, case=False, na=False)
    cves = df2.loc[mask, "cve"].dropna().astype(str)
    mapping[k] = cves.iloc[0] if len(cves) else None  # 只保留一个

df1["matched_cve"] = df1[col_key].map(lambda v: mapping.get(str(v).strip()) if pd.notna(v) else None)

# 写出：带表头、不写索引、带 BOM 便于 Excel 识别
Path(out_file).parent.mkdir(parents=True, exist_ok=True)
df1.to_csv(out_file, index=False, header=True, encoding="utf-8-sig")

print(f"完成，已写入：{out_file}（键列：{col_key}，只保留一个CVE）")
