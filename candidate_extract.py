import pandas as pd
import os

current_dir = os.getcwd()

# 读取CSV文件
df = pd.read_csv('commit_data.csv')
# unique_cve = df['cve'].nunique()
# print(unique_cve)

# owner_repo = df[['commit_message', 'diff']].drop_duplicates()

# 显示前几行数据
# print(df.head())
# print(owner_repo)
# owner_repo.to_csv(os.path.join(current_dir, 'filter.csv'), index=False)

# 用于记录已经出现过的 (col2, col3) 组合
seen = set()
seen_2 = set()

# 用于存储需要保留的行
rows_to_keep = []
rows_2 = []

# # 遍历每一行
# for index, row in df.iterrows():
#     # 获取当前行的 col2 和 col3 的值
#     col2_col3_pair = (row[1], row[2])
#
#     # 如果该组合没有出现过，则保留该行
#     if col2_col3_pair not in seen:
#         rows_to_keep.append(row)
#         seen.add(col2_col3_pair)

df_filtered = df.dropna(subset=['diff'])
df_clean = df_filtered.drop_duplicates(subset=['diff'], keep='first')



# 创建新的 DataFrame
# df_filtered = pd.DataFrame(rows_to_keep)
# df_clean = df.dropna(subset=['diff'])
unique_commit_id = df_clean['commit_id'].nunique()
print(unique_commit_id)
# print(df_filtered)
print(df_clean)

df_cve = pd.read_csv('cve_data.csv')
# print(df_cve)


merged_df = pd.merge(df_cve, df_clean, on='commit_id', how='inner')

for index, row in merged_df.iterrows():
    # 获取当前行的 col2 和 col3 的值
    col2_col3_pair = (row[0], row[5], row[6])

    # 如果该组合没有出现过，则保留该行
    if col2_col3_pair not in seen_2:
        rows_2.append(row)
        seen_2.add(col2_col3_pair)

df_drop = pd.DataFrame(rows_2)
df_drop.to_csv(os.path.join(current_dir, 'final_data.csv'), index=False)
print(df_drop)
unique_cve = df_drop['cve'].nunique()
print(unique_cve)
#
# unique_diff = df_drop['diff'].nunique()
# print(unique_diff)



# print(merged_df)

