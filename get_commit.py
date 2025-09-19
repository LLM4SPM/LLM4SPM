from github import Github
from github import GithubException
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import time
import re

current_dir = os.getcwd()
commit_data = pd.DataFrame(columns=['commit_id', 'commit_message', 'diff'])

# GitHub 访问令牌（可选，用于私有仓库或提高 API 速率限制）
access_token = ''
g = Github(access_token)
headers = {'Authorization': f'token {access_token}'}

# 检查速率限制
remaining, limit = g.rate_limiting
reset_time = g.rate_limiting_resettime
print(f"Remaining requests: {remaining}/{limit}")
print(e)


# 设置超时时间
timeout = httpx.Timeout(30.0)  # 设置超时时间为 30 秒

df = pd.read_csv('123.csv')
repo_commits = []

# 遍历 DataFrame 的值
for row in df.values:
    # print(f"processing {row[0]}")
    # 获取仓库
    owner = row[1]
    repo_name = row[2]
    commit_id = row[4]
    repo_path = owner + '/' + repo_name + ':' + commit_id
    repo_commits.append(repo_path)
    # print(repo_path)


# 异步获取特定 commit 数据的函数
async def fetch_commit(repo_commit, retries=3):
    repo_name, commit_id = repo_commit.split(':')
    url = f'https://api.github.com/repos/{repo_name}/commits/{commit_id}'
    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    commit = response.json()
                    patch = ""
                    for file in commit['files']:
                        if file['filename'].endswith(('.c', '.cpp', '.cc')):
                            patch = patch + "\n\n" + file.get('patch', 'No changes (e.g., binary file or file deletion).')
                    merge_patch = re.sub(r'^\s*\n', '', patch, 2)  # 移除前两行空行
                    return repo_name, commit_id, commit['commit']['message'], merge_patch
                else:
                    print(f"Attempt {attempt + 1}: Error {response.status_code}, {response.text}")
            except httpx.ConnectTimeout:
                print(f"Attempt {attempt + 1}: Timeout for commit {commit_id} from {repo_name}")
            await asyncio.sleep(1)  # 等待 1 秒后重试
        return None


# 多线程运行异步任务
def run_async_tasks(repo_commit):
    return asyncio.run(fetch_commit(repo_commit))


# 使用多线程并发爬取
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(run_async_tasks, repo_commit) for repo_commit in repo_commits]

    for future in futures:
        repo_name, commit_id, message, patch = future.result()
        if message:
            print(f"Current commit: {commit_id} is been processed")
            # print(f"Message: {message}")
            # print(f"Changes: {patch}")
            # print("------")
            commit_data.loc[len(commit_data)] = [commit_id, message, patch]
        else:
            print(f"Error: Commit ID '{commit_id}' in repository '{repo_name}' does not exist. Skipping...")


#     try:
#         repo = g.get_repo(f'{owner}/{repo_name}')
#
#         # 根据 commit id 获取 commit
#         commit = repo.get_commit(commit_id)
#     except GithubException as e:
#         print(f"Error: Commit ID '{commit_id}' does not exist. Skipping...")
#         continue
#
#     # 打印 commit 信息
#     # print("Commit Message:", commit.commit.message)
#
#     # 获取文件变更
#     # print(commit.files)
#     # print(len(commit.files))
#     diff = ""
#     for file in commit.files:
#         if file.filename.endswith(('.c', '.cpp', '.cc')):
#             diff = diff + "\n\n" + file.patch
#             # print(f"File: {file.filename}")
#             # print("Changes:")
#             # print(file.patch)  # 代码变更
#
#     merge_diff = re.sub(r'^\s*\n', '', diff, 2)  # 移除前两行空行
#
#     commit_data.loc[len(commit_data)] = [commit_id, commit.commit.message, merge_diff]
#
#
# # print(commit_data)
commit_data.to_csv(os.path.join(current_dir, 'commit_data.csv'), index=False)
