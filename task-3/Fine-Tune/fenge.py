import os
import random


def split_dataset(testdata_path, train_file, test_file, train_ratio=0.9):
    # 获取所有子目录名（提交 ID）
    commit_ids = [name for name in os.listdir(testdata_path) if os.path.isdir(os.path.join(testdata_path, name))]

    # 随机打乱提交 ID
    random.shuffle(commit_ids)

    # 计算分割点
    split_index = int(len(commit_ids) * train_ratio)

    # 切分训练集和测试集
    train_ids = commit_ids[:split_index]
    test_ids = commit_ids[split_index:]

    # 将训练集和测试集写入文件
    with open(train_file, 'w', encoding='utf-8') as train_f:
        for commit_id in train_ids:
            train_f.write(commit_id + '\n')

    with open(test_file, 'w', encoding='utf-8') as test_f:
        for commit_id in test_ids:
            test_f.write(commit_id + '\n')


# 设置路径和文件名
testdata_path = '../../myproject/code/testdata/'
train_file = './dataset/data_split/new_commit_time_train.txt'
test_file = './dataset/data_split/new_commit_time_test.txt'

# 调用函数进行数据分割
split_dataset(testdata_path, train_file, test_file)
