import os

import torch
import pandas as pd
from torch.utils.data import random_split


def remove_dum(code):
    """
    数据集去重并划分
    :param code:
    :return:
    """
    train_set = pd.read_feather(f"data/{code}/train.feather")
    test_set = pd.read_feather(f"data/{code}/test.feather")

    # 合并train和test
    total_set = pd.concat([train_set, test_set], ignore_index=True)

    # 去重
    total_set = total_set.drop_duplicates(subset=['question', 'api_sequence'], keep='first')

    # 得到划分索引
    torch.manual_seed(42)
    train_length = int(len(total_set) * 0.9)
    test_length = len(total_set) - train_length
    train_set, test_set = random_split(dataset=total_set, lengths=[train_length, test_length])

    # 划分数据集 重置索引
    total_set = total_set.reset_index(drop=True)
    train_set = train_set.dataset.iloc[train_set.indices].reset_index(drop=True)
    test_set = test_set.dataset.iloc[test_set.indices].reset_index(drop=True)

    # 保存数据集
    os.makedirs(f"output/1.数据集去重并划分/{code}", exist_ok=True)
    total_set.to_feather(f"output/1.数据集去重并划分/{code}/total.feather")
    train_set.to_feather(f"output/1.数据集去重并划分/{code}/train.feather")
    test_set.to_feather(f"output/1.数据集去重并划分/{code}/test.feather")


def union(code):
    """
    数据集合并
    :param code:
    :return:
    """
    train_set = pd.read_feather(f"data/{code}/train.feather")
    test_set = pd.read_feather(f"data/{code}/test.feather")

    # 合并train和test
    total_set = pd.concat([train_set, test_set], ignore_index=True)

    # 保存数据集
    os.makedirs(f"output/1.数据集去重并划分/{code}", exist_ok=True)
    total_set.to_feather(f"output/1.数据集去重并划分/{code}/total.feather")
    train_set.to_feather(f"output/1.数据集去重并划分/{code}/train.feather")
    test_set.to_feather(f"output/1.数据集去重并划分/{code}/test.feather")


if __name__ == '__main__':
    # remove_dum('java')
    # remove_dum('python')
    union('java')
    union('python')
