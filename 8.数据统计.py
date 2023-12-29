import numpy as np
import pandas as pd
from datasets import tqdm


def a():
    api_count = 8482
    df = pd.read_feather("output/4.构造n-gram数据集/change/python/gram.feather")

    array = [0] * api_count
    matrix = np.zeros((api_count, api_count), dtype=int)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['gram_type'] == 'previous' and row['gram_value'] == 1:
            matrix[row['api']][row['gram'][0]] += row['count']
            array[row['api']] += row['count']

    ratio = np.zeros((api_count, api_count), dtype=float)
    max_ratio = [0] * api_count
    for i in range(api_count):
        if array[i] == 0:
            continue

        for j in range(api_count):
            ratio[i][j] = matrix[i][j] / array[i]
            max_ratio[i] = max(max_ratio[i], matrix[i][j] / array[i])

    sum = [0] * 5
    count = [0] * 5
    for i in range(len(ratio)):
        if array[i] < 100:
            continue

        row = sorted(ratio[i], reverse=True)
        for j in range(5):
            sum[j] += row[j]
            count[j] += 1

    for i in range(5):
        print(sum[i] / count[i])


def b():
    api_count = 13054
    # api_count = 8482
    # df = pd.read_feather("output/4.构造n-gram数据集/change/python/gram.feather")
    df = pd.read_feather("output/4.构造n-gram数据集/change/java/gram.feather").head(200000)

    array = np.zeros((api_count, 5), dtype=int)
    ratio = np.zeros((api_count, 5, 5), dtype=float)
    ratio_index = np.zeros((api_count, 5), dtype=int)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['gram_type'] == 'previous':
            array[row['api']][row['gram_value'] - 1] += row['count']

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        api = row['api']
        api_index = ratio_index[api][row['gram_value'] - 1]
        if row['gram_type'] == 'previous' and api_index < 5:
            ratio[api][row['gram_value'] - 1][api_index] = row['count'] / array[api][row['gram_value'] - 1]
            ratio_index[api][row['gram_value'] - 1] += 1

    sum = [0, 0, 0, 0, 0]
    count = [0, 0, 0, 0, 0]
    for i in range(api_count):
        for j in range(5):
            if array[i][j] < 10:
                continue

            sum[j] += ratio[i][j]
            count[j] += 1

    for i in range(5):
        print(sum[i] / count[i])


if __name__ == '__main__':
    b()
