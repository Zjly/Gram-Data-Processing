import os

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def create_matrix(code):
    """
    构造token-API共现矩阵
    :param code:
    :return:
    """
    dataset = pd.read_feather(f"output/5.数据集tokenized/remove_dup/CodeT5/{code}/train.feather")

    matrix = np.zeros((13053, 32100), dtype=int)

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        question = row['question'][:-1]
        api_sequence = row['api_sequence'][:-1]

        for token in question:
            for api in api_sequence:
                matrix[api - 32100][token] += 1

    os.makedirs('output/6.构造token-API共现矩阵', exist_ok=True)
    csr_matrix = sparse.csr_matrix(matrix)
    sparse.save_npz(f"output/6.构造token-API共现矩阵/{code}.npz", csr_matrix)

def print_matrix(code):
    matrix = sparse.load_npz(f"output/6.构造token-API共现矩阵/remove_dup/{code}.npz").toarray()
    print(matrix)

if __name__ == '__main__':
    # create_matrix('java')
    # create_matrix('python')
    print_matrix('java')
    print_matrix('python')