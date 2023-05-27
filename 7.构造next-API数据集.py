import os

import pandas as pd
from tqdm import tqdm


def create_next_api_dataset(code, type):
    dataset = pd.read_feather(f"output/5.数据集tokenized/CodeT5/{code}/{type}.feather")

    next_api_sequence_list = []
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        api = [row['api_sequence'][0], row['api_sequence'][-1]]
        next_api_sequence_list.append(api)

    dataset['api_sequence'] = next_api_sequence_list

    os.makedirs(f"output/7.构造next-API数据集/{code}", exist_ok=True)
    dataset.to_feather(f"output/7.构造next-API数据集/{code}/{type}.feather")


if __name__ == '__main__':
    create_next_api_dataset('java', 'train')
    create_next_api_dataset('java', 'test')
    create_next_api_dataset('python', 'train')
    create_next_api_dataset('python', 'test')
