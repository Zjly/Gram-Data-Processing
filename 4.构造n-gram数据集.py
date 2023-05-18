import pandas as pd
from tqdm import tqdm


def create_api_dataset(code, gram_num):
    """
    构造n-gram数据集
    :param code:
    :param gram_num:
    :return:
    """
    dataset = pd.read_feather(f"output/3.数据集转化为token/{code}/train.feather")

    api_list = []
    gram_list = []
    gram_type_list = []

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        api_sequence = row['api_sequence']

        for i in range(len(api_sequence)):
            current_api = api_sequence[i]

            for j in range(1, gram_num + 1):
                if i - j >= 0:
                    api_list.append(current_api)
                    gram_list.append(api_sequence[i - j:i + 1])
                    gram_type_list.append(-j)

                if i + j < len(api_sequence):
                    api_list.append(current_api)
                    gram_list.append(api_sequence[i:i + j + 1])
                    gram_type_list.append(j)

    gram_dataset = pd.DataFrame()
    gram_dataset['api'] = api_list
    gram_dataset['gram'] = gram_list
    gram_dataset['gram_type'] = gram_type_list
    gram_dataset.to_feather(f"output/4.构造n-gram数据集/{code}/gram.feather")


if __name__ == '__main__':
    create_api_dataset('java', 3)
    create_api_dataset('python', 3)
