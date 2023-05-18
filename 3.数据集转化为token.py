import pandas as pd
from tqdm import tqdm


def trans2token(code, type):
    """
    数据集转化为token
    :param code:
    :param type:
    :return:
    """
    dataset = pd.read_feather(f"output/1.数据集去重并划分/{code}/{type}.feather")
    api_dataset = pd.read_feather(f"output/2.根据新数据集重建api_dataset/{code}/api.feather")
    api_index_dict = api_dataset.set_index('api')['index'].to_dict()

    api_sequence_list = []
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        str_api_sequence = row['api_sequence']

        api_sequences = str_api_sequence.split(" ")
        api_list = []
        for api in api_sequences:
            api_list.append(api_index_dict[api])

        api_sequence_list.append(api_list)

    dataset['api_sequence'] = api_sequence_list
    dataset.to_feather(f"output/3.数据集转化为token/{code}/{type}.feather")


if __name__ == '__main__':
    trans2token('java', 'total')
    # trans2token('java', 'train')
    # trans2token('java', 'test')
    trans2token('python', 'total')
    # trans2token('python', 'train')
    # trans2token('python', 'test')
