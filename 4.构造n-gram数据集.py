import pandas as pd
from tqdm import tqdm


class Gram:
    def __init__(self, api, gram, gram_type):
        self.api = api
        self.gram = gram
        self.gram_type = gram_type

    def __hash__(self):
        return hash(str(self.api) + " " + str(self.gram) + " " + str(self.gram_type))

    def __eq__(self, other):
        return str(self.api) == str(other.api) and str(self.gram) == str(other.gram) and str(
            self.gram_type) == str(other.gram_type)


def create_api_dataset(code, gram_num):
    """
    构造n-gram数据集
    :param code:
    :param gram_num:
    :return:
    """
    dataset = pd.read_feather(f"output/3.数据集转化为token/{code}/train.feather")

    # 用于统计gram的数量
    gram_count_map = {}

    # 用于构造gram数据集
    api_list = []
    gram_list = []
    gram_type_list = []
    count_list = []

    # 遍历数据集
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        api_sequence = row['api_sequence']

        # 遍历api序列
        for i in range(len(api_sequence)):
            current_api = api_sequence[i]

            for j in range(1, gram_num + 1):
                # 统计当前api前的gram数量
                if i - j >= 0:
                    gram = Gram(current_api, api_sequence[i - j:i + 1], -j)
                    gram_count_map[gram] = gram_count_map.setdefault(gram, 0) + 1

                # 统计当前api后的gram数量
                if i + j < len(api_sequence):
                    gram = Gram(current_api, api_sequence[i:i + j + 1], j)
                    gram_count_map[gram] = gram_count_map.setdefault(gram, 0) + 1

    for gram in gram_count_map:
        api_list.append(gram.api)
        gram_list.append(gram.gram)
        gram_type_list.append(gram.gram_type)
        count_list.append(gram_count_map[gram])

    gram_dataset = pd.DataFrame()
    gram_dataset['api'] = api_list
    gram_dataset['gram'] = gram_list
    gram_dataset['gram_type'] = gram_type_list
    gram_dataset['count'] = count_list
    gram_dataset.to_feather(f"output/4.构造n-gram数据集/{code}/gram.feather")


def sort(code):
    df = pd.read_feather(f"output/4.构造n-gram数据集/{code}/gram.feather")
    gram_df = df.sort_values(by=['api', 'count', 'gram_type'], ascending=[True, False, True])
    gram_df = gram_df.reset_index(drop=True)
    gram_df.to_feather(f"output/4_5.排序数据集/{code}/gram.feather")


if __name__ == '__main__':
    # create_api_dataset('java', 3)
    # create_api_dataset('python', 3)
    sort('java')
    sort('python')
