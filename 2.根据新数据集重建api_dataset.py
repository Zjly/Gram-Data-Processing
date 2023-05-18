import pandas as pd
from tqdm import tqdm


def create_api_dataset(code):
    """
    根据新数据集重建api_dataset
    :param code:
    :return:
    """
    dataset = pd.read_feather(f"output/1.数据集去重并划分/{code}/total.feather")
    api_dataset = pd.read_feather(f"data/{code}/api.feather")
    api_description_dict = api_dataset.set_index('api')['description'].to_dict()

    # 统计每个api出现的次数
    api_count_dict = {}
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        api_sequence = row['api_sequence'].split(" ")

        for api in api_sequence:
            api_count_dict[api] = api_count_dict.setdefault(api, 0) + 1

    # 创建新数据集
    new_api_dataset = pd.DataFrame(columns=['index', 'api', 'description'])
    index_list = []
    api_list = []
    description_list = []

    # 根据排序后的结果创建新数据集
    api_count_list = sorted(api_count_dict.items(), key=lambda x: x[1], reverse=True)
    index = 0
    for row in api_count_list:
        api = row[0]
        index_list.append(index)
        api_list.append(api)
        description_list.append(api_description_dict[api])

        index += 1

    new_api_dataset['index'] = index_list
    new_api_dataset['api'] = api_list
    new_api_dataset['description'] = description_list
    new_api_dataset.to_feather(f"output/2.根据新数据集重建api_dataset/{code}/api.feather")


if __name__ == '__main__':
    create_api_dataset('java')
    create_api_dataset('python')
