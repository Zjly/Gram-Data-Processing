import os

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import BartTokenizer, PLBartTokenizer, AutoTokenizer, T5Tokenizer, BartForConditionalGeneration, \
    PLBartForConditionalGeneration, AutoModel, T5ForConditionalGeneration, RobertaTokenizer

data_type = 'java'
model_type = 'CodeT5'

if model_type == 'BART':
    tokenizer = BartTokenizer.from_pretrained("dataset/BART")
    api_tokenizer = BartTokenizer.from_pretrained("dataset/BART")
    model = BartForConditionalGeneration.from_pretrained("dataset/BART")
elif model_type == 'PLBART':
    tokenizer = PLBartTokenizer.from_pretrained("dataset/PLBART")
    api_tokenizer = PLBartTokenizer.from_pretrained("dataset/PLBART")
    model = PLBartForConditionalGeneration.from_pretrained("dataset/PLBART")
elif model_type == 'CodeBERT':
    tokenizer = AutoTokenizer.from_pretrained("dataset/CodeBERT")
    api_tokenizer = AutoTokenizer.from_pretrained("dataset/CodeBERT")
    model = AutoModel.from_pretrained("dataset/CodeBERT")
elif model_type == 'T5':
    tokenizer = T5Tokenizer.from_pretrained("dataset/T5")
    api_tokenizer = T5Tokenizer.from_pretrained("dataset/T5")
    model = T5ForConditionalGeneration.from_pretrained("dataset/T5")
elif model_type == 'CodeT5':
    tokenizer = RobertaTokenizer.from_pretrained("dataset/CodeT5")
    api_tokenizer = RobertaTokenizer.from_pretrained("dataset/CodeT5")
    model = T5ForConditionalGeneration.from_pretrained("dataset/CodeT5")
else:
    raise Exception("Type error")

# 创建新的tokenizer
api_dataset = pd.read_feather(f"output/2.根据新数据集重建api_dataset/{data_type}/api.feather")
api_list = api_dataset['api'].tolist()
api_tokenizer.add_tokens(api_list)


def model_vocab_preprocessing():
    # 获取原始embedding的信息
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if model_type == 'T5':
        old_num_tokens = 32100

    # 新建vocab的embedding
    api_embeddings = nn.Embedding(api_dataset.shape[0], old_embedding_dim)
    for index, row in tqdm(api_dataset.iterrows(), total=api_dataset.shape[0]):
        # 得到api名
        api = row['api']

        # 得到api的编码
        if model_type == 'BART':
            api_tokenized = tokenizer.encode(api)[1:-1]
        elif model_type == 'PLBART':
            api_tokenized = tokenizer.encode(api)[:-1]
        elif model_type == 'CodeBERT':
            api_tokenized = tokenizer.encode(api)[1:-1]
        elif model_type == 'T5':
            api_tokenized = tokenizer.encode(api)[:-1]
        elif model_type == 'CodeT5':
            api_tokenized = tokenizer.encode(api)[1:-1]
        else:
            raise Exception("Type error")

        # 得到新token的embedding表示 其表示为tokenized的平均值
        token_tensor_list = []
        for token_id in api_tokenized:
            token_tensor_list.append(old_embeddings.weight.data[token_id, :])
        new_token_embedding = torch.stack(token_tensor_list).mean(dim=0)

        # 加入到vocab的embedding中
        api_embeddings.weight.data[index, :] = new_token_embedding

    # 为模型更改embedding
    model.resize_token_embeddings(len(tokenizer) + len(api_list))

    # 写入到new_embeddings之中
    new_embeddings = model.get_input_embeddings()
    new_embeddings.weight.data[old_num_tokens:] = api_embeddings.weight.data[:, :]
    model.set_input_embeddings(new_embeddings)

    # 绑定权重
    model.tie_weights()

    # 保存新模型
    path = f"output/5.数据集tokenized/{model_type}/{data_type}"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)


def create_api_dataset():
    api_index_list = []
    api_description_list = []
    for index, row in tqdm(api_dataset.iterrows(), total=api_dataset.shape[0]):
        api_description = row['description']
        if api_description == "":
            api_description = row['api']

        # 得到api_description的编码
        if model_type == 'BART':
            api_description_tokenized = tokenizer.encode(api_description, max_length=128)[1:]
        elif model_type == 'PLBART':
            api_description_tokenized = tokenizer.encode(api_description, max_length=128)
        elif model_type == 'CodeBERT':
            api_description_tokenized = tokenizer.encode(api_description, max_length=128)[1:]
        elif model_type == 'T5':
            api_description_tokenized = tokenizer.encode(api_description, max_length=128)
        elif model_type == 'CodeT5':
            api_description_tokenized = tokenizer.encode(api_description, max_length=128)[1:]
        else:
            raise Exception("Type error")

        api_index_list.append(index + len(tokenizer))
        api_description_list.append(api_description_tokenized)

    api_dataset['index'] = api_index_list
    api_dataset['tokenized_description'] = api_description_list

    api_dataset.to_feather(f"output/5.数据集tokenized/{model_type}/{data_type}/api.feather")


def create_tokenized_dataset(dataset_path, dtype):
    dataset = pd.read_feather(dataset_path)
    new_dataset = pd.DataFrame(columns=['question', 'api_description', 'api_sequence'])

    question_list = []
    api_description_list = []
    api_sequence_list = []

    api_dataset = pd.read_feather(f"output/5.数据集tokenized/{model_type}/{data_type}/api.feather")
    api_description_dict = api_dataset.set_index('index')['tokenized_description'].to_dict()

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        question = row['question']
        api_sequence = row['api_sequence']

        # 得到编码
        if model_type == 'BART':
            question_tokenized = tokenizer.encode(question, max_length=128)[1:]
        elif model_type == 'PLBART':
            question_tokenized = tokenizer.encode(question, max_length=128)
        elif model_type == 'CodeBERT':
            question_tokenized = tokenizer.encode(question, max_length=128)[1:]
        elif model_type == 'T5':
            question_tokenized = tokenizer.encode(question, max_length=128)
        elif model_type == 'CodeT5':
            question_tokenized = tokenizer.encode(question, max_length=128)[1:]
        else:
            raise Exception("Type error")

        api_sequence_tokenized = [api + len(tokenizer) for api in api_sequence]
        api_sequence_tokenized.append(question_tokenized[-1])

        # 对数据进行encode
        question_list.append(question_tokenized)
        api_sequence_list.append(api_sequence_tokenized)

        # 对描述信息进行encode
        api_description_tokenized = []
        for api in api_sequence_tokenized:
            if api != question_tokenized[-1]:
                api_description_tokenized.append(api_description_dict[api])

        api_description_list.append(api_description_tokenized)

    new_dataset['question'] = question_list
    new_dataset['api_sequence'] = api_sequence_list
    new_dataset['api_description'] = api_description_list
    new_dataset.to_feather(f"output/5.数据集tokenized/{model_type}/{data_type}/{dtype}.feather")


if __name__ == '__main__':
    model_vocab_preprocessing()
    create_api_dataset()
    create_tokenized_dataset(f"output/3.数据集转化为token/{data_type}/train.feather", "train")
    create_tokenized_dataset(f"output/3.数据集转化为token/{data_type}/test.feather", "test")
