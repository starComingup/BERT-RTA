import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import *
import torch.utils.data as Data
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class RandomDeleteAndSwap:
    def __init__(self, change_type="delAndSwap"):
        self.change_type = change_type

    def __call__(self, ori, idx, delete_ratio=0.2, swap_ratio=0.2):
        words = list(ori)  # 基于原始文本的副本进行操作，避免修改原数据

        # 随机删除
        words_to_delete = [word for word in words if random.random() > delete_ratio]  # 假设20%的概率删除每个单词
        deleted_text = ' '.join(words_to_delete)

        # 随机交换
        words_for_swap = words.copy()
        for _ in range(int(len(words_for_swap) * swap_ratio)):  # 假设20%的单词会被交换
            if len(words_for_swap) > 1:
                i, j = random.sample(range(len(words_for_swap)), 2)

                words_for_swap[i], words_for_swap[j] = words_for_swap[j], words_for_swap[i]
        swapped_text = ' '.join(words_for_swap)

        return deleted_text, swapped_text, ori

def label_mapping(categories):
    # create LabelEncoder 对象
    label_encoder = LabelEncoder()
    # use LabelEncoder to encoder the categories
    encoded_labels = label_encoder.fit_transform(categories)
    print(f"classes: {label_encoder.classes_}")
    return encoded_labels

def get_data(data_path,text_col_name,label_col_name, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased', train_aug=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path)

    # use a subset for a dataset
    # proportion_to_keep = 0.1
    # sampled_data = []
    # for category, group in train_df.groupby(label_col_name):
    #     category_sample = group.sample(frac=proportion_to_keep, random_state=1)
    #     sampled_data.append(category_sample)
    #
    # content_sampled = pd.concat(sampled_data)
    # content_sampled.reset_index(drop=True, inplace=True)
    # train_df = content_sampled

    labels = label_mapping(train_df[label_col_name])
    # labels = pd.to_numeric(train_df[label_col_name], errors="coerce").astype(int)
    train_labels = np.array([v for v in labels])

    train_text = np.array([v for v in train_df[text_col_name]])

    # 设置随机种子以确保结果的可复现性
    np.random.seed(0)

    # 使用train_test_split函数，test_size=0.3 表示30%的数据作为验证集，其余70%作为训练集
    train_text, test_text, train_labels, test_labels = train_test_split(
        train_text, train_labels, test_size=0.3, random_state=0,
        stratify=train_labels if isinstance(train_labels, np.ndarray) and train_labels.dtype.kind == 'i' else None
    )

    n_labels = max(test_labels) + 1
    print(f"n_labels: {n_labels}")

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[train_labeled_idxs], train_labels[train_labeled_idxs], tokenizer, max_seq_len)
    train_unlabeled_dataset = loader_unlabeled(
        train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, RandomDeleteAndSwap(train_text[train_unlabeled_idxs]))
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs), len(test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0):
    """Split the original training set into labeled training set, unlabeled training set, development set

    Arguments:
        labels {list} -- List of labeles for original training set
        n_labeled_per_class {int} -- Number of labeled data per class
        unlabeled_per_class {int} -- Number of unlabeled data per class
        n_labels {int} -- The number of classes

    Keyword Arguments:
        seed {int} -- [random seed of np.shuffle] (default: {0})

    Returns:
        [list] -- idx for labeled training set, unlabeled training set, development set
    """
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if n_labels == 2:
            # IMDB
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
        elif n_labels == 10:
            # DBPedia
            train_pool = np.concatenate((idxs[:500], idxs[10500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + unlabeled_per_class])
            val_idxs.extend(idxs[-2000:])
        else:
            # Yahoo/AG News
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]), (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        if self.aug is not None:
            u, v, ori = self.aug(self.text[idx], self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori))
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)