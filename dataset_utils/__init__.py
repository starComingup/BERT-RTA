import random
from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from t5Dataset import T5FineTuningDataset


class FileLoader:
    """
    Base class for file loaders.

    - filename: the file location on your computer.
    """

    def __init__(self, filename, has_header=False):
        self.filename = filename
        self.header = has_header

    def read_file(self):
        raise NotImplementedError


class CsvReader(FileLoader):
    def read_file(self):
        """
         Read the csv file.

        Returns:
        - content: The file content.
        """
        if self.header:
            content = pd.read_csv(self.filename, encoding="utf-8", encoding_errors="replace")
        else:
            content = pd.read_csv(self.filename, header=None, encoding="utf-8", encoding_errors="replace")
        return content


class JsonReader(FileLoader):
    def read_file(self):
        """
        Read the json file.

        Returns:
        - content: The file content.
        """
        content = pd.read_json(self.filename, lines=True)
        return content


def load_file(filename, has_header):
    """
    Load the file.

    Args:
    - filename: The file name.

    Returns:
    - data: The file data.
    """
    if filename.endswith('.json'):
        loader = JsonReader(filename, has_header)
    elif filename.endswith('.csv'):
        loader = CsvReader(filename, has_header)
    else:
        raise ValueError("Unsupported file type")
    return loader.read_file()


def load_dataset(
        dataset,
        split_ratio: Union[float, float] = 0.8):
    """
    Load the dataset as train and valid dataset

    :param dataset: the dataset implement form Dataset
    :param split_ratio: split the dataset into train and valid dataset with a ratio. default: 0.8:0.2
    :return: train dataset and valid dataset
    """
    dataset_size = len(dataset)
    train_ratio = split_ratio
    train_size = int(train_ratio * dataset_size)
    valid_size = dataset_size - train_size

    train_indices, valid_indices = random_split(range(dataset_size), [train_size, valid_size])

    train_dataset = type(dataset)(
        tokenizer=dataset.tokenizer,
        max_length=dataset.max_length
    )
    train_dataset.texts = [dataset.texts[i] for i in train_indices]
    train_dataset.labels = dataset.labels[train_indices]

    valid_dataset = type(dataset)(
        tokenizer=dataset.tokenizer,
        max_length=dataset.max_length
    )
    valid_dataset.texts = [dataset.texts[i] for i in valid_indices]
    valid_dataset.labels = dataset.labels[valid_indices]

    return train_dataset, valid_dataset


def balance_dataset_by_min_class_count(dataset, tokenizer, max_len):
    class_counts = dataset.get_class_counts()
    min_class_count = min(class_counts.values())
    balanced_texts = []
    balanced_labels = []

    for label in class_counts.keys():
        class_indices = [i for i, l in enumerate(dataset.labels) if l.item() == label]
        sampled_indices = random.sample(class_indices, min_class_count)
        balanced_texts.extend([dataset.texts[i] for i in sampled_indices])
        balanced_labels.extend([dataset.labels[i] for i in sampled_indices])

    balanced_dataset = T5FineTuningDataset(balanced_texts, balanced_labels, tokenizer=tokenizer, max_length=max_len)
    return balanced_dataset


def append_dataset(dataset, new_texts, new_labels):
    # append new data and label to train dataset
    dataset.texts.extend(new_texts)
    new_label_tensors = torch.tensor(new_labels)
    dataset.labels = torch.cat((dataset.labels, new_label_tensors), dim=0)
    return dataset
