# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
import math
import random
from random import shuffle

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from torch_dataset.SpamDataset import SpamDataset
from torch_dataset.TwitterDataset import TwitterDataset
from torch_dataset.YoutubeDataset import YoutubeDataset
from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.RedditDataset import RedditDataset
from torch_dataset.FakeJobDataset import FakeJobDataset
from data_balance import train, validate
import dataset_utils as dru

random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

# cleaning up text
import re


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if len(clean_line) == 0:
        return clean_line
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################


from nltk.corpus import wordnet


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    # sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            if len(words) != 0:
                a_words = random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))

    # rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            if len(words) != 0:
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

    # rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            if len(words) != 0:
                a_words = random_deletion(words, p_rd)
                augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    return augmented_sentences


def cal_every_minority_data_need_to_expand_num(count, sample_num_dict):
    max_key = max(count, key=count.get)
    max_value = count[max_key]
    every_category_lack_dict = {}
    for key, value in count.items():
        if key != max_key:
            expand_num = int((max_value-value) / sample_num_dict[key])
            every_category_lack_dict[key] = expand_num
    return every_category_lack_dict


def synthetic_samples_for_binary_classes(data1_class, data1, expand_dict):
    alpha_sr = 0.1  # default
    alpha_ri = 0.1
    alpha_rs = 0.1
    alpha_rd = 0.1
    synthe_texts = []
    synthe_labels = []
    num_aug = expand_dict[data1_class]
    for sentence in data1:
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        synthe_texts.extend(aug_sentences)
    data1_len = len(synthe_texts)
    synthe_labels.extend([data1_class] * data1_len)

    return synthe_texts, synthe_labels

def synthetic_samples_for_three_classes(data1_class, data1, data2_class, data2, expand_dict):
    alpha_sr = 0.1  # default
    alpha_ri = 0.1
    alpha_rs = 0.1
    alpha_rd = 0.1
    synthe_texts = []
    synthe_labels = []
    num_aug = expand_dict[data1_class]
    for sentence in data1:
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        synthe_texts.extend(aug_sentences)
    data1_len = len(synthe_texts)
    synthe_labels.extend([data1_class] * data1_len)

    num_aug = expand_dict[data2_class]
    for sentence in data2:
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        synthe_texts.extend(aug_sentences)
    data2_len = len(synthe_texts) - data1_len
    synthe_labels.extend([data2_class] * data2_len)

    return synthe_texts, synthe_labels

def random_balanced():
    balanced_text_path = './dataset/youtube statistic/' + 'eda_random_sampled_balanced_data.csv'
    bert_tokenizer = None

    dataset = YoutubeDataset(tokenizer=bert_tokenizer, max_length=128)
    trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=1)

    category_to_expand = 0
    category_1_text = trainDataset.get_text_by_category(category_to_expand)

    sampled_num = 21
    random_samples = random.sample(category_1_text, sampled_num)

    category_to_expand2 = 1
    category_2_text = trainDataset.get_text_by_category(category_to_expand2)
    sampled_num2 = 166
    random_samples2 = random.sample(category_2_text, sampled_num2)


    class_count = trainDataset.get_class_counts()
    # category_to_expand_dict = cal_every_minority_data_need_to_expand_num(class_count, sampled_num)
    # synthetic_texts, synthetic_labels = synthetic_samples_for_binary_classes(category_to_expand,
    #                                                       random_samples, category_to_expand_dict)
    sample_num_dict = {category_to_expand: sampled_num, category_to_expand2: sampled_num2}
    category_to_expand_dict = cal_every_minority_data_need_to_expand_num(class_count, sample_num_dict)
    synthetic_texts, synthetic_labels = synthetic_samples_for_three_classes(category_to_expand, random_samples,
                                                                            category_to_expand2, random_samples2,
                                                                            category_to_expand_dict)
    print("data synthetic successful!", len(synthetic_texts), ",", len(synthetic_labels))

    trainDataset = dru.append_dataset(trainDataset, synthetic_texts, synthetic_labels)
    data = {'balanced_texts': trainDataset.texts, 'balanced_labels': trainDataset.labels}
    df = pd.DataFrame(data)
    df.to_csv(balanced_text_path, index=False)
    print("data synthetic write successful!")

def selected_balanced():
    bert_tokenizer = None
    selected_data_path = './dataset/twitter sentiment analysis/'+'selected_text.csv'
    aug_text_path = './dataset/twitter sentiment analysis/' + 'eda_selected.csv'
    balanced_text_path = './dataset/twitter sentiment analysis/' + 'eda_selected_sampled_balanced_data.csv'
    dataset = TwitterDataset(tokenizer=bert_tokenizer, max_length=128)

    content = pd.read_csv(selected_data_path)
    # selected_texts = content['selected_texts']

    dataset, ignore = dru.load_dataset(dataset,split_ratio=1)

    class_count = dataset.get_class_counts()

    category_to_expand = 0
    category_to_expand2 = 2
    selected_texts = content[content['selected_labels'] == category_to_expand]['selected_texts']
    selected_texts2 = content[content['selected_labels'] == category_to_expand2]['selected_texts']
    sample_num_dict = {category_to_expand:len(selected_texts), category_to_expand2:len(selected_texts2)}
    category_to_expand_dict = cal_every_minority_data_need_to_expand_num(class_count, sample_num_dict)
    synthetic_texts, synthetic_labels = synthetic_samples_for_three_classes(category_to_expand,selected_texts,
                                                                            category_to_expand2,selected_texts2,
                                                                            category_to_expand_dict)
    # category_to_expand = 1
    # selected_texts = content[content['selected_labels'] == category_to_expand]['selected_texts']
    # sample_num_dict = {category_to_expand: len(selected_texts)}
    # category_to_expand_dict = cal_every_minority_data_need_to_expand_num(class_count, sample_num_dict)
    # synthetic_texts, synthetic_labels = synthetic_samples_for_binary_classes(category_to_expand, selected_texts, category_to_expand_dict)
    print("data synthetic successful!", len(synthetic_texts), ",", len(synthetic_labels))
    data0 = {'balanced_texts': synthetic_texts, 'balanced_labels': synthetic_labels}
    df = pd.DataFrame(data0)
    df.to_csv(aug_text_path, index=False)
    dataset = dru.append_dataset(dataset, synthetic_texts, synthetic_labels)
    data = {'balanced_texts': dataset.texts, 'balanced_labels': dataset.labels}
    df = pd.DataFrame(data)
    df.to_csv(balanced_text_path, index=False)
    print("data synthetic write successful!")

if __name__ == '__main__':
    selected_balanced()

