import math
import random
import sentencepiece
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import dataset_utils as dru
from torch_dataset.AmazonDataset import AmazonDataset
from torch_dataset.RedditDataset import RedditDataset
from torch_dataset.SpamDataset import SpamDataset
from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.TwitterDataset import TwitterDataset
from torch_dataset.YoutubeDataset import YoutubeDataset
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, T5ForConditionalGeneration, \
    T5Tokenizer, AdamW, BertTokenizer, BertForSequenceClassification
from t5_generate import generate_texts_using_t5


def train(model, device, dataloader, optimizer, criterion, is_multi):
    model.train()
    accuracy = 0.0
    recall = 0.0
    f1 = 0.0
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        # 将张量转换为NumPy数组
        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        loss.backward()
        optimizer.step()

        if is_multi:
            # 计算准确率和召回率
            accuracy += accuracy_score(y_true, y_pred)
            recall += recall_score(y_true, y_pred, average='macro', zero_division=1)
            f1 += f1_score(y_true, y_pred, average='macro', zero_division=1)
        else:
            accuracy += accuracy_score(y_true, y_pred)
            recall += recall_score(y_true, y_pred)
            f1 += f1_score(y_true, y_pred)

    return total_loss / len(dataloader), accuracy / len(dataloader), recall / len(dataloader), f1 / len(dataloader)


def validate(model, device, dataloader, criterion, is_multi):
    model.eval()
    accuracy = 0.0
    recall = 0.0
    f1 = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 将张量转换为NumPy数组
            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()

            if is_multi:
                # 计算准确率和召回率
                accuracy += accuracy_score(y_true, y_pred)
                recall += recall_score(y_true, y_pred, average='macro', zero_division=1)
                f1 += f1_score(y_true, y_pred, average='macro', zero_division=1)
            else:
                accuracy += accuracy_score(y_true, y_pred)
                recall += recall_score(y_true, y_pred)
                f1 += f1_score(y_true, y_pred)

    return total_loss / len(dataloader), accuracy / len(dataloader), recall / len(dataloader), f1 / len(dataloader)


def compute_similarity_matrix(embeddings):
    # Compute cosine similarity in batches to save memory
    simi_batch_size = 1000
    num_samples = embeddings.shape[0]
    simi_result = np.zeros((num_samples, num_samples), dtype=np.float32)

    for i in range(0, num_samples, simi_batch_size):
        start = i
        end = min(i + simi_batch_size, num_samples)
        batch_embeddings = embeddings[start:end]
        batch_similarity = cosine_similarity(batch_embeddings, embeddings)
        simi_result[start:end] = batch_similarity

    return simi_result


def find_simi_neighbors(matrix, neighbor_nums):
    return np.argsort(-matrix)[1:neighbor_nums]


def find_low_similar_neighbors(row_indices, col_indices, dataset,
                               major_cls, neighbor_num=6, neighbor_ratio_threshold=3):
    # debug var, delete it when completed.
    diff_count = 0
    category_counts = {}
    # sample_index is a dict contains low similarity text and its highest similarity neighbor(default: 5)
    '''
    Example:
    {24: [330, 518, 1980, 2096, 6779]}
    '''
    sample_index = {}

    for i in range(len(row_indices)):
        row_index = row_indices[i]
        col_index = col_indices[i]

        if not torch.eq(dataset.labels[row_index], dataset.labels[col_index]):
            diff_count += 1

            row_label = dataset.labels[row_index].item()
            col_label = dataset.labels[col_index].item()

            if row_label != major_cls:
                neighbors = find_simi_neighbors(similarities[row_index], neighbor_num)
                # Filter neighbors to keep only those with the same label as row_label
                same_label_neighbors = [item for item in neighbors if dataset.labels[item].item() == row_label]

                # Check if the number of neighbors with the same label is greater than or equal to the threshold
                neighbors_check = len(same_label_neighbors) >= neighbor_ratio_threshold
                if neighbors_check:
                    sample_index[row_index] = same_label_neighbors

            if col_label != major_cls:
                neighbors = find_simi_neighbors(similarities[col_index], neighbor_num)
                same_label_neighbors = [item for item in neighbors if dataset.labels[item].item() == col_label]
                neighbors_check = len(same_label_neighbors) >= neighbor_ratio_threshold
                if neighbors_check:
                    sample_index[col_index] = same_label_neighbors

            category = dataset.labels[row_index].item()
            category_counts[category] = category_counts.get(category, 0) + 1
    print(f'similarity matrix have {diff_count} pair samples')
    for key in category_counts:
        print(f'category_counts: label {key} have {category_counts[key]} samples')
    print(f'sample_index have {len(sample_index)} samples')
    return sample_index


def cal_data_num_to_expand(dataset, samples_to_expand, major_cls):
    data_to_expand = {}
    class_counts = dataset.get_class_counts()
    for value in samples_to_expand.values():
        for index in value:
            label = dataset.labels[index].item()
            if label != major_cls:
                if label not in data_to_expand:
                    data_to_expand[label] = [dataset.texts[index]]
                elif index not in data_to_expand[label]:
                    data_to_expand[label].append(dataset.texts[index])
    for key in data_to_expand.keys():
        print(f'data_to_expand label: {key} have {len(data_to_expand[key])} samples')

    major_num = class_counts[major_cls]
    print(f'trainDataset major label is {major_cls}, have {major_num} samples')
    minority_dict = {}
    for key in class_counts.keys():
        if key != major_cls:
            cls_to_expand_num = major_num - class_counts[key]
            minority_dict[key] = cls_to_expand_num
    for key in minority_dict.keys():
        print(f'minority class {key} need to expand {minority_dict[key]}')
    return data_to_expand, minority_dict


def balance_to_major_class_num(old_data_dict, lack_num_dict):
    new_texts = []
    new_labels = []

    # Iterate through the lack_num_dict to process each category
    for key, value in lack_num_dict.items():
        # Check if the key exists in the old_data_dict
        if key in old_data_dict:
            texts_to_generate_num = value  # Number of texts to generate for this category
            existing_texts = old_data_dict[key]  # Existing texts in this category
            expect_num_for_every_sample = math.floor(texts_to_generate_num / len(existing_texts))
            # Generate new texts using T5 model and add them to new_texts
            # Here, you would use your T5 model to generate new texts based on the existing ones
            generated_texts = generate_texts_using_t5(existing_texts, expect_num_for_every_sample,
                                                      t5_tokenizer=t5_tokenizer, t5_model=t5_model, dev=device0)
            new_texts.extend(generated_texts)

            # Add corresponding labels to new_labels based on the key
            new_labels.extend([key] * len(generated_texts))
        else:
            print(f"Category {key} not found in old_data_dict.")

    return new_texts, new_labels


def t5_model_fine_tuning(trainDataset, epochs, batch_size, learning_rate):
    optimizer = optim.AdamW(t5_model.parameters(), lr=learning_rate)
    balanced_dataset = dru.balance_dataset_by_min_class_count(trainDataset, t5_tokenizer, max_len)
    dataloader_balanced = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0

        for batchX in dataloader_balanced:
            ids = batchX["input_ids"].to(device0)
            att_mask = batchX["attention_mask"].to(device0)
            b_labels = batchX['labels'].to(device0)

            optimizer.zero_grad()

            # Forward pass
            model_outputs = t5_model(input_ids=ids, attention_mask=att_mask, labels=b_labels)
            loss = model_outputs.loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader_balanced)
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")


def bert_model_finetune(loader, finetune_epoch):
    finetune_optimizer = AdamW(bert_model.parameters(), lr=1e-5)
    # train the model
    for num_epoch in range(finetune_epoch):
        total_loss = 0.0
        bert_model.train()
        for train_batch in loader:
            ids = train_batch['input_ids'].to(device0)
            mask = train_batch['attention_mask'].to(device0)
            train_labels = train_batch['labels'].to(device0)
            train_outputs = bert_model(ids, attention_mask=mask, labels=train_labels)
            loss = train_outputs.loss
            loss.backward()
            finetune_optimizer.step()
            finetune_optimizer.zero_grad()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        print(f"Epoch {num_epoch + 1}, Loss: {average_loss:.4f}")


if __name__ == '__main__':
    num_classes = 3
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path_name = "./dataroot/models/bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(model_path_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_path_name, num_labels=num_classes)
    max_len = 128
    batch_size = 128
    dataset = RedditDataset(tokenizer=bert_tokenizer, max_length=max_len)
    trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)
    bert_model.to(device0)

    '''
        reddit 0.04（nei_ratio_threshold = 4)
        spam 0.8
        financial 0.6
        youtube 0.3
        twitter 0.21
        amazon 0.4（待定）
        '''
    threshold = 0.04
    # we advise if you want to find 5 neighbors, you should set neighbor_num as 6.
    neighbor_num = 6
    '''
    pls read the data README to adjust the param
    '''
    major_cls = 1
    # the neighbor ratio, set 3 means the neighbor label same as the samples num should more than 3.
    nei_ratio_threshold = 3
    # Load the T5 model and tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained("./dataroot/models/flan-t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("./dataroot/models/flan-t5-base",
                                               model_max_length=max_len)
    emotion_mapping = {
        0: 'neutral',
        1: 'positive',
        2: 'negative'
    }
    middle_text_check = './result/middle/reddit_middle.csv'
    result_file_path = './result/bert/reddit_result_balanced_finetune.csv'

    dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    finetuneEpoch = 5
    bert_model_finetune(dataloader, finetuneEpoch)

    embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device0)
            attention_mask = batch['attention_mask'].to(device0)
            labels = batch['labels'].to(device0)
            outputs = bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            paragraph_embedding, _ = last_hidden_states.max(dim=1)
            embeddings.append(paragraph_embedding)

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    similarities = compute_similarity_matrix(embeddings)

    # release GPU
    del embeddings
    del dataloader
    torch.cuda.empty_cache()

    # get the samples under the similarity threshold
    similarities = np.array(similarities)

    # Set the lower triangle and diagonal to a high value
    similarities_tri = np.triu(similarities, k=1)

    # Get the indices of elements below the threshold and not equal to zero
    row_indices, col_indices = np.where((similarities_tri < threshold) & (similarities_tri != 0))
    print(f'row_indices have:{len(row_indices)}, col_indices have:{len(col_indices)}')

    #    row_indices1, col_indices1 = np.where((similarities > 0.98) & (similarities != 1))
    #    print(f"{len(row_indices1)} simi are higher than threshold.")
    #    diff_count1 = 0
    #    for i in range(len(row_indices1)):
    #        if not torch.eq(amazonTrainDataset.labels[row_indices1[i]], amazonTrainDataset.labels[col_indices1[i]]):
    #            diff_count1 += 1
    #    print(diff_count1)
    #    print(f"{len(row_indices)} simi are lower than threshold.")

    samples_index_dict = find_low_similar_neighbors(row_indices, col_indices,
                                                    trainDataset, major_cls, neighbor_num, nei_ratio_threshold)

    data_to_expand_dict, minority_lack_num_dict = cal_data_num_to_expand(trainDataset, samples_index_dict, major_cls)

    del similarities

    t5_model.to(device0)

    # model_fine_tuning(trainDataset, t5_num_epoch, t5_batch_size, t5_learning_rate)

    synthetic_texts, synthetic_labels = balance_to_major_class_num(data_to_expand_dict, minority_lack_num_dict)
    data = {'s_texts': synthetic_texts, 's_labels': synthetic_labels}
    df = pd.DataFrame(data)
    df.to_csv(middle_text_check, index=False)

    trainDataset = dru.append_dataset(trainDataset, synthetic_texts, synthetic_labels)

    # Clear GPU memory
    torch.cuda.empty_cache()

    # clear the model grad
    bert_model.zero_grad()
    bert_model.to(device1)

    num_epochs = 50
    lr = 3e-5
    min_lr = 1e-8
    T0 = 10
    T_min = 1
    alpha = 0.95

    optimizer = optim.AdamW(bert_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=1, eta_min=min_lr, last_epoch=-1)

    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validDataset, batch_size=batch_size, shuffle=True)

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    train_recall_list = []
    train_f1_list = []
    valid_loss_list = []
    valid_acc_list = []
    valid_recall_list = []
    valid_f1_list = []

    # judge is multi classes or not
    multi_classes = num_classes > 2
    for epoch in range(num_epochs):
        train_loss, train_acc, train_recall, train_f1 = \
            train(bert_model, device1, train_dataloader, optimizer, criterion, multi_classes)
        val_loss, val_acc, val_recall, val_f1 = validate(bert_model, device1, val_dataloader, criterion, multi_classes)
        scheduler.step(epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")
        epoch_list.append(epoch + 1)
        train_acc_list.append(round(train_acc, 4))
        train_loss_list.append(round(train_loss, 4))
        train_recall_list.append(round(train_recall, 4))
        train_f1_list.append(round(train_f1, 4))
        valid_acc_list.append(round(val_acc, 4))
        valid_loss_list.append(round(val_loss, 4))
        valid_recall_list.append(round(val_recall, 4))
        valid_f1_list.append(round(val_f1, 4))

    # create a blank file to store the index of DataFrame
    columns = ['Epoch', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1', 'Val Loss', 'Val Acc', 'Val Recall',
               'Val F1']
    df = pd.DataFrame(list(zip(epoch_list, train_loss_list, train_acc_list, train_recall_list, train_f1_list,
                               valid_loss_list, valid_acc_list, valid_recall_list, valid_f1_list)), columns=columns)
    # save DataFrame to csv file
    df.to_csv(result_file_path, index=False)