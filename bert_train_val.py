import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
def train(model, device, dataloader, optimizer, criterion, is_multi):
    model.train()
    all_predictions = []
    all_labels = []
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
        all_predictions.extend(y_pred)
        all_labels.extend(y_true)

        loss.backward()
        optimizer.step()

    if is_multi:
        # 计算准确率和召回率
        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)
    else:
        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

    return total_loss / len(dataloader), accuracy, recall, f1


def validate(model, device, dataloader, criterion, is_multi):
    model.eval()
    acc = 0.0
    re = 0.0
    f1_sc = 0.0
    total_loss = 0.0
    all_predictions = []
    all_labels = []
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
            all_predictions.extend(y_pred)
            all_labels.extend(y_true)

    if is_multi:
        # 计算准确率和召回率
        acc += accuracy_score(all_labels, all_predictions)
        re += recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
        f1_sc += f1_score(all_labels, all_predictions, average='weighted', zero_division=1)
    else:
        acc += accuracy_score(all_labels, all_predictions)
        re += recall_score(all_labels, all_predictions, average='weighted')
        f1_sc += f1_score(all_labels, all_predictions, average='weighted')
    cnf_matrix = confusion_matrix(np.array(all_labels), np.array(all_predictions))

    return total_loss / len(dataloader), acc, re, f1_sc, cnf_matrix
