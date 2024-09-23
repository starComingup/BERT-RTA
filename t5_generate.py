import math
import pandas as pd
import random
import dataset_utils as dru

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.SpamDataset import SpamDataset

random.seed(1)
# def t5_model_fine_tuning(trainDataset, epochs, batch_size, learning_rate):
#     optimizer = optim.AdamW(t5_model.parameters(), lr=learning_rate)
#     balanced_dataset = dru.balance_dataset_by_min_class_count(trainDataset, t5_tokenizer, max_len)
#     dataloader_balanced = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
#     for epoch in range(epochs):
#         total_loss = 0.0
#
#         for batchX in dataloader_balanced:
#             ids = batchX["input_ids"].to(device1)
#             att_mask = batchX["attention_mask"].to(device1)
#             b_labels = batchX['labels'].to(device1)
#
#             optimizer.zero_grad()
#
#             # Forward pass
#             model_outputs = t5_model(input_ids=ids, attention_mask=att_mask, labels=b_labels)
#             loss = model_outputs.loss
#
#             # Backpropagation and optimization
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         average_loss = total_loss / len(dataloader_balanced)
#         print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")
def cal_data_need_to_expand(texts,labels,count):
    max_key = max(count, key=count.get)
    max_value = count[max_key]
    data_to_expand = {}
    for i in range (0,len(texts)):
        text = texts[i]
        label = labels[i]
        if label not in data_to_expand:
            data_to_expand[label] = [text]
        else:
            data_to_expand[label].append(text)
    for key in data_to_expand.keys():
        print(f'data_to_expand label: {key} have {len(data_to_expand[key])} samples')

    major_num = count[max_key]
    print(f'trainDataset major label is {max_key}, have {major_num} samples')
    minority_dict = {}
    for key in count.keys():
        if key != max_key:
            cls_to_expand_num = major_num - count[key]
            minority_dict[key] = cls_to_expand_num
    for key in minority_dict.keys():
        print(f'minority class {key} need to expand {minority_dict[key]}')
    return data_to_expand, minority_dict

def balance_to_major_class_num(old_data_dict, lack_num_dict, data_label_map):
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    max_len = 128
    # Load the T5 model and tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained("./dataroot/models/flan-t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("./dataroot/models/flan-t5-base",
                                               model_max_length=max_len)
    t5_model.to(device1)

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
            data_category = data_label_map[key]
            generated_texts = generate_texts_using_t5(existing_texts, expect_num_for_every_sample, data_category,
                                                      t5_tokenizer=t5_tokenizer, t5_model=t5_model, dev=device1)
            new_texts.extend(generated_texts)

            # Add corresponding labels to new_labels based on the key
            new_labels.extend([key] * len(generated_texts))
        else:
            print(f"Category {key} not found in old_data_dict.")

    return new_texts, new_labels
def generate_texts_using_t5(existing_texts, num_to_generate, data_category, t5_tokenizer, t5_model, dev):
    generated_texts = []

    # Assuming "existing_texts" is a list of existing texts
    for item in existing_texts:
        input_text = f"generate a similar {data_category} sentence to {item}"
        generated_text = generate_text_with_t5(input_text, t5_tokenizer=t5_tokenizer,
                                               t5_model=t5_model, num=num_to_generate, dev=dev)
        generated_texts.extend(generated_text)

    return generated_texts


def generate_text_with_t5(input_text, t5_tokenizer, t5_model, dev, num):
    # Here, you would use your loa ded T5 model to generate new text based on the input_text
    # You can use the model's "generate" method or other appropriate techniques
    # Encode the input text
    ids = t5_tokenizer.encode(input_text, return_tensors="pt").to(dev)

    # Generate text using the T5 model
    with torch.no_grad():
        output_ids = t5_model.generate(
            ids, max_length=128, min_length=5, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=num)

    # Decode the generated text
    generated_texts = [t5_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in output_ids]
    return generated_texts


# if __name__ == '__main__':
#     # 加载BART模型和标记器
#     model_name = "./dataroot/models/flan-t5-base-tweet-sentiment"
#     tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=128)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     # 已知情感的句子
#
#     emotional_sentences = [
#         "Here in NZ 50% of retailers don’t even have contactless credit card machines like pay-wave which support "
#         "Apple Pay.",
#         "I feel really sad right now.",
#         "This situation makes me angry.",
#     ]
#     emotion_labels = ["positive", "negative", "neutral"]
#     in_text = f"generate a similar sentiment sentence: {emotional_sentences[0]}"
#     gen_text = generate_text_with_t5(in_text, t5_tokenizer=tokenizer, t5_model=model, dev=device, num=200)
#     print(gen_text)

def cal_every_minority_data_need_to_expand_num(count, sample_num):
    max_key = max(count, key=count.get)
    max_value = count[max_key]
    every_category_lack_dict = {}
    for key, value in count.items():
        if key != max_key:
            expand_num = int((max_value-value) / sample_num)
            every_category_lack_dict[key] = expand_num
    return every_category_lack_dict

def selected_balance():
    bert_tokenizer = None
    selected_data_path = './dataset/financial sentiment analysis/' + 'selected_text.csv'
    balanced_text_path = './dataset/financial sentiment analysis/' + 't5_selected_sampled_balanced_data.csv'
    dataset = FinancialDataset(tokenizer=bert_tokenizer, max_length=128)

    content = pd.read_csv(selected_data_path)
    selected_texts = content['selected_texts']
    selected_label = content['selected_labels']

    dataset, ignore = dru.load_dataset(dataset, split_ratio=1)

    class_count = dataset.get_class_counts()

    data_to_expand_dict, category_to_expand_dict = cal_data_need_to_expand(selected_texts,selected_label,class_count)

    synthetic_texts, synthetic_labels = balance_to_major_class_num(data_to_expand_dict, category_to_expand_dict,
                                                                   data_label_map=dataset.get_data_label_map())

    dataset = dru.append_dataset(dataset, synthetic_texts, synthetic_labels)
    data = {'balanced_texts': dataset.texts, 'balanced_labels': dataset.labels}
    df = pd.DataFrame(data)
    df.to_csv(balanced_text_path, index=False)
    print("data synthetic successful!")

def random_balance():
    balanced_text_path = './dataset/financial sentiment analysis/' + 't5_random_sampled_balanced_data.csv'
    selected_data_path = './dataset/financial sentiment analysis/' + 'selected_text.csv'
    bert_tokenizer = None

    dataset = FinancialDataset(tokenizer=bert_tokenizer, max_length=128)

    category_sampled_num_dict = {}
    content = pd.read_csv(selected_data_path)
    selected_labels = content['selected_labels']

    for label in selected_labels:
        if label in category_sampled_num_dict:
            category_sampled_num_dict[label] += 1
        else:
            category_sampled_num_dict[label] = 1

    random_texts = []
    for key in category_sampled_num_dict:
        category_text = dataset.get_text_by_category(key)
        random_texts = random.sample(category_text, category_sampled_num_dict[key])

    class_count = dataset.get_class_counts()
    data_to_expand_dict, category_to_expand_dict = cal_data_need_to_expand(random_texts, selected_labels, class_count)
    synthetic_texts, synthetic_labels = balance_to_major_class_num(data_to_expand_dict, category_to_expand_dict,
                                                                   data_label_map=dataset.get_data_label_map())
    print("data synthetic successful!", len(synthetic_texts), ",", len(synthetic_labels))
    dataset = dru.append_dataset(dataset, synthetic_texts, synthetic_labels)
    data = {'balanced_texts': dataset.texts, 'balanced_labels': dataset.labels}
    print("dataset texts len!", len(dataset.texts), "labels len", len(dataset.labels))
    df = pd.DataFrame(data)
    df.to_csv(balanced_text_path, index=False)


    print("data synthetic write successful!")

if __name__ == '__main__':
    selected_balance()
