import math
import random

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def generate_texts_using_t5(existing_texts, num_to_generate, t5_tokenizer, t5_model, dev):
    generated_texts = []

    # Assuming "existing_texts" is a list of existing texts
    for item in existing_texts:
        input_text = f"generate a similar emotion sentence: {item}"
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


if __name__ == '__main__':
    # 加载BART模型和标记器
    model_name = "./dataroot/models/flan-t5-base-tweet-sentiment"
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=128)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 已知情感的句子

    emotional_sentences = [
        "Here in NZ 50% of retailers don’t even have contactless credit card machines like pay-wave which support "
        "Apple Pay.",
        "I feel really sad right now.",
        "This situation makes me angry.",
    ]
    emotion_labels = ["positive", "negative", "neutral"]
    in_text = f"generate a similar sentiment sentence: {emotional_sentences[0]}"
    gen_text = generate_text_with_t5(in_text, t5_tokenizer=tokenizer, t5_model=model, dev=device, num=200)
    print(gen_text)
