from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# 加载BART模型和标记器
model_name = "./dataroot/models/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 已知情感的句子

emotional_sentences = [
    "I am so happy today!",
    "I feel really sad right now.",
    "This situation makes me angry.",
]

# 所选情感（可以根据需要更改）
desired_emotion = "negative"

# 随机选择一个已知情感的句子作为输入
input_text = f"generate a similar {desired_emotion} emotion sentence: {emotional_sentences[2]}"  # 这里选择第一个句子，你可以根据需要随机选择
# 将文本编码为输入张量
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# 生成文本
with torch.no_grad():
    generated_ids = model.generate(input_ids["input_ids"], max_length=50,
                                   do_sample=True, top_k=100, top_p=0.8, num_return_sequences=10, num_beams=1)

# 解码生成的文本
generated_texts = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]

print("Generated texts:")
for text in generated_texts:
    print(text)
