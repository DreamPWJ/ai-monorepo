from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    # 加载模型和分词器
    model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"  # Llama3模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 输入文本
    input_text = "Some input text to translate."

    # 分词和编码
    input_tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.encode(input_tokens, return_tensors="pt")

    # 使用模型进行翻译
    translated_ids = model.generate(input_ids)

    # 解码翻译后的文本
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    # 输出翻译结果
    print("Input text:", input_text)
    print("Translated text:", translated_text)


if __name__ == "__main__":
    main()