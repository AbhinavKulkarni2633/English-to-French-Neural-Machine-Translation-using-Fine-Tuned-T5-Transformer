from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "./finetuned_t5_en_fr"

print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

print("Model loaded successfully.")
print("Type an English sentence to translate.")
print("Type 'exit' to quit.\n")

while True:
    user_text = input("Enter English text: ").strip()

    if user_text.lower() == "exit":
        print("Exiting translator.")
        break

    if not user_text:
        print("Please enter some text.\n")
        continue

    input_text = "translate English to French: " + user_text

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("French translation:", translated_text)
    print()