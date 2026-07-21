from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

pairs = [
    {"en": "I love coffee", "fr": "J'aime le café"},
    {"en": "The cat is sleeping", "fr": "Le chat dort"},
    {"en": "She is reading a book", "fr": "Elle lit un livre"},
    {"en": "The weather is nice today", "fr": "Il fait beau aujourd'hui"},
    {"en": "I want to go to Paris", "fr": "Je veux aller à Paris"},
    {"en": "Where is the library", "fr": "Où est la bibliothèque"},
    {"en": "This book is interesting", "fr": "Ce livre est intéressant"},
    {"en": "Please help me", "fr": "S'il vous plaît aidez-moi"},
    {"en": "I like playing football", "fr": "J'aime jouer au football"},
    {"en": "We are learning machine learning", "fr": "Nous apprenons l'apprentissage automatique"},
    {"en": "Hello", "fr": "Bonjour"},
    {"en": "Hi how are you", "fr": "Salut comment ça va"},
    {"en": "Good morning", "fr": "Bonjour"},
    {"en": "Good evening", "fr": "Bonsoir"},
    {"en": "Good night", "fr": "Bonne nuit"},
    {"en": "Thank you", "fr": "Merci"},
    {"en": "Thank you very much", "fr": "Merci beaucoup"},
    {"en": "You are welcome", "fr": "De rien"},
    {"en": "Excuse me", "fr": "Excusez-moi"},
    {"en": "I am fine", "fr": "Je vais bien"},
    {"en": "What is your name", "fr": "Comment vous appelez-vous"},
    {"en": "My name is John", "fr": "Je m'appelle John"},
    {"en": "Where are you going", "fr": "Où allez-vous"},
    {"en": "I am going home", "fr": "Je rentre à la maison"},
    {"en": "Do you speak English", "fr": "Parlez-vous anglais"},
    {"en": "I do not understand", "fr": "Je ne comprends pas"},
    {"en": "Please repeat", "fr": "Répétez s'il vous plaît"},
    {"en": "How much is this", "fr": "Combien cela coûte"},
    {"en": "Where is the restaurant", "fr": "Où est le restaurant"},
    {"en": "I would like some water", "fr": "Je voudrais de l'eau"},
    {"en": "The food is delicious", "fr": "La nourriture est délicieuse"},
    {"en": "I like this place", "fr": "J'aime cet endroit"},
    {"en": "This is my friend", "fr": "C'est mon ami"},
    {"en": "We are students", "fr": "Nous sommes étudiants"},
    {"en": "They are playing football", "fr": "Ils jouent au football"},
    {"en": "The dog is barking", "fr": "Le chien aboie"},
    {"en": "The sun is shining", "fr": "Le soleil brille"},
    {"en": "It is raining today", "fr": "Il pleut aujourd'hui"},
    {"en": "I am studying machine learning", "fr": "J'étudie l'apprentissage automatique"},
    {"en": "Artificial intelligence is interesting", "fr": "L'intelligence artificielle est intéressante"}
]

MODEL_NAME = "t5-small"
OUTPUT_DIR = "./finetuned_t5_en_fr"

dataset = Dataset.from_list(pairs)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess_function(example):
    source_text = "translate English to French: " + example["en"]

    model_inputs = tokenizer(
        source_text,
        max_length=64,
        truncation=True
    )

    labels = tokenizer(
        text_target=example["fr"],
        max_length=64,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, remove_columns=["en", "fr"])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=30,
    save_total_limit=1,
    logging_steps=1,
    save_strategy="epoch",
    predict_with_generate=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuned model saved successfully.")