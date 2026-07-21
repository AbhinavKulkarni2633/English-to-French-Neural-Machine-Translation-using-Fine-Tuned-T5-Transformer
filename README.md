# 🌍 English-to-French Neural Machine Translation using Fine-Tuned T5 Transformer

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

This project implements an **English-to-French Neural Machine Translation (NMT)** system using Google's **T5-small Transformer** model. The model is fine-tuned on a custom English–French parallel corpus and can accurately translate English sentences into French through an interactive command-line interface.

The project demonstrates the complete workflow of transformer-based machine translation, including dataset preparation, model fine-tuning, model saving, and inference using the Hugging Face Transformers library.

---

## 🎯 Objectives

- Fine-tune a pre-trained **T5-small** model for machine translation.
- Translate English sentences into French.
- Build a simple and interactive translation system.
- Demonstrate the practical application of transformer-based sequence-to-sequence models.

---

## 🚀 Features

- ✅ Fine-tuned T5-small Transformer
- ✅ English → French Neural Machine Translation
- ✅ Interactive command-line translator
- ✅ Custom English–French parallel dataset
- ✅ Built using Hugging Face Transformers
- ✅ Simple and modular implementation

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Hugging Face Transformers | Pre-trained Transformer Models |
| Hugging Face Datasets | Dataset Creation |
| T5-small | Sequence-to-Sequence Transformer Model |

---

## 🧠 Model Architecture

The project uses Google's **T5-small** (Text-to-Text Transfer Transformer) model.

The model is fine-tuned specifically for the task:

```
English Sentence
        │
        ▼
Tokenizer
        │
        ▼
Fine-Tuned T5-small
        │
        ▼
French Translation
```

---

## 📂 Dataset

A custom English–French parallel corpus was created using the Hugging Face **datasets** library.

Each sample consists of:

- English sentence
- Corresponding French translation

Example:

| English | French |
|----------|---------|
| Hello | Bonjour |
| I love coffee | J'aime le café |
| The cat is sleeping | Le chat dort |
| I want to go to Paris | Je veux aller à Paris |

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | T5-small |
| Learning Rate | 5e-5 |
| Batch Size | 4 |
| Epochs | 30 |
| Max Sequence Length | 64 |
| Optimizer | AdamW (default Trainer optimizer) |

---

## 📁 Project Structure

```
English-to-French-Neural-Machine-Translation-using-Fine-Tuned-T5-Transformer/
│
├── train_t5.py              # Fine-tunes the T5 model
├── test_t5.py               # Performs inference using the trained model
├── requirements.txt         # Project dependencies
├── README.md
├── LICENSE
├── .gitignore
└── images/
    └── sample_output.png
```

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/AbhinavKulkarni2633/English-to-French-Neural-Machine-Translation-using-Fine-Tuned-T5-Transformer.git
```

Navigate into the project directory

```bash
cd English-to-French-Neural-Machine-Translation-using-Fine-Tuned-T5-Transformer
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run:

```bash
python train_t5.py
```

The fine-tuned model will be saved locally inside:

```
finetuned_t5_en_fr/
```

---

## 🌐 Testing the Model

Run:

```bash
python test_t5.py
```

Example:

```
Enter English text:
Hello

French translation:
Bonjour
```

---

## 📸 Sample Output

> *(Add your screenshot inside the `images` folder as `sample_output.png`.)*

```markdown
![Sample Output](images/sample_output.png)
```

---

## 🔮 Future Improvements

- Train using a significantly larger English–French dataset.
- Improve translation quality using T5-base or T5-large.
- Develop a web interface using Streamlit.
- Support multiple languages.
- Evaluate translation quality using BLEU scores.
- Deploy the model using Hugging Face Spaces.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Transformer-based Natural Language Processing
- Sequence-to-Sequence Learning
- Fine-tuning Pre-trained Language Models
- Tokenization and Text Preprocessing
- Hugging Face Transformers Library
- Neural Machine Translation
- Model Saving and Inference

---

## 👨‍💻 Author

**Abhinav Kulkarni**

B.E. Electronics and Communication Engineering  
KLE Technological University, Hubballi

GitHub: https://github.com/AbhinavKulkarni2633

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful, consider giving it a star!
