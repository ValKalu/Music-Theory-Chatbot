
# 🎵 Music Roleplay Chatbot

A GPT-2-based chatbot fine-tuned to simulate roleplay conversations about **music theory and performance**.

## 📊 Project Summary
This project leverages HuggingFace's GPT-2 to generate contextual, informative, and stylistically consistent chatbot responses. Designed for music students and enthusiasts, the bot explains musical concepts through natural, character-driven dialogue.

## 🔧 Installation
```bash
pip install -r requirements.txt

🚀 Running the Project

Preprocess your data into train.txt (one prompt-response per pair)

Run training: python scripts/train_gpt2.py

python scripts/train_gpt2.py

Serve ui/index.html locally in a browser to chat with the bot

Serve ui/index.html locally in a browser to chat with the bot

music-roleplay-chatbot/
├── data/train.txt              # Training data (text file)
├── model/rp_model/            # Fine-tuned model
├── ui/index.html              # HTML-based chat interface
├── ui/style.css               # Interface styling (CSS)
├── scripts/train_gpt2.py      # Model training logic
├── evaluation/metrics.md      # Performance and feedback
├── requirements.txt           # Project dependencies
└── README.md

🔬 Model Info

Base: GPT-2

Dataset: Roleplay-based music theory dataset (Kaggle)

Epochs: 3

Perplexity: ~22.4

BLEU: ~0.38

📊 Evaluation

Metric

Value

Notes

Perplexity

~22.4

Indicates coherent generation

BLEU

~0.38

Acceptable for creative text

Feedback

Positive

Music students reported usefulness

🤝 Credits

Kaggle: vampelium/roleplay-dataset

HuggingFace Transformers

Google Colab Pro

#Contributor 
Valentine Kalu
#
