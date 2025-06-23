
# 🎵 Music Roleplay Chatbot

A GPT-2-based chatbot fine-tuned to simulate roleplay conversations about **music theory and performance**.

## 📊 Project Summary
This project leverages HuggingFace's GPT-2 to generate contextual, informative, and stylistically consistent chatbot responses. Designed for music students and enthusiasts, the bot explains musical concepts through natural, character-driven dialogue.

## 🔧  1. Environment SetupInstallation
```bash
pip install -r requirements.txt

2. Prepare Data
Make sure your dataset is in data/train.txt format (alternating prompt/response pairs).

3. Fine-Tune GPT-2
python scripts/train_gpt2.py
This will:

Load gpt2 from Hugging Face

Tokenize train.txt

Train for 3 epochs (customizable)

Save model to model/rp_model/

4. Launch Web UI
Open ui/index.html in any browser.


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

| Metric            | Value    | Notes                                      |
| ----------------- | -------- | ------------------------------------------ |
| **Perplexity**    | \~22.4   | Acceptable for conversational roleplay     |
| **BLEU Score**    | \~0.38   | Open-domain, Q\&A structure affects metric |
| **User Feedback** | Positive | Good coherence and character retention     |

🧪 Technologies Used
Python & PyTorch

HuggingFace Transformers

Google Colab (for training)

HTML/CSS (for UI)

Pandas & Datasets (for preprocessing)

✨ Features
GPT-2 fine-tuned on roleplay dialogues

In-character music education assistant

Stylish chat interface (CSS based on dark theme)

Offline deployment possible with saved model

🧑‍🎓 Educational Value
This chatbot supports learners by:

Providing accurate explanations of music theory

Maintaining a fun, immersive tone via character roleplay

Reinforcing concepts through interactive questioning

📌 Future Work
Add voice-based input/output

Expand to multi-turn dialogue memory

Evaluate with larger roleplay datasets (e.g., from Reddit, Discord)




🤝 Credits

Kaggle: vampelium/roleplay-dataset

HuggingFace Transformers

Google Colab Pro

#Contributor 
Valentine Kalu
#
