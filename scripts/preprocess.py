import pandas as pd

# Load the data
df = pd.read_json("/kaggle/input/roleplay-dataset/roleplay_dataset.json")

# Format into alternating chatbot conversation format
with open("train.txt", "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        prompt = row['prompt'].strip()
        response = row['response'].strip()
        f.write(f"User: {prompt}\nBot: {response}\n")
