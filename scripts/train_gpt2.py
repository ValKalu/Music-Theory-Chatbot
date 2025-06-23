import pandas as pd
import os
import kagglehub

# Load the dataset
# Use the path obtained from kagglehub.dataset_download
# Assuming 'path' variable from previous cell holds the dataset root directory
json_file_path = os.path.join(path, "roleplay_dataset.json")

# Verify the file exists before trying to read it
if not os.path.exists(json_file_path):
    print(f"Error: JSON file not found at {json_file_path}")
    # Optionally, re-download the dataset here if needed
    path = kagglehub.dataset_download("vampelium/roleplay-dataset")
    json_file_path = os.path.join(path, "roleplay_dataset.json")
    if not os.path.exists(json_file_path):
         raise FileNotFoundError(f"JSON file still not found after re-download at {json_file_path}")


df = pd.read_json(json_file_path)

# Optional: limit rows for faster fine-tuning/testing
# df = df.sample(n=10000, random_state=42)

# Concatenate prompt and response into one text block
df["chat_pair"] = df["prompt"] + " " + df["response"]

# Save to train.txt
with open("train.txt", "w", encoding="utf-8") as f:
    for line in df["chat_pair"]:
        f.write(line.strip() + "\n")
        
with open("train.txt", "w", encoding="utf-8") as f:
    for line in df["chat_pair"]:
        f.write(line.strip() + "\n")
# Ensure necessary libraries are installed
#!pip install datasets transformers torch

from datasets import load_dataset # Ensure this is from the 'datasets' library
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Set the pad token to the eos token for GPT-2
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Method 1: Explicitly load using datasets.load_dataset
# This is the preferred method and *should* return a datasets.DatasetDict
try:
    # ✅ Load dataset with no caching to disk
    # Explicitly setting the type to 'text' and providing the data files
    dataset = load_dataset(
        "text",
        data_files={"train": "/content/train.txt"},
        keep_in_memory=True,
        cache_dir="/content/hf_cache",  # Needed to avoid default root
    )
    print(f"Successfully loaded dataset using datasets.load_dataset. Type: {type(dataset)}")

except Exception as e:
    print(f"Error loading dataset with datasets.load_dataset: {e}")
    print("Attempting to load manually using datasets.Dataset.from_text")
    # Method 2: If Method 1 fails or still results in TextDataset,
    # manually create a Dataset from the text file
    try:
        from datasets import Dataset
        with open("/content/train.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Create a Dataset object from the list of lines
        # The 'text' column name is conventional for text datasets
        dataset = Dataset.from_dict({"text": lines})
        # Wrap it in a DatasetDict for consistency with Trainer
        dataset = {"train": dataset}
        print(f"Successfully created dataset manually. Type: {type(dataset)}")
    except Exception as e_manual:
        print(f"Error creating dataset manually: {e_manual}")
        raise # Re-raise the error if manual creation also fails


# Tokenize function
def tokenize_function(example):
    # Ensure that 'example' has the expected 'text' key from the loaded dataset
    # The 'text' key should be present if loaded using 'text' dataset type
    if "text" not in example:
        raise ValueError("Example dictionary does not contain 'text' key. Check dataset loading.")
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Check the type of the dataset *after* loading/creation
print(f"Final dataset type before tokenization: {type(dataset)}")

# Ensure dataset is a DatasetDict and access the 'train' split for mapping
if isinstance(dataset, dict) and "train" in dataset:
    print("Dataset is a DatasetDict with a 'train' key. Accessing the train split.")
    # Apply the map function to the 'train' split
    tokenized_datasets = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
elif isinstance(dataset, Dataset):
     print("Dataset is a single Dataset object. Applying map directly.")
     # Apply the map function directly if it's a single Dataset
     tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
     )
else:
    # If after both loading attempts, the dataset is still not a DatasetDict or Dataset
    raise TypeError(f"Loaded dataset is not a DatasetDict or a Dataset object after loading/creation. Type: {type(dataset)}. Cannot apply map.")


# Collator for GPT2
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./rp_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=50,
    save_total_limit=2,
    report_to=[],  # ⛔ Prevents logging to wandb or other integrations
)

# Train the model
# Pass the tokenized dataset object to the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets, # Pass the tokenized Dataset or DatasetDict split
    data_collator=data_collator,
)

print("\nStarting training...")
trainer.train()
print("Training finished.")

# Save the fine-tuned model and tokenizer
print("Saving model and tokenizer...")
trainer.save_model("./rp_model")
tokenizer.save_pretrained("./rp_model")
print("Model and tokenizer saved to ./rp_model")

# Show example from the *original* loaded dataset for context (if successful)
if isinstance(dataset, dict) and "train" in dataset and len(dataset["train"]) > 0:
    print("\nExample from original dataset:")
    print(dataset["train"][0])

# Show example from the *tokenized* dataset
if isinstance(tokenized_datasets, Dataset) and len(tokenized_datasets) > 0:
     print("\nExample from tokenized dataset:")
     print(tokenized_datasets[0])