from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os # Import os module

# Load model
# Before loading, let's verify if the directory and expected files exist
model_dir = "./musicbot_model"

if not os.path.exists(model_dir):
    print(f"Error: Model directory '{model_dir}' not found.")
    # Consider re-running the training cell if the directory is missing
else:
    # Check for key tokenizer files
    vocab_file_exists = os.path.exists(os.path.join(model_dir, "vocab.json"))
    merges_file_exists = os.path.exists(os.path.join(model_dir, "merges.txt"))
    tokenizer_config_exists = os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))

    if not (vocab_file_exists and merges_file_exists and tokenizer_config_exists):
        print(f"Warning: Not all required tokenizer files found in '{model_dir}'.")
        print(f"  vocab.json exists: {vocab_file_exists}")
        print(f"  merges.txt exists: {merges_file_exists}")
        print(f"  tokenizer_config.json exists: {tokenizer_config_exists}")
        print("Please ensure the previous cell executed successfully and saved the tokenizer files.")
    else:
        print(f"Required tokenizer files found in '{model_dir}'. Attempting to load.")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
            model = GPT2LMHeadModel.from_pretrained(model_dir)
            print("Model and tokenizer loaded successfully.")

            # Generate response function
            def chat_with_bot(user_input):
                prompt = f"User: {user_input}\nBot:"
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                # Ensure input_ids is on the same device as the model if using GPU
                if torch.cuda.is_available():
                    model.to('cuda')
                    input_ids = input_ids.to('cuda')

                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=100,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                return response.split("Bot:")[-1].strip()

            # Test
            print("\nTesting chatbot:")
            response = chat_with_bot("What is a major scale?")
            print(f"Bot: {response}")

        except Exception as e:
            print(f"An error occurred while loading the model or tokenizer: {e}")
            print("Please check the files in the directory and verify they are not corrupted.")