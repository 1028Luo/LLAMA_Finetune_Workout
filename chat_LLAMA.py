import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
login(token="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")

def chat_LLAMA():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_input},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        print(output[0]['generated_text'])



 # Load Llama model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if it does not exist
if tokenizer.pad_token is None:
    # You can either set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Enable GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
chat_LLAMA()