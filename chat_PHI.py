import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
login(token="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")


def chat_PHI():
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


# Load microsoft/Phi-3.5-mini-instruct
torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    )
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
chat_PHI()