import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
login(token="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")

def chat_LLAMA():
    print("You can start chatting with the model! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # Tokenize the input and set attention mask
        inputs = tokenizer(user_input, return_tensors='pt', padding=True).to(device)
        
        # Generate the output
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,  # Pass attention mask to avoid the warning
                max_length=150, 
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        if response.startswith(user_input):
            response = response[len(user_input):].strip()
        print("Model: ", response)
        print("")

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


model_selection = input("please select model: ")
if model_selection == "1":
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

elif model_selection == "2":
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