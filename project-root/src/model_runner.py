import os
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

# Assuming data_loader.py and utils are in the same src directory
from data_loader import load_config # To load tokenizer and general config
from utils.helpers import get_device

def generate_poem(prompt, config_path='configs/model_config.yaml'):
    config = load_config(config_path)

    model_name = config['model_name']
    output_dir = config['output_dir']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Reconstruct LoRA config for loading adapters
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType[config['lora']['task_type']] # Convert string to TaskType enum
    )
    lora_model = get_peft_model(base_model, lora_config)

    # Load the state dictionary from the .safetensors file
    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        print(f"Error: LoRA adapter file not found at {adapter_path}.")
        print("Please ensure fine-tuning completed successfully and the file was saved to the correct output_dir.")
        return "Error: Model not found."

    lora_state_dict = load_file(adapter_path, device="cpu") # Load to CPU first
    lora_model.load_state_dict(lora_state_dict, strict=False)

    lora_model.eval() # Set model to evaluation mode

    device = get_device()
    lora_model.to(device)

    # Prepare prompt
    prompt_text = prompt if prompt else config['generation']['prompt']
    inputs = tokenizer.encode_plus(
        prompt_text,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=config['generation']['max_length'] # Use max_length from config for input encoding
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    print(f"Prompt: {prompt_text}")

    # Generate text
    output = lora_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=config['generation']['max_length'],
        num_return_sequences=config['generation']['num_return_sequences'],
        no_repeat_ngram_size=config['generation']['no_repeat_ngram_size'],
        repetition_penalty=config['generation']['repetition_penalty'],
        num_beams=config['generation']['num_beams'],
        do_sample=config['generation']['do_sample'],
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Poem:\n", generated_text)

    # Save to samples.txt in outputs folder
    samples_path = os.path.join(config['output_dir'], "samples.txt")
    os.makedirs(os.path.dirname(samples_path), exist_ok=True) # Ensure outputs dir exists
    with open(samples_path, "a") as f:
        f.write(f"--- Prompt: {prompt_text} ---\n")
        f.write(f"{generated_text}\n\n")
    print(f"Generated sample saved to {samples_path}")

    return generated_text

if __name__ == "__main__":
    # Example usage:
    # Use prompt from config or provide a custom one
    generate_poem(prompt=None)
    # generate_poem(prompt="O, my true love, let us wander")