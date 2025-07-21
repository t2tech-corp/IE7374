# project-root/src/model_runner.py

import os
import torch
import yaml
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

# Determine project_root based on the current file's location
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports
from src.data_loader import load_config
from utils.helpers import get_device

def generate_poem(prompt=None, config_path=None):
    # If config_path is not provided (e.g., when called from __main__ or another script),
    # construct its absolute path relative to project_root.
    if config_path is None:
        config_path = os.path.join(project_root, 'configs', 'model_config.yaml')

    config = load_config(config_path=config_path)

    model_name = config['model_name']
    output_dir = config['output_dir'] # This is likely relative from config

    # Resolve output_dir to an absolute path for loading
    full_output_dir = os.path.join(project_root, output_dir)

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
        task_type=TaskType[config['lora']['task_type']]
    )
    lora_model = get_peft_model(base_model, lora_config)
    
    # Load the state dictionary from the .safetensors file
    adapter_path = os.path.join(full_output_dir, "adapter_model.safetensors") # Use full path

    if not os.path.exists(adapter_path):
        print(f"Error: LoRA adapter file not found at {adapter_path}.")
        print("Please ensure fine-tuning completed successfully and the file was saved to the correct output_dir.")
        return "Error: Model not found."

    lora_state_dict = load_file(adapter_path, device="cpu")
    lora_model.load_state_dict(lora_state_dict, strict=False)

    lora_model.eval()

    device = get_device()
    lora_model.to(device)

    # Prepare prompt
    prompt_text = prompt if prompt is not None else config['generation']['prompt'] # Handle prompt=None explicitly
    inputs = tokenizer.encode_plus(
        prompt_text,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=config['generation']['max_length']
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

    samples_path = os.path.join(full_output_dir, "samples.txt") # Use full path
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    with open(samples_path, "a") as f:
        f.write(f"--- Prompt: {prompt_text} ---\n")
        f.write(f"{generated_text}\n\n")
    print(f"Generated sample saved to {samples_path}")

    return generated_text

if __name__ == "__main__":
    # When running model_runner.py directly, check for command-line prompt
    if len(sys.argv) > 1:
        custom_prompt = sys.argv[1]
        generate_poem(prompt=custom_prompt)
    else:
        generate_poem(prompt=None)