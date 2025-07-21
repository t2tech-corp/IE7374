# project-root/src/train.py

import os
import torch
import yaml
import sys
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Determine project_root based on the current file's location
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from 'src' to get to 'project-root'
project_root = os.path.abspath(os.path.join(current_file_dir, os.pardir))

# Add project_root to sys.path if not already there, for importing other modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import data_loader and helpers
from src.data_loader import load_config, load_and_prepare_dataset
from utils.helpers import get_device


def train_model():
    # Construct the absolute path to the config file
    config_path = os.path.join(project_root, 'configs', 'model_config.yaml')
    config = load_config(config_path=config_path) # Pass the explicit path

    # Load dataset and tokenizer
    tokenized_dataset, data_collator, tokenizer = load_and_prepare_dataset(config)

    model_name = config['model_name']
    output_dir = config['output_dir']
    
    # Ensure output directory exists for Trainer
    full_output_dir = os.path.join(project_root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType[config['lora']['task_type']]
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure TrainingArguments
    training_args = TrainingArguments(
        output_dir=full_output_dir, # Use the full, resolved output path
        overwrite_output_dir=True,
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        logging_dir=os.path.join(full_output_dir, "logs"), # Log dir also relative to full_output_dir
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        prediction_loss_only=True,
        fp16=config['training']['fp16'] and torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    try:
        trainer.train()
        model.save_pretrained(full_output_dir) # Save to the full, resolved path
        tokenizer.save_pretrained(full_output_dir) # Save tokenizer to the full, resolved path
        print(f"Fine-tuning complete! Model and tokenizer saved to {full_output_dir}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Attempting to save current model state...")
        try:
            model.save_pretrained(full_output_dir)
            tokenizer.save_pretrained(full_output_dir)
            print(f"Partial model and tokenizer saved to {full_output_dir}")
        except Exception as save_e:
            print(f"Error saving partial model: {save_e}")

if __name__ == "__main__":
    train_model()