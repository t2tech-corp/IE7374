import os
import torch
import yaml
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Assuming data_loader.py and utils are in the same src directory
from data_loader import load_config, load_and_prepare_dataset
from utils.helpers import get_device # Using get_device from helpers

def train_model():
    config = load_config()

    # Load dataset and tokenizer
    tokenized_dataset, data_collator, tokenizer = load_and_prepare_dataset(config)

    model_name = config['model_name']
    output_dir = config['output_dir']
    
    # Ensure output directory exists for Trainer
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType[config['lora']['task_type']] # Convert string to TaskType enum
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        prediction_loss_only=True,
        fp16=config['training']['fp16'] and torch.cuda.is_available(), # Use fp16 if config allows AND CUDA is available
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
        # Save the LoRA adapters after successful training
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuning complete! Model and tokenizer saved to {output_dir}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Attempting to save current model state...")
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Partial model and tokenizer saved to {output_dir}")
        except Exception as save_e:
            print(f"Error saving partial model: {save_e}")

if __name__ == "__main__":
    train_model()