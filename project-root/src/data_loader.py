import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# Load configuration from YAML
def load_config(config_path='configs/model_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def filter_renaissance_love(example):
    """Filters dataset for Renaissance love poems."""
    return example['age'] == 'Renaissance'

def load_and_prepare_dataset(config):
    """
    Loads, filters, and tokenizes the dataset.
    Returns the tokenized dataset and data collator.
    """
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    max_length = config['generation']['max_length'] # Using generation max_length for tokenization truncate_size

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['content'], truncation=True, max_length=max_length) # Use max_length from config

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    filtered_dataset = dataset['train'].filter(filter_renaissance_love)
    print(f"Original dataset size: {len(dataset['train'])}")
    print(f"Filtered dataset size (Renaissance Love Poems): {len(filtered_dataset)}")

    tokenized_dataset = filtered_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['author', 'content', 'poem name', 'age', 'type']
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenized_dataset, data_collator, tokenizer # Return tokenizer too for later use