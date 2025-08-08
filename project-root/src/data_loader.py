import os
import yaml
#from datasets import load_dataset
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import re

# --- Local File Loading Function ---
def load_local_poetry_file(file_path):
    """
    Loads poems from a local .txt file.
    Assumes poems are separated by '<|endofpoem|>'.
    Assigns placeholder metadata for consistency.
    """
    poems_list = []
    delimiter = "<|endofpoem|>" # Define the delimiter explicitly
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # Split the text by the delimiter
        # re.escape() ensures that any special regex characters in the delimiter are treated literally.
        # .strip() on the full_text first to remove any leading/trailing whitespace before the first/after the last poem.
        raw_poems_segments = re.split(re.escape(delimiter), full_text.strip())

        for i, poem_content in enumerate(raw_poems_segments):
            cleaned_content = poem_content.strip() # Strip whitespace from each segment
            if cleaned_content: # Only add if the content is not empty after stripping (handles empty segments from split)
                poems_list.append({
                    'content': cleaned_content,
                    'author': 'Unknown/Various (Local File)', # Placeholder for local file
                    'poem name': f"Local Poem {i+1}", # Placeholder for local file
                    'age': 'Renaissance', # Assume content is already filtered for this
                    'type': 'Love'        # Assume content is already filtered for this
                })
        print(f"Loaded {len(poems_list)} poems from local file: {file_path}")
    except FileNotFoundError:
        print(f"Error: Local poetry file not found at {file_path}. Please ensure it exists.")
    except Exception as e:
        print(f"Error loading local poetry file: {e}")
    return poems_list
    
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

    # Define the path to your local poetry file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, os.pardir))
    
    #local_poetry_file_path = os.path.join(project_root, 'data', 'processed', 'renaissance_poetry_for_gpt2_finetune.txt')
    local_poetry_file_path = os.path.join(project_root, 'data', 'processed', 'renaissance_poetry_corpus.txt')

    # Tokenization
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['content'], truncation=True, max_length=max_length) # Use max_length from config

    # --- Load and filter merve/poetry ---
    print(f"Loading dataset: {dataset_name}")
    dataset_merve_raw = load_dataset(dataset_name)
    filtered_dataset_merve = dataset_merve_raw['train'].filter(filter_renaissance_love)
    print(f"Filtered dataset size (merve/poetry Renaissance Love): {len(filtered_dataset_merve)}")

    #
    print(f"Attempting to load local poetry from: {local_poetry_file_path}")
    local_poems_list = load_local_poetry_file(local_poetry_file_path)
    #

    
    # --- Load and parse local .txt poems ---
    print(f"Loading dataset: Project Gutenberg")
    local_poems_list = load_local_poetry_file(local_poetry_file_path)
    if not local_poems_list: # If file not found or empty, ensure it's an empty Dataset
        local_dataset = Dataset.from_dict({
            'content': [], 'author': [], 'poem name': [], 'age': [], 'type': []
        })
    else:
        local_dataset = Dataset.from_list(local_poems_list)
    print(f"Total poems parsed from local .txt file: {len(local_dataset)}")

    # --- Combine Datasets ---
    combined_dataset = concatenate_datasets([filtered_dataset_merve, local_dataset])
    print(f"Combined dataset size (merve/poetry + local .txt): {len(combined_dataset)}")

    
    tokenized_dataset = combined_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['author', 'content', 'poem name', 'age', 'type']
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenized_dataset, data_collator, tokenizer