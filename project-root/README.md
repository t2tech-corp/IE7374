# Renaissance Style Love Poetry GPT-Neo LoRA Fine-Tuning

This project explores the fine-tuning of GPT-family language models (LLMs) to generate poetry in the style of Renaissance love poems.
Utilizing Parameter-Efficient Fine-tuning (PEFT) with LoRA, the project aims to adapt powerful pre-trained models efficiently for a niche stylistic task.

## Repository Structure

* **`project-root/`**: The top-level directory for the entire project.
* **`configs/`**: Stores configuration files for the models, experiments, or hyperparameters.
    * **`model_config.yaml`**: A YAML file to define model parameters, hyperparameters, and other configuration settings.
* **`data/`**: Directory for storing datasets.
    * **`processed/`**: Specifically for cleaned and preprocessed data, differentiating it from raw data.
* **`notebooks/`**: Contains Jupyter notebooks for exploration, experimentation, or demonstrations.
    * **`demo_pipeline.ipynb`**: A notebook demonstrating the end-to-end pipeline or key functionalities.
* **`outputs/`**: Dedicated directory for storing generated output from the models or experiments.
    * **`samples.txt`**: For text-based outputs, logs, or generated samples.
* **`src/`**: Contains the core source code for the project.
    * **`data_loader.py`**: Script responsible for loading and preprocessing datasets.
    * **`model_runner.py`**: Script for loading trained models and performing inference (making predictions).
    * **`train.py`**: Script for training or fine-tuning models.
* **`utils/`**: Contains utility functions or scripts that are shared across different parts of the project.
    * **`helpers.py`**: Contains general helper functions.
* **`requirements.txt`**: Lists all the Python dependencies required for the project, allowing others to easily set up their environment using `pip install -r requirements.txt`.
* **`README.md`**: A crucial file providing an overview of the project, setup instructions, how to run the code, and any other relevant information for users or collaborators.

## Setup and Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/t2tech-corp/IE7374/tree/main/project-root
    cd https://github.com/t2tech-corp/IE7374/tree/main/project-root
    ```
 2. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ``` 

## Configuration

All model parameters, training hyperparameters, and generation settings are managed in `configs/model_config.yaml`.
Before running any scripts, review and adjust this file according to your needs and available hardware.

```yaml
# Example snippet from model_config.yaml
# General Settings
model_name: "EleutherAI/gpt-neo-1.3B"
dataset_name: "merve/poetry"
output_dir: "outputs/gpt_neo_renaissance_love_poems_lora" # Relative path to project-root
```

## How to Run

1. **Train the Model**
   To start the fine-tuning process, run the ``train.py`` script:
   ```bash
    python src/train.py
    ```
   This script will load the dataset, set up the model with LoRA, and begin training based on the parameters in ``model_config.yaml``.
   The fine-tuned LoRA adapters and tokenizer will be saved to the ``outputs/gpt_neo_renaissance_love_poems_lora/`` directory.

