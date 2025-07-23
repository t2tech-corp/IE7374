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

## Core Pipeline Components

The project's functionality is organized into several key Python scripts and a central configuration file, ensuring modularity, reusability, and clarity.

**``model_config.yaml``**   
This YAML file serves as the single source of truth for all configurable parameters in the project, enhancing modularity and ease of experimentation.
It stores all hyperparameters, model names, directory paths, and generation settings in a human-readable and easily modifiable format.

Centralizing these settings in a YAML file allows for rapid iteration on experiments without modifying the Python code, promoting consistency
and reusability across different runs and components of the pipeline.

* **General Settings:** Defines the ``model_name`` (e.g., "EleutherAI/gpt-neo-1.3B"), ``dataset_name`` ("merve/poetry"), and the ``output_dir`` for saving results.
* **Training Parameters:** Specifies ``learning_rate``, ``per_device_train_batch_size``, ``num_train_epochs``, logging frequency, save limits, and ``fp16`` (mixed precision) settings for the ``Trainer``.
* **LoRA Configuration:** Details the LoRA parameters such as ``r`` (rank), ``lora_alpha`` (scaling factor), ``lora_dropout``, ``bias`` settings, and the ``task_type`` for PEFT.
* **Text Generation Parameters:** Includes all parameters critical for controlling the generated output, such as ``prompt``, ``max_length``, ``num_return_sequences``, ``no_repeat_ngram_size``, ``repetition_penalty``, ``num_beams`` (for beam search), and ``do_sample``.

**``data_loader.py``**   
This script centralizes all operations related to dataset acquisition and preprocessing. It is responsible for loading, filtering, and preparing the
raw data into a format suitable for model training and inference.

* **Configuration Access:** Contains a function (``load_config``) to read project-wide settings from ``configs/model_config.yaml``.
* **Dataset Loading:** Fetches the ``raw merve/poetry`` dataset from the Hugging Face Hub.
* **Targeted Filtering:** Implements specific logic (``filter_renaissance_love``) to narrow down the dataset to only Renaissance love poems, ensuring the model is fine-tuned on highly relevant content.
* **Tokenizer Initialization:** Loads the appropriate ``AutoTokenizer`` for the chosen LLM, handling special token requirements (e.g., setting ``pad_token = eos_token`` for GPT-family models).
* **Tokenization Pipeline:** Defines and applies a tokenization function to convert raw poem text into numerical ``input_ids``, managing ``truncation`` and ``max_length`` to prepare sequences for the model.
* **Data Collator Setup:** Configures a ``DataCollatorForLanguageModeling`` to handle dynamic padding of batches and automatic label generation for causal language modeling.
* **Output:** Returns the tokenized dataset, the configured data collator, and the tokenizer itself, ready for use by ``train.py`` and ``model_runner.py``.

**``train.py``**   
This script serves as the orchestrator for the model fine-tuning process. It brings together data preparation, model loading, and the training loop.
It executes the fine-tuning of the selected pre-trained language model (e.g., GPT-Neo) using LoRA adapters.

* **Configuration Loading:** Loads all operational parameters, hyperparameters, and model settings from ``configs/model_config.yaml``.
* **Data Integration:** Utilizes functions from ``data_loader.py`` to load, filter, and tokenize the ``merve/poetry`` dataset, preparing it for the training process.
* **Model Initialization:** Loads the chosen pre-trained base LLM (``EleutherAI/gpt-neo-1.3B``).
* **LoRA Application:** Configures and applies LoRA adapters to the base model using the ``peft`` library, making only a small fraction of the model's parameters trainable.
* **Training Execution:** Sets up ``TrainingArguments`` (defining learning rate, batch size, epochs, logging, saving strategies, and mixed precision) and initiates the training process via the Hugging Face ``Trainer``.
* **Model Saving:**  Upon successful completion of fine-tuning, it saves only the trained LoRA adapters (``adapter_model.safetensors``) and the tokenizer configuration to the specified ``output_dir``, ensuring a lightweight and reusable fine-tuned model.

**``model_runner.py``**   
This script is dedicated to inference, allowing one to generate new poems using the fine-tuned model. It loads a previously fine-tuned model and generates text based on user-provided or default prompts.

* **Configuration Loading:** Retrieves all necessary settings, including model name, output directories, and generation parameters, from ``configs/model_config.yaml``.
* **Model Reconstruction:** Loads the original pre-trained base LLM and then intelligently reconstructs the ``PeftModel`` structure by applying the saved LoRA configuration.
* **Adapter Loading:** Loads the trained LoRA adapter weights (``adapter_model.safetensors``) into the reconstructed model, effectively "activating" the fine-tuned knowledge.
* **Tokenizer Loading:** Loads the corresponding tokenizer, crucial for encoding input prompts and decoding generated text.
* **Prompt Preparation:** Encodes the input prompt into the model's numerical format, including generating an ``attention_mask`` for optimal generation quality.
* **Text Generation:** Utilizes the model's ``generate()`` method with carefully selected parameters (``num_beams`` for coherence, ``repetition_penalty`` for diversity,``max_length`` for output control) to produce new poetic text.
* **Output Saving:** Appends the generated poems, along with their prompts, to ``outputs/samples.txt`` for evaluation.

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

2. **Generate Poems**   
   After training, you can generate new poems using the ``model_runner.py`` script:
   ```bash
   python src/model_runner.py
    ```
   By default, this will use the prompt defined in ``model_config.yaml``. You can also provide a custom prompt:
   ```bash
   python src/model_runner.py "My dearest heart, your beauty bright"
    ```

3. **Explore with Jupyter Notebooks**    
   For interactive experimentation and to see the end-to-end pipeline in a notebook environment, launch Jupyter:
   ```bash
   jupyter notebook notebooks/demo_pipeline.ipynb
    ```

## Framework Selection

The selection of NLP frameworks and libraries for this project was primarily driven by the need for efficient fine-tuning
of large pre-trained language models (LLMs) on a specific, moderately sized dataset, balanced with the goal of achieving
high-quality text generation.

Key frameworks include **Hugging Face Transformers** for LLM management, **Hugging Face Datasets** for data handling, 
and **PEFT (Parameter-Efficient Fine-tuning)** with LoRA for efficient adaptation. **PyTorch** serves as the underlying deep
learning backend, complemented by ``accelerate`` for optimized training and ``safetensors`` for efficient model loading.
This combination provided a robust, flexible, and efficient toolkit for the project's requirements.

## Dataset Preparation

Dataset preparation involved a meticulous process to transform the raw ``merve/poetry`` dataset into a fine-tune-ready format.

Steps included:
1. **Loading:** Utilizing ``datasets.load_dataset()`` for initial data acquisition.
2. **Targeted Filtering:** Applying precise filters (``age='Renaissance'``, ``type='Love'``) to select the relevant subset of poems, after careful validation of dataset metadata.
3. **Tokenization:** Employing ``AutoTokenizer`` to convert text into numerical ``input_ids``, handling ``max_length`` truncation, and setting ``tokenizer.pad_token = tokenizer.eos_token`` for consistent batching.
4. **Data Collator:** Using ``DataCollatorForLanguageModeling`` to prepare batches with dynamic padding and language modeling labels for the training process.

## Training & Fine Tuning

The project extensively leveraged **transfer learning**, fine-tuning pre-trained GPT-family LLMs to generate Renaissance love poetry. 

The core of this process relied on:
* **LoRA (Low-Rank Adaptation):** As a PEFT method, LoRA significantly reduced the number of trainable parameters, enabling efficient fine-tuning on consumer-grade hardware by injecting small, adaptable matrices.
* **Hugging Face ``Trainer``:** This API streamlined the training loop, managing hyperparameters (``learning_rate``, ``batch_size``, ``num_train_epochs``), logging, checkpointing, and leveraging mixed-precision training (``fp16``) for performance. This adaptation aimed to imbue the general-purpose LLMs with the unique linguistic nuances and stylistic elements of the target poetry domain.

## Preliminary Experiments

Initial small-scale tests were crucial for validating the feasibility of selected approaches. 

This phase involved:
* **Dataset Filtering Validation:** Ensuring accurate data selection and debugging issues like case sensitivity in metadata.
* **LoRA Integration Check:** Confirming that LoRA correctly reduced trainable parameters and integrated with the base models.
* **Basic Training Loop Feasibility:** Verifying that the data pipeline and ``Trainer`` setup executed without fundamental errors and that the model began to learn.
* **Initial Text Generation Sanity Checks:** Confirming the model's ability to produce output post-training as a baseline for future improvements.

## Experimentation with Prompts, Prompt Size, and Output Size

Controlling the input and output dimensions was key to refining generation quality

Findings include:
* **Prompt Content:** Experiments focused on how initial phrases (e.g., "O, my love, you are like") could steer thematic and stylistic direction.
* **Prompt Size:** While not extensively varied, understanding the role of input context for guiding generation was implicit.
* **Output Size (``max_length``):** Systematically adjusting ``max_length`` (e.g., from 150 down to 30-40 tokens) revealed a critical insight: the fine-tuned models (especially ``GPT-Neo 1.3B``) could reliably initiate generation in the desired Renaissance poetic style for a few lines, but consistently drifted into conversational prose if allowed to generate longer outputs. This highlighted a trade-off between output length and stylistic purity, guiding the strategy to capture the most poetic fragments.

## Experimentation with Beam Search

Beam Search was implemented as a decoding strategy to significantly enhance generated text quality.

* **Mechanism:** Instead of simple greedy decoding, Beam Search explored multiple probable sequences (``num_beams=5``), selecting globally optimal paths.
* **Impact:** This led to a dramatic improvement in overall coherence and fluency of the generated text, reducing repetition effectively (in conjunction with ``repetition_penalty`` and ``no_repeat_ngram_size``).
* **Limitations:** While greatly improving general text quality, Beam Search did not inherently resolve the stylistic drift from poetic to prose form, as it optimizes for what the underlying model considers most probable, which often defaults to the general prose learned during pre-training.

## Evaluation Statistics

Using the 12 generated poems from the fine-tuned model, a series of evaluations were performed to assess model performance.

**Perplexity**   
Measures how “surprised” a language model is by a given text. Lower perplexity means the model assigns higher probability to the text.
* Lower scores indicate the model is better at modeling the style/content of your corpus.
* Extremely low scores can also hint at overly safe or repetitive outputs.

Average Perplexity: 30.97

An average perplexity of 30.97 indicates that the fine-tuned GPT-Neo model is generating well-formed and statistically plausible English sentences.
This is a positive outcome for the fundamental language modeling aspect.

**Word-Level Diversity**   
Quantifies how many unique n-grams appear in the generated texts, relative to the total number of n-grams produced. It’s a simple diversity measure.
* Measures word-level diversity (unigrams).
* Values closer to 1.0 mean high diversity (few repeats)
* Values closer to 0 mean the model is repeating the same words/phrases.

Average Word-Level Diversity: 0.4958

An average word-level diversity of 0.4958 is a moderate score. It would suggest that while the model avoids repeating phrases, it might still
repeat individual words more often than desired for highly lexically rich poetry, indicating an area for potential improvement in vocabulary variety.

**Phrase-Level Diversity**   
Quantifies how many unique n-grams appear in the generated texts, relative to the total number of n-grams produced. It’s a simple diversity measure.
* Measures phrase-level diversity (bigrams).
* Values closer to 1.0 mean high diversity (few repeats)
* Values closer to 0 mean the model is repeating the same words/phrases.

Average Phrase-Level Diversity: 0.8493

An average phrase-level diversity of 0.8493 is a strong score. It suggests that the generated poems are highly diverse at the phrase level,
meaning the model is very good at constructing different word combinations and avoiding repetitive phrasing across the outputs.

**Self-BLEU**   
Evaluates how similar the generated samples are to one another. It’s a reverse of BLEU: treating each generation as a “hypothesis” and all the others as “references.”
* Scores range from 0 to 1.
* Higher Self-BLEU means samples are very similar to each other (low diversity).
* Lower Self-BLEU means samples are more distinct.

Average Self-BLEU: 0.1252

An Average Self-BLEU of 0.1252 is a very low score. It strongly suggests that your fine-tuned model is capable of generating a wide variety of 
different poems (or poetic fragments) from the diverse prompts provided.

## Style Metrics

**TTR (Type-Token Ratio)**   
Reflects the variety of vocabulary used in a text, with higher TTRs indicating greater diversity.
* High TTR (closer to 1): lots of different words — strong variety.
* Low TTR (closer to 0): repeat the same words more often — less variety.

Average TTR: 0.7202

An average TTR of 0.7202 is a relatively high score for lexical diversity. For generated text, especially relatively short fragments like those
generated by the model, this is a very good result. It indicates that the model is actively using a varied vocabulary within each generated output.

**Simpson Diversity Index**   
Quantifies the diversity of text or vocabulary within a corpus.
* High Simpson (closer to 1): high probability that two randomly picked tokens are different—strong diversity.
* Low Simpson (closer to 0): high chance that two picks are the same token—low diversity.

Average Simpson Diversity: 0.9706

An average Simpson Diversity of 0.9706 is an exceptionally high score for lexical diversity. It strongly indicates that the model is generating
text with an incredibly varied vocabulary, where word repetition within individual generated poems is extremely low.

**POS KL Divergence**   
Compares the distribution of POS tags in different texts or linguistic models.
* Low KL (near 0): generated poem’s POS mix is very similar to the reference style.
* High KL: generated poem’s POS proportions deviate strongly from that style.

Average POS KL Divergence: 0.3209

An average POS KL Divergence of 0.3209 indicates a moderate level of divergence. It suggests that while the model has learned some aspects of
the Renaissance love poem style, its overall grammatical composition still differs noticeably from the statistical patterns found in the original training data. 

**Novelty Detection (Cosine Similarity)**   
Assesses similarity in various applications, including text analysis. 
* Scores range from -1 to 1
* 1: Indicates the vectors are perfectly aligned and identical in direction (maximum similarity).
* 0: Indicates the vectors are orthogonal (no similarity).
* -1: Indicates the vectors are diametrically opposed (maximum dissimilarity).

Average Novelty: 0.1885

An average Novelty Score of 0.1885 is relatively low. This suggests that, on average, a significant portion of the generated content is not entirely new
compared to the ``merve/poetry`` fine-tuning dataset. It implies that the model might be heavily influenced by, or even directly reproducing short phrases
or common patterns it encountered during training.

## Summary

The evaluation statistics and style metrics indicate that the project successfully developed a fine-tuned GPT-Neo model capable of generating highly fluent,
coherent, and diverse text that effectively initiates a Renaissance love poem style. The technical fine-tuning with LoRA and careful parameter tuning proved
highly effective in controlling basic text quality and activating the desired stylistic mode.

However, the primary challenge is the model's ability to sustain that specific poetic style and linguistic fidelity over an entire generation,
and to produce truly novel content beyond variations of learned patterns. This points to the inherent limitations of fine-tuning with a comparatively
small dataset against the large and generalized knowledge encoded in the base LLM. To achieve deeper stylistic mimicry and higher originality, a larger,
more diverse collection of Renaissance poetry for fine-tuning would be the most impactful next step.
