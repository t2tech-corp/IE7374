# IE7374 Semester Project
## Poetry Style Generative AI: Fine-tuning a Language Model for Interactive Poetry Style Generation

## Table of Contents

- [Introduction](#introduction)
- [Project Objective](#project-objective)
- [Literature Review](#literature-review)
- [Benchmarking](#benchmarking)
- [Experiments](#experiments)
- [Framework Selection](#framework-selection)
- [Dataset Preparation](#dataset-preparation)
- [Model Development](#model-development)
- [Training and Fine-Tuning](#training-and-fine-tuning)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Future Work](#future-work)
- [License](#license)
- [Team Members](#team-members)

## Introduction

The ability of Large Language Models (LLMs) to generate coherent and contextually relevant text has advanced remarkably. However, a significant
challenge remains in maintaining a nuanced stylistic control, especially within creative domains like poetry. While generic LLMs can produce verse,
their outputs often lack the consistent stylistic qualities, thematic depth, and structural conventions of a specific poet or genre. This project
addresses the problem of insufficient stylistic fidelity in general-purpose text generation, aiming to empower users to generate poetry that is not
merely grammatically correct but authentically represents the voice and form of a chosen poet or established genre. Solving this enhances the utility
of generative AI for creative expression, education, and literary analysis.


## Project Objective

The primary objective of this project is to develop and evaluate a generative AI system capable of producing novel poetic content that adheres to the
distinct stylistic characteristics of a chosen literary author or genre. This directly involves language modeling, specifically fine-tuning a pre-trained
Large Language Model to specialize its generative capabilities on specific poetic corpora. While the main task is generative text creation, the project
implicitly touches upon aspects of text classification during the evaluation phase, as a style classifier will be used to quantitatively assess how well
the generated text matches the target style. The system will act as a form of sequence-to-sequence modeling where an input prompt (a short text sequence)
is transformed into an output poem (a longer text sequence) under explicit stylistic constraints.

## Literature Review

1.	Bandi, A., Adapa, P. V. S. R., & Kuchi, Y. E. V. P. K. (2023). The power of generative ai: A review of requirements, models, input–output formats, evaluation metrics, and challenges. Future Internet, 15(8), 260.
2.	Raiaan, M. A. K., Mukta, M. S. H., Fatema, K., Fahad, N. M., Sakib, S., Mim, M. M. J., ... & Azam, S. (2024). A review on large language models: Architectures, applications, taxonomies, open issues and challenges. IEEE access, 12, 26839-26874.
3.  Marvin, G., Hellen, N., Jjingo, D., & Nakatumba-Nabende, J. (2023, June). Prompt engineering in large language models. In International conference on data intelligence and cognitive informatics (pp. 387-402). Singapore: Springer Nature Singapore.
4.  Alhafni, B., Kulkarni, V., Kumar, D., & Raheja, V. (2024). Personalized text generation with fine-grained linguistic control. arXiv preprint arXiv:2402.04914.
5.  Ajwani, R. D., Zhu, Z., Rose, J., & Rudzicz, F. (2024). Plug and Play with Prompts: A Prompt Tuning Approach for Controlling Text Generation. arXiv preprint arXiv:2404.05143.

## Benchmarking

**Summary of GPT-Family Models used in Experiments**

| Feature / Model                          | **DistilGPT2** | **GPT-2 (base)** | **GPT-2 Large** | **GPT-Neo 1.3B** |
| :--------------------------------------- | :----------------------- | :----------------------- | :----------------------- | :--------------------------- |
| **Parameters** | 82 Million               | 117 Million              | 774 Million              | 1.3 Billion                  |
| **Training Data** | Subset of WebText (distilled from GPT-2) | WebText (large internet corpus) | WebText                  | The Pile (large, diverse, higher quality than WebText) |
| **Architecture** | Distilled Transformer    | Transformer              | Transformer              | Transformer (similar to GPT-3 architecture elements) |
| **VRAM / Compute Needs (Relative)** | Lowest | Low                      | Medium                   | Medium-High |
| **Training Time (Per Step with LoRA, Relative)** | Fastest | Fast                     | Medium                   | Slower than GPT-2 Large      |
| **General Generation Quality (Coherence, Fluency)** | Fair (can be repetitive/less coherent) | Good                     | Very Good                | Excellent |
| **Stylistic Adaptation (with LoRA & Small Dataset)** | Very Limited (Least capacity to learn new style) | Limited (Prone to drift to general prose) | Better (Can initiate style, but struggles to sustain) | Good Potential (Showed strong poetic initiation |
| **Output Observed in Project** | Most generic/incoherent | Drifted quickly into generic prose. | Better coherence, but still generic prose. | Best Poetic Initiation with high coherence. |


## Experiments

Before proceeding with full-scale fine-tuning and extensive model training, preliminary experiments were crucial to validate the feasibility of the selected approaches and 
identify potential issues early in the development cycle. These small-scale tests ensured that the chosen frameworks and methodologies were appropriate for the project's objectives.

Preliminary experiments included:
*  **Dataset Filtering Validation:** Initial tests focused on confirming the correct application of filtering criteria to the ``merve/poetry`` dataset. This involved verifying that the ``age='Renaissance'`` and ``type='Love'`` filters correctly identified and isolated the intended subset of poems, including debugging initial case-sensitivity issues to ensure accurate data selection.
*  **LoRA Integration Check:** Small-scale runs were conducted to ensure that LoRA (Low-Rank Adaptation) was correctly integrated with the base language models. This involved verifying that the ``get_peft_model`` function successfully wrapped the model and that the number of trainable parameters was drastically reduced, confirming LoRA's efficiency benefits.
*  **Basic Training Loop Feasibility:** Short training runs (fewer epochs, smaller batch sizes) were performed to confirm that the entire fine-tuning pipeline, from data loading and tokenization to model training with the Hugging Face ``Trainer``, executed without fundamental errors. This included validating that the model's loss decreased during training, indicating it was actively learning.
*  **Initial Text Generation Sanity Check:** Even before extensive fine-tuning, preliminary attempts at text generation post-training were made to confirm that the model could produce any output and that the generation parameters were correctly applied. These initial outputs, though not yet stylistically refined, served as a baseline for future improvements.
* **Prompt Content Experimentation:** In order to observe how different initial phrases could steer the model's creative direction and thematic focus, a foundational prompt was created ("O, my love, you are like"). As experimentation continued, longer, more detailed prompts were created to provide more specific guidance to the model regarding desired themes, tone, or even specific poetic forms, theoretically reducing stylistic drift.
* **Output Size Experimentation:** A critical series of experiments involved systematically adjusting the ``max_length`` parameter in the ``generate()`` function (150 down to 50, then 40, and 30 tokens). These experiments revealed a consistent pattern: the fine-tuned model (especially ``GPT-Neo 1.3B``) exhibited a strong ability to initiate generation in the desired Renaissance poetic style for a few lines. However, if allowed to generate longer sequences, it consistently drifted into conversational prose. By reducing ``max_length``, it was possible to capture high-quality, stylistically relevant opening phrases before this drift occurred. This demonstrated a trade-off between the length of the generated output and its adherence to the target poetic form and style.
* **Experimentation with Beam Search:** Beyond basic sampling, the project extensively experimented with **Beam Search** as a decoding strategy to significantly enhance the quality and coherence of the generated text. The ``num_beams`` parameter in the ``lora_model.generate()`` function was set to a base value of 5 with subsequent experiments at 6, 7, and 8. Using ``num_beams=5`` had the most immediate and significant impact with a substantial increase in the logical flow and readability of the generated text. Outputs transitioned from highly fragmented or repetitive to grammatically correct and semantically coherent narratives. Increases to ``num_beams`` actually reduced the quality of the generated output across all experiments.
* **Targeted Repetition Control:** Experimentation with ``repetition_penalty`` and increasing ``no_repeat_ngram_size`` proved highly effective. These parameters specifically address, and almost entirely eliminated, the pervasive issue of repetitive text, significantly improving the fluency and readability of the generated output. This was a distinct success beyond general coherence.

## Framework Selection

The selection of NLP frameworks and libraries for this project was primarily driven by the need for efficient fine-tuning of large pre-trained language models (LLMs) on a specific,
moderately sized dataset, balanced with the goal of achieving high-quality text generation.

The core frameworks and libraries chosen include:

*  **Hugging Face Transformers:** This was the foundational choice due to its unparalleled support for state-of-the-art pre-trained LLMs (such as GPT-2, GPT-2 Large, DistilGPT2, and GPT-Neo), readily available tokenizers, and robust training utilities (like ``TrainingArguments`` and ``Trainer``). Its comprehensive ecosystem allowed for seamless loading, configuration, and management of various GPT-family models.
*  **Hugging Face Datasets:** Essential for efficient data handling, this library provided a streamlined way to load the ``merve/poetry`` dataset, apply complex filtering criteria (e.g., ``age='Renaissance'``, ``type='Love'``), perform batch tokenization, and manage dataset caching. Its integration with ``transformers`` ensured a smooth data pipeline.
*  **PEFT (Parameter-Efficient Fine-tuning):** Specifically, the LoRA (Low-Rank Adaptation) method from the PEFT library was crucial. Given the objective of fine-tuning LLMs on a relatively small, domain-specific dataset, LoRA allowed for highly efficient training by significantly reducing the number of trainable parameters. This enabled the project to perform fine-tuning on consumer-grade hardware while still leveraging powerful base models.
*  **PyTorch:** As the underlying deep learning framework for the Hugging Face ecosystem in this project, PyTorch provided the computational backbone for model operations, gradient calculations, and GPU acceleration. Its flexibility and performance were key to optimizing the training process, especially when combined with mixed-precision training.
*  **Accelerate:** While often used implicitly by the Hugging Face ``Trainer``, ``accelerate`` further optimizes the training loop, especially when utilizing features like mixed precision, contributing to faster and more memory-efficient fine-tuning on available hardware.
*  **Safetensors:** Used for efficiently loading model weights (specifically the LoRA adapters), ``safetensors`` offers a secure and faster alternative to traditional PyTorch ``.bin`` files, contributing to quicker model loading post-training.

## Dataset Preparation

The selected dataset for this project is the ``merve/poetry`` dataset sourced from Hugging Face. This dataset is specifically chosen for its focus on diverse
poetic styles and time periods, which is crucial for fine-tuning our generative AI model to emulate specific poetic genres. 

The dataset consists of 573 poems authored by 67 unique poets. Its total file size is less than 1KB, making it efficient for training and deployment. Each sample is presented as structured text,
with clear delineations for different attributes of the poem. 

The ``merve/poetry`` dataset is monolingual (English) and is well-suited for our project due to its structured nature and rich content.

It includes key features for each poem:
*  **Content:** The full text of the poem, which will serve as the primary input for the LLM fine-tuning process. This is directly relevant to learning poetic structures, vocabulary, and stylistic nuances.
*  **Author:** The name of the poet, enabling us to train and differentiate models based on individual authorial styles.
*  **Poem name:** The title of the poem.
*  **Age:** The period to which the poem belongs (e.g., ``Renaissance``, ``Modern``), which can be utilized to guide the model towards specific historical or genre-based poetic characteristics.
*  **Type:** The subject or genre of the poem (e.g., ``Love``, ``Nature``, ``Mythology & Folklore``), allowing for genre-specific model fine-tuning and content generation.

Dataset preparation is a critical phase for fine-tuning any language model, ensuring the data is in a format the model can understand and learn from effectively. For the ``merve/poetry`` dataset in this project, the preparation involved several key steps:
*  **Dataset Loading:** The ``merve/poetry`` dataset was loaded directly from the Hugging Face Hub using ``datasets.load_dataset("merve/poetry")``. This provided structured access to the entire dataset.
*  **Targeted Filtering:** To align with the project's specific goal of fine-tuning on Renaissance love poems, the loaded dataset was filtered. A custom Python function was used to select only those entries where the ``'age'`` metadata field was ``'Renaissance'`` and the ``'type'`` metadata field was ``'Love'``. This ensured that the model was exposed solely to the desired stylistic and thematic content.
*  **Column Management:** During tokenization, irrelevant original columns such as ``'author'``, ``'poem name'``, ``'age'``, and ``'type'`` were specified for removal using ``remove_columns`` to streamline the dataset and retain only the processed tokenized inputs required by the model.
*  **Tokenizer Loading:** An appropriate tokenizer (``AutoTokenizer.from_pretrained(MODEL_NAME)``) for the chosen GPT-family model (DistilGPT2, GPT-2, GPT-2 Large, GPT-Neo) was loaded. This tokenizer is responsible for converting raw text into numerical ``input_ids`` that the model understands.
*  **Special Token Handling:** GPT-family models, by default, do not have a dedicated padding token. For consistent batching during training, the ``tokenizer.pad_token`` was explicitly set to ``tokenizer.eos_token`` (end-of-sequence token). This allowed the tokenizer to pad shorter sequences in a batch to the ``max_length`` using the ``eos_token``.
*  **Tokenization Function:** A function (``tokenize_function``) was defined to apply the tokenizer to the ``content`` of each poem. This function included ``truncation=True`` and ``max_length=512`` to handle poems longer than the model's maximum sequence length, preventing memory issues and ensuring a consistent input size.
*  **Batch Processing:** The ``dataset.map()`` method was used with ``batched=True`` to apply the ``tokenize_function`` efficiently across the entire filtered dataset, processing multiple poems at once.
*  **Data Collator for Language Modeling:** A ``DataCollatorForLanguageModeling`` was employed. This component is essential for preparing batches of data during the training loop. For causal language modeling (like with GPT models), it performs dynamic padding to the longest sequence in each batch and automatically generates the ``labels`` (the target output tokens for training) by shifting the ``input_ids``. The ``mlm=False`` argument confirms that Masked Language Modeling (MLM), which is typically used for BERT-like models, is not being applied.

## Model Development

The model development phase focused on leveraging state-of-the-art pre-trained language models and implementing them in a modular and reusable manner,
central to the project's goal of fine-tuning for specific stylistic generation.

Key aspects of model implementation and development included:
*  **Utilization of Pre-trained LLMs:** Instead of building a language model from scratch, the project strategically utilized powerful, pre-trained models from the GPT family, including ``DistilGPT2``, ``GPT-2 (base)``, ``GPT-2 Large``, and ``GPT-Neo 1.3B``. These models, readily available through the **Hugging Face Transformers** library, serve as robust starting points with extensive linguistic knowledge, allowing the fine-tuning process to focus solely on adapting to the unique nuances of Renaissance love poetry.
*  **Modular Fine-tuning with LoRA:** The chosen fine-tuning method, **LoRA (Low-Rank Adaptation)** from the PEFT library, inherently promotes modularity. LoRA works by adding small, trainable adapter modules alongside the original pre-trained model's layers. This design ensures efficiency where only a fraction of the total model parameters are trained, significantly reducing computational requirements. The fine-tuned "knowledge" is encapsulated within small LoRA adapters, which can be easily swapped, saved, and loaded independently of the large base model. This makes the fine-tuning highly flexible and shareable.
*  **Causal Language Model Implementation:** The selected GPT-family models are all causal language models, designed to predict the next token in a sequence. Their implementation within the Hugging Face ecosystem provided a consistent API for loading, tokenization, and generation, ensuring compatibility throughout the project.
*  **Code Modularity and Reusability:** The codebase was structured to enhance clarity and reusability by using configurable parameters for model choices, output directories, and training hyperparameters. This allows for easy modification and experimentation with different models or training parameters/options. Dedicated functions for specific tasks were created for items like dataset fitering and tokenization. This allows for a clean data pipeline and adaptable for future dataset changes. Lastly, the project flow is logically divided into distinct steps (Configuration, Data Preparation, Model Loading, Fine-tuning, Generation), allowing for easier debugging, iteration, and understanding of the overall process.

## Training and Fine-Tuning

The core of this project involved fine-tuning large pre-trained language models (LLMs) to adapt them to the unique style of Renaissance love poetry.
This process leverages transfer learning, where a model that has already learned extensive language patterns from a massive, diverse dataset is further
trained on a smaller, domain-specific dataset to acquire new capabilities or specialized knowledge.

Key aspects of the training and fine-tuning process included:
*  **Parameter-Efficient Fine-tuning (PEFT) with LoRA:** To make the fine-tuning process computationally feasible and memory-efficient, **LoRA (Low-Rank Adaptation)** was employed. Instead of updating all millions/billions of parameters in the pre-trained LLM, LoRA injects small, trainable low-rank matrices into select layers. This significantly reduces the number of parameters that need to be trained, making it possible to fine-tune powerful models like GPT-2 and GPT-Neo on consumer-grade GPUs.
*  **Hugging Face Trainer:** The fine-tuning loop was managed using the Hugging Face Trainer API.
*  **Hyperparameter Management:** Defining and applying essential training parameters such as ``learning_rate``, ``per_device_train_batch_size``, and ``num_train_epochs``.
*  **Logging and Checkpointing:** Automatic logging of training loss and saving model checkpoints at specified intervals, allowing for monitoring progress and resuming training.
*  **Mixed Precision Training:** Leveraging ``fp16`` (half-precision floating point) with a CUDA-enabled GPU to significantly speed up training and reduce memory consumption.
*  **Model Adaptation:** Through the fine-tuning process, the pre-trained LLMs were guided to learn the linguistic nuances, thematic elements, and metaphorical structures prevalent in the curated dataset of Renaissance love poems. This aimed to enable them to generate new text that resonated with the target poetic style.

## Evaluation and Metrics

Assessing the performance of a fine-tuned language model, particularly for a subjective and stylistic task like generating Renaissance love poetry,
requires a multifaceted approach. While traditional quantitative metrics provide insights into language fluency and likelihood, deeper linguistic
analysis and qualitative human evaluation are crucial for capturing stylistic fidelity.

* **Perplexity (Quantitative - General Language Modeling Proficiency):** Perplexity measures how well a probability model predicts a sample. In the context of language models, a lower perplexity score indicates that the model assigns a higher probability to the test data, suggesting it is a more confident and fluent predictor of language. It serves as a general indicator of the model's ability to learn the statistical regularities of the fine-tuning dataset. A decreasing perplexity during training confirms that the model is learning from the poetry.
* **N-gram Analysis (Quantitative - Repetition & Novelty):** N-gram metrics involve counting sequences of N words. Assessing the percentage of unique n-grams (e.g., unigrams, bigrams, trigrams) in the generated text provides insight into how repetitive the output is and its overall diversity. This helps confirm that ``repetition_penalty`` and ``no_repeat_ngram_size`` are effective during training and fine-tuning. Comparing n-grams in generated text with those in the training set can indicate memorization versus true generation. High overlap might suggest memorization.
* **Part-of-Speech (POS) Usage (Linguistic - Stylistic Fingerprinting):** This involves analyzing the frequency and patterns of different parts of speech (nouns, verbs, adjectives, adverbs, etc.) in the generated poems compared to the original Renaissance poetry dataset. Specific poetic styles often have characteristic POS distributions (e.g., high adjective/noun ratio for descriptive poetry, high verb usage for action-oriented text). This is a powerful metric for assessing stylistic transfer. If the model successfully adopts the Renaissance style, its POS usage patterns should align more closely with the fine-tuning data than with generic prose. Libraries like ``spaCy`` or ``NLTK`` can be used for POS tagging.
* **Cosine Similarity on Embeddings (Semantic - Content Similarity):** This involves converting poems (generated and original) into numerical vector representations (embeddings) using a pre-trained sentence transformer model. Cosine similarity then measures the cosine of the angle between two embedding vectors, indicating their semantic similarity (ranging from -1 for opposite to 1 for identical). Cosine Similarity can assess if the semantic content or overall meaning of the generated poems aligns with the themes present in the Renaissance love poetry dataset. However, high semantic similarity does not directly imply stylistic mimicry; two poems can be about "love" but sound vastly different.
* **Human Evaluation (Qualitative - Style, Creativity, Novelty):** For subjective and creative tasks like poetry generation, human judgment remains indispensable. This involves having human evaluators read and score the generated poems based on criteria such as stylistic fidelity, coherence and fluency, creativity or novelty, and overall poetic quality. Human evaluation likely provides the most direct and accurate assessment of whether the model has truly achieved the project's artistic and stylistic goals.

## Future Work

*  **Exploring different models or techniques:** Based on the results of preliminary experiments with the identified GPT models, it may be worthy to expand experiments to test other pre-trained LLMs to compare stylistic generation where different or larger corpus were used.
*  **Expanding the training corpus:** The ``merve/poetry`` dataset is relatively small when compared to the size of the training corpus used in the creation of the pre-trained model. Additional fine-tuning data can be found in sources such as **Project Gutenberg** which can augment the existing fine-tuning data to generate higher quality stylistic output. 
*  **Improving model fine-tuning:** Improvements include increasing **LoRA Attention Dimension/Rank** (``r``). A higher rank ``r`` allows the LoRA adapters to be more expressive and capture more complex patterns. This means the fine-tuning process has more "capacity" to learn the specific stylistic nuances of Renaissance poetry. This will increase the number of trainable parameters. The **LoRA Scaling Factor** (``lora_alpha``) can be adjusted in coordination with ``r``. ``lora_alpha`` acts as a scaling factor and can influence the learning dynamics and how effectively the LoRA updates are applied to the base model's weights, potentially leading to faster or better convergence. 

## License
The MIT License (MIT)

Copyright (c) 2015 Chris Kibble

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Team Members
- Adrian Rodriguez 
- Xiaodong Lian
- Vy Hoang
- Tim Terry 

























