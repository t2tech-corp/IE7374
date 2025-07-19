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
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
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

## Model Development

For this project, we will employ a pre-trained GPT-2 model. GPT-2 (Generative Pre-trained Transformer 2) is a Transformer-based decoder-only architecture. 
Developed by OpenAI, it utilizes a stack of Transformer decoder blocks. This architecture is autoregressive, meaning it predicts the next token in a sequence
based on all the preceding tokens, making it highly effective for generative tasks. GPT-2 was notable for its training on a massive and diverse corpus of internet
text, allowing it to learn extensive linguistic patterns, grammar, syntax, and semantics.

Leveraging a model already trained on a vast corpus of text significantly reduces the computational resources and time required for training from scratch. 
The pre-trained model has already acquired a broad understanding of language, which we will then specialize through fine-tuning on our specific poetry dataset.
This approach is highly recommended for achieving high-quality generative outputs efficiently within the scope of this project.

GPT-2 is an appropriate choice for our project due to several key reasons directly aligning with our research questions and project goals:
*  Strong Generative Capabilities: As a generative pre-trained transformer, GPT-2 excels at producing coherent, contextually relevant, and human-like text. This is fundamental to our goal of generating new poetry that embodies specific styles. Its autoregressive nature allows for the sequential generation of text, which is natural for poetic forms.
*  Fine-tuning Potential for Stylistic Emulation: GPT-2's architecture and pre-training enable effective fine-tuning on smaller, domain-specific datasets. This is crucial for our project, as we aim to fine-tune the model on curated datasets of poetry (like merve/poetry) to capture and replicate the unique stylistic characteristics of chosen poets or poetic genres. The model's learned general linguistic knowledge serves as a powerful starting point, which can then be specialized for the intricate patterns of poetic language.
*  Adaptability to Prompts: GPT-2 can generate conditional samples, meaning it can generate text based on a given prompt or topic. This directly supports the user interaction feature of our web application, where users provide prompts, and the model generates poetry in response.
*  Open-Source Availability and Community Support: While larger, more recent models exist, GPT-2 offers a balance of strong performance and accessibility. Its open-source nature, coupled with robust support from libraries like Hugging Face Transformers, makes it practical for research and development, allowing for easier implementation, fine-tuning, and deployment within a web application.

## Training and Fine-Tuning

## Evaluation and Metrics

* Programming language used: Python.
* Libraries used: 
* Structure of the code: 

## Requirements

All code can be run using the provided Python scripts and notebooks with the related datasets found in the corpus. Final Python and library versions will be provided upon final model decisions.

## Usage

Proper usage of the finalized scripts and selected models will be provided upon completion of project.

## Results

Results for models selected will be provided upon project completion once model refinements are settled. Results criteria will be as follows:

* Performance metrics.
* Comparisons of approaches.
* Any visualizations or graphs to illustrate the results.

## Future Work

*  Exploring different models or techniques.
*  Expanding the training corpus.
*  Improving model fine-tuning.

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

























