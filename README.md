# Fine-tuning GPT-2 for Text Generation on Jane Austen's 'Emma'

## Project Overview

This repository contains the code and resources for fine-tuning a pre-trained GPT-2 language model on Jane Austen's classic novel, "Emma." The goal of this project is to adapt GPT-2 to generate text in the distinctive style and context of the novel, demonstrating the principles of domain-specific text generation.

## Project Goals

* **Data Preparation:** Load and preprocess the "Emma" dataset from NLTK's Gutenberg corpus.
* **Model Fine-tuning:** Fine-tune the GPT-2 (base) model using the Hugging Face Transformers library.
* **Quantitative Evaluation:** Calculate the perplexity of the fine-tuned model on the dataset.
* **Qualitative Evaluation:** Generate text samples based on prompts and analyze their coherence, grammaticality, and stylistic adherence to Jane Austen's writing.

## Dataset

The dataset used is Jane Austen's "Emma," accessed via the `nltk.corpus.gutenberg` module. The raw text was cleaned to remove Project Gutenberg boilerplate before being used for training and evaluation.

## Model

The project utilizes the `gpt2` model from the Hugging Face Transformers library. GPT-2 is a Transformer-based, decoder-only architecture well-suited for causal language modeling and text generation.

## Setup and Installation

To run this project, you can use Google Colab (recommended for GPU access) or a local Python environment.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your_username/your_repo_name.git](https://github.com/your_username/your_repo_name.git) # Replace with your actual repo URL
    cd your_repo_name
    ```
2.  **Install Dependencies:**
    ```bash
    !pip install transformers torch sentencepiece datasets accelerate scikit-learn nltk
    ```
    *(Note: If running locally, ensure you have a compatible PyTorch installation for your GPU or CPU.)*
3.  **Download NLTK Data:**
    ```python
    import nltk
    nltk.download('gutenberg')
    ```

## Usage

The primary workflow is demonstrated in the `final.ipynb` notebook.

1.  **Open `final_notebook.ipynb`:** Launch the notebook in Google Colab or your local Jupyter environment.
2.  **Run All Cells:** Execute all cells sequentially from top to bottom.
    * The notebook will handle data loading, cleaning, tokenizer initialization, model training, and evaluation.
    * It will save the fine-tuned model to `./fine_tuned_gpt2_alice/` locally within the Colab environment.
    * It will also mount Google Drive and copy the trained model there for persistent storage.
3.  **Text Generation:** The notebook includes a section for generating text samples using the fine-tuned model. You can modify the prompts and generation parameters (`temperature`, `top_k`, `top_p`) to experiment with different outputs.

## Results and Evaluation

### Quantitative Evaluation (Perplexity)

The model was evaluated on the raw "emma.txt" dataset.

* **Evaluation Loss:** `2.6336`
* **Perplexity:** `13.92`

A perplexity of `13.92` indicates that the model has learned significant patterns from the novel, performing substantially better than a random language model. The evaluation on the raw text (including boilerplate) contributes to this specific score.

### Qualitative Evaluation (Text Generation)

The fine-tuned model demonstrated a strong ability to generate coherent and contextually relevant text in the style of Jane Austen.

**Example Generated Text:**

* **Prompt:** "Emma, in her confusion, declared that"
* **Generated Sample:** `Emma, in her confusion, declared that she had been engaged in a scheme to procure the confession of some of the Highbury ladies who were with her when Emma married. Miss Fairfax was to be called to Hartfield.`

This sample showcases the model's capacity to produce grammatically correct sentences, maintain narrative flow, and incorporate elements (characters, locations) consistent with the novel.

## Trained Model

The trained GPT-2 model files are available for download from Google Drive due to their size. You can access them here:

[**https://drive.google.com/drive/folders/1liG6X-q_RYm1db0cqcTPEGw_4znKVptx?usp=sharing**]
*(Please ensure the link permissions are set to "Anyone with the link can view.")*

## Files in this Repository

* `final_notebook.ipynb`: The main Jupyter Notebook with all project code and execution steps.
* `Final_report.pdf`: The detailed project report (replace with `.md` if you submit Markdown).
* `emma.txt`: The raw "Emma" dataset used.
* `experiments.ipynb`: An auxiliary notebook showing additional experimental outputs and development steps.

## Future Work

* **Larger Dataset:** Fine-tune on a broader collection of Jane Austen's works or similar historical fiction to enhance stylistic consistency and generalization.
* **Parameter-Efficient Fine-Tuning (PEFT):** Explore techniques like LoRA to adapt the model more efficiently and mitigate catastrophic forgetting on smaller datasets.
* **Advanced Hyperparameter Tuning:** Conduct more extensive tuning of training and generation parameters for optimal output quality.

## Contact

Harshit Vishnoi
www.linkedin.com/in/harshitvishnoi-ai

