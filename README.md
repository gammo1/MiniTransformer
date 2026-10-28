# MiniTransformer — Movie Review Sentiment Analysis

A small Transformer encoder built from scratch with PyTorch to classify movie reviews as positive or negative. The project explores self-attention, model training, evaluation, and controlled experiments through an interactive web application.

## Demo

The application lets users enter a movie review and receive a sentiment prediction with the model's confidence score.

**Interface:** Streamlit  
**Framework:** PyTorch  
**Dataset:** IMDb movie reviews

## Results

The main Transformer achieved **82.5% test accuracy** on a fixed subset of 1,000 IMDb test reviews.

| Model | Test Accuracy |
|---|---:|
| TF-IDF + Logistic Regression baseline | 87.1% |
| Transformer — 10,000-word vocabulary, 200 tokens | 82.5% |
| Transformer — 5,000-word vocabulary, 200 tokens | 81.7% |
| Transformer — 10,000-word vocabulary, 100 tokens | 78.7% |

The TF-IDF baseline outperformed the Transformer in these experiments. This is an important result: more complex neural architectures do not automatically perform better than simpler approaches.

Results are specific to this project's data split and experimental setup.

## Model Architecture

The classifier includes:

- **Token embeddings** to represent vocabulary items as vectors.
- **Learned positional embeddings** to represent token positions.
- **Multi-head self-attention** using learned Query, Key, and Value projections.
- **Scaled dot-product attention** to calculate relationships between tokens.
- **Feed-forward network** to transform token representations.
- **Residual connections and LayerNorm** to support stable learning.
- **Masked mean pooling** to combine token representations.
- **Linear classification head** to predict positive or negative sentiment.

The implementation uses one Transformer encoder block.

## Experiments

Three controlled experiments explored the effect of training duration, vocabulary size, and sequence length.

### 1. Training duration

Training for 10 epochs improved test accuracy to 82.5%, compared with 81.5% for the earlier 5-epoch run. The checkpoint with the best validation accuracy was saved from epoch 8.

### 2. Vocabulary size

Reducing the vocabulary from 10,000 to 5,000 tokens reduced the model's trainable parameter count from approximately 335,010 to 175,010. Test accuracy decreased from 82.5% to 81.7%.

### 3. Sequence length

Reducing the maximum sequence length from 200 to 100 tokens made training approximately twice as fast per epoch on the development machine. Test accuracy decreased from 82.5% to 78.7%.

These experiments demonstrate trade-offs between model size, training speed, input context, and predictive performance.

## Project Structure


MiniTransformer/
├── app.py
├── README.md
├── models/
│   ├── sentiment_model_10epochs.pth
│   └── sentiment_model_5k_vocab.pth
└── src/
    ├── data.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    └── baseline.py


## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MiniTransformer
```

### 2. Create and activate a virtual environment

Windows PowerShell:


python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 3. Install dependencies

pip install torch streamlit datasets numpy scikit-learn


### 4. Run the web application


streamlit run app.py

The application loads the saved model checkpoint and opens in your browser.

## Dataset and Evaluation

The project uses the IMDb dataset provided through Hugging Face Datasets (`stanfordnlp/imdb`). The vocabulary is built using training reviews only. A separate validation subset is used for checkpoint selection, and a fixed subset of 1,000 test reviews is used for the reported test results.

The reported accuracy is not a claim of state-of-the-art performance. It represents the measured performance under this project's experimental setup.

## Limitations

- The evaluation uses a 1,000-review test subset rather than the entire IMDb test set.
- The vocabulary and tokenizer are relatively simple and do not use subword tokenization.
- The model has one encoder block and is intentionally small.
- Predicted confidence is not necessarily calibrated and does not guarantee correctness.
- The baseline performs better than the Transformer in the current experiments.

## Learning Objectives

This project was built to develop practical understanding of Transformer internals, PyTorch model implementation, training and validation, classification metrics, baseline comparisons, and controlled experimentation.

## License

Add a license before distributing this project if you want to specify how others may reuse the code.
