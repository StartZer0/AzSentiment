# 🇦🇿 AzSentiment Analyzer

Sentiment analysis for Azerbaijani customer reviews using fine-tuned aLLMA-BASE.

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **F1-Score** | 0.94 |
| **Precision** | 0.94 |
| **Recall** | 0.94 |

## Dataset

[hajili/azerbaijani_review_sentiment_classification](https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification)
- Train: 124,774 samples
- Test: 31,215 samples
- Labels: Binary (Positive/Negative)

## Project Structure

```
AzSentiment/
├── src/
│   ├── data.py       # Data loading and preprocessing
│   ├── train.py      # Training script
│   └── app.py        # Streamlit application
├── notebooks/
│   └── train_colab.ipynb  # Colab training notebook
├── models/           # Fine-tuned model
├── requirements.txt
└── README.md
```

## Quick Start

### Training (Google Colab)
1. Open `notebooks/train_colab.ipynb` in Colab
2. Set runtime to GPU (T4)
3. Run all cells
4. Model saved to Google Drive

### Run the App
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Example Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./models/az-sentiment-best")

classifier("Bu mehsul cox yaxsidir!")  # Positive
classifier("Xidmet pis idi")           # Negative
```

## Model

Uses [allmalab/bert-base-aze](https://huggingface.co/allmalab/bert-base-aze) (aLLMA-BASE) - a monolingual Azerbaijani BERT model.

## License

MIT
