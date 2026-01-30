# 🇦🇿 AzSentiment Analyzer

Sentiment analysis for Azerbaijani customer reviews using the aLLMA-BASE model.

## Model Choice Justification

We use **allmalab/bert-base-aze** (aLLMA-BASE), a monolingual Azerbaijani BERT model trained from scratch on the DOLLMA corpus (651M words). According to the benchmark results from [Isbarov et al., 2024]:

| Model | Avg F1 (6 tasks) | Notes |
|-------|------------------|-------|
| **aLLMA-BASE** | **86.26** | Best in 4/6 Az NLU tasks |
| mDeBERTa-v3-BASE | ~86 | Multilingual, larger |
| Multilingual BERT | 74.88 | Baseline |

Key advantages:
- Native SentencePiece tokenizer trained on Azerbaijani (64k vocab)
- Outperforms multilingual models by 11+ F1 points
- Handles Azerbaijani morphology correctly

## Dataset

[hajili/azerbaijani_review_sentiment_classification](https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification)
- Train: 124,774 samples
- Test: 31,215 samples
- Labels: Binary (Positive/Negative)

Note: This dataset is NOT used in the aLLMA paper benchmarks, making this project a novel application.

## Project Structure

```
AzSentiment/
├── src/
│   ├── data.py       # Data loading and preprocessing
│   ├── train.py      # Training script
│   └── app.py        # Streamlit application
├── notebooks/
│   └── train_colab.ipynb  # Colab training notebook
├── models/           # Fine-tuned model (after training)
├── requirements.txt
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training (Google Colab)
1. Upload this repository to GitHub
2. Open `notebooks/train_colab.ipynb` in Google Colab
3. Set runtime to GPU (T4)
4. Run all cells
5. Model will be saved to Google Drive

### Run the App
```bash
streamlit run src/app.py
```

## Example Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./models/az-sentiment-best")

# Positive
classifier("Bu mehsul cox yaxsidir!")  # This product is very good!

# Negative
classifier("Xidmet pis idi")  # Service was bad
```

## Citation

This project uses the aLLMA foundation models. If you use this work, please cite:

```bibtex
@inproceedings{isbarov-etal-2024-open,
    title = "Open foundation models for {A}zerbaijani language",
    author = "Isbarov, Jafar and
      Huseynova, Kavsar and
      Mammadov, Elvin and
      Hajili, Mammad and
      Ataman, Duygu",
    booktitle = "Proceedings of the First Workshop on Natural Language Processing for Turkic Languages (SIGTURK 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigturk-1.2",
    pages = "18--28"
}
```

## License

MIT
