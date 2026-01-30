"""
AzSentiment - Data Loading Module
Loads and preprocesses the Azerbaijani sentiment dataset.
Uses pandas for robust loading to avoid datasets library issues.
"""
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


# Direct HuggingFace Hub URLs for the CSV files
TRAIN_URL = "https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification/resolve/main/train.csv"
TEST_URL = "https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification/resolve/main/test.csv"


def load_az_sentiment():
    """
    Load the Azerbaijani sentiment classification dataset.
    
    Preprocessing:
    - Rename 'content' -> 'text'
    - Convert 'score' (1-5) to binary 'label' (0=negative, 1=positive)
      - score 1-2: negative (0)
      - score 4-5: positive (1)
      - score 3: neutral (filtered out for binary classification)
    """
    print("📥 Downloading train.csv...")
    train_df = pd.read_csv(TRAIN_URL)
    print("📥 Downloading test.csv...")
    test_df = pd.read_csv(TEST_URL)
    
    def preprocess(df):
        # Rename column
        df = df.rename(columns={"content": "text"})
        
        # Filter out neutral scores (3) for binary classification
        df = df[df["score"] != 3].copy()
        
        # Create binary label
        df["label"] = df["score"].apply(lambda x: 1 if x >= 4 else 0)
        
        # Keep only needed columns
        return df[["text", "label"]]
    
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    
    print(f"✅ After filtering: Train={len(train_df)}, Test={len(test_df)}")
    
    # Convert to HuggingFace Dataset format
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    
    return {"train": train_ds, "test": test_ds}


def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenize the dataset with the given tokenizer."""
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    return dataset.map(tokenize_fn, batched=True)


def prepare_datasets(model_id: str, test_size: float = 0.2):
    """
    Load, tokenize, and split the dataset.
    
    Args:
        model_id: HuggingFace model ID for the tokenizer
        test_size: Fraction for validation split
    
    Returns:
        train_ds, val_ds, test_ds, tokenizer
    """
    # Load raw data
    data = load_az_sentiment()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenize
    train_tokenized = tokenize_dataset(data["train"], tokenizer)
    test_tokenized = tokenize_dataset(data["test"], tokenizer)
    
    # Split train into train/val
    split = train_tokenized.train_test_split(test_size=test_size, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = test_tokenized
    
    return train_ds, val_ds, test_ds, tokenizer


if __name__ == "__main__":
    # Quick test
    data = load_az_sentiment()
    print(f"✅ Train: {len(data['train'])}, Test: {len(data['test'])}")
    print(f"Columns: {data['train'].column_names}")
    print(f"Sample: {data['train'][0]}")
