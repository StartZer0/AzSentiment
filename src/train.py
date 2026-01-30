"""
AzSentiment - Training Script
Fine-tunes a RoBERTa model on Azerbaijani sentiment data.
Run this script on Google Colab with GPU for best performance.
"""
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from data import prepare_datasets

# Configuration
MODEL_ID = "allmalab/bert-base-aze"  # aLLMA-BASE: 86.26 avg F1 on Az benchmarks
NUM_LABELS = 2  # Positive, Negative
OUTPUT_DIR = "./models/az-sentiment"
BEST_MODEL_DIR = "./models/az-sentiment-best"

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels)["f1"]
    }


def main():
    print("=" * 50)
    print("AzSentiment Training")
    print(f"Model: {MODEL_ID}")
    print("=" * 50)
    
    # Load and prepare data
    print("\n📊 Loading dataset...")
    train_ds, val_ds, test_ds, tokenizer = prepare_datasets(MODEL_ID)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Load model
    print("\n🤖 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        push_to_hub=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save best model
    print(f"\n💾 Saving best model to {BEST_MODEL_DIR}...")
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    
    # Final evaluation on test set
    print("\n📈 Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    print(f"Test Results: {test_results}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
