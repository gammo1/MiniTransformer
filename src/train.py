
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import prepare_data
from model import SentimentClassifier


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Keep the original model configuration unchanged.
D_MODEL = 32
NUM_HEADS = 4
FF_DIM = 64
MAX_LENGTH = 200
EPOCHS = 10
LEARNING_RATE = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    print(f"Using device: {DEVICE}")
    print(f"Training epochs: {EPOCHS}")
    print("Experiment: 24,000 training reviews")

    train_loader, val_loader, test_loader, vocab = prepare_data()

    model = SentimentClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = -1.0

    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the previous model; save this experiment separately.
    model_path = output_dir / "sentiment_model_10epochs.pth"

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("Training started. Epoch timing will be reported below.")

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_index + 1) % 100 == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Elapsed: {elapsed / 60:.1f} min"
                )

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion
        )

        epoch_time = time.perf_counter() - epoch_start

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Epoch duration: {epoch_time / 60:.2f} minutes")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "vocab_size": len(vocab),
                        "d_model": D_MODEL,
                        "num_heads": NUM_HEADS,
                        "ff_dim": FF_DIM,
                        "max_length": MAX_LENGTH,
                    },
                    "epoch": epoch + 1,
                    "best_val_accuracy": best_val_accuracy,
                    "training_reviews": len(train_loader.dataset),
                    "seed": SEED,
                },
                model_path,
            )

            print(f"Saved best model to {model_path}")

        print("-" * 50)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved at: {model_path}")


if __name__ == "__main__":
    main()
