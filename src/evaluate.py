
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data import prepare_data
from model import SentimentClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "sentiment_model_10epochs.pth.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run python src/train.py first."
        )

    print("Preparing test data...")
    _, _, test_loader, _ = prepare_data()

    # Load the best checkpoint saved during training.
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    model = SentimentClassifier(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        max_length=config["max_length"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    print("Evaluating on the test set...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)

            logits = model(inputs)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    print("\n===== TEST RESULTS =====")
    print(f"Test reviews: {len(all_labels)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Rows = actual; columns = predicted")
    print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
