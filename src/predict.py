
import torch
from pathlib import Path

from data import encode_text
from model import SentimentClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model_10epochs.pth"


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run python src/train.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    model = SentimentClassifier(
        vocab_size=len(vocab),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        max_length=config["max_length"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, config


def predict_sentiment(review, model, vocab, config):
    token_ids = encode_text(
        review,
        vocab,
        config["max_length"],
    )

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_class = int(probabilities.argmax().item())
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    confidence = probabilities[predicted_class].item() * 100

    return sentiment, confidence


if __name__ == "__main__":
    model, vocab, config = load_model()

    print("MiniTransformer Sentiment Predictor")
    print("Enter a movie review, or type 'quit' to exit.")

    while True:
        review = input("\nReview: ").strip()

        if review.lower() == "quit":
            print("Goodbye!")
            break

        if not review:
            print("Please enter a review.")
            continue

        sentiment, confidence = predict_sentiment(
            review, model, vocab, config
        )

        print(f"Predicted sentiment: {sentiment}")
        print(f"Model confidence: {confidence:.2f}%")
