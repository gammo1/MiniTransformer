
import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

MAX_LENGTH = 200
MAX_VOCAB_SIZE = 10000
BATCH_SIZE = 32


def tokenize(text):
    """Convert a review into lowercase word and punctuation tokens."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())


def build_vocabulary(texts, max_vocab_size=MAX_VOCAB_SIZE):
    """Build vocabulary using training reviews only."""
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for token, _ in counts.most_common(max_vocab_size - 2):
        vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length=MAX_LENGTH):
    """Convert tokens into IDs, then truncate or pad."""
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

    ids = ids[:max_length]
    ids += [vocab[PAD_TOKEN]] * (max_length - len(ids))

    return ids


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.inputs = [
            encode_text(text, vocab)
            for text in texts
        ]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.inputs[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def prepare_data():
    """Load all IMDb training reviews and preserve a separate validation set."""
    print("Loading IMDb dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    # Shuffle all 25,000 labeled training reviews deterministically.
    selected_train = dataset["train"].shuffle(seed=42)

    # Preserve the same 1,000 validation reviews used in the earlier experiment.
    val_indices = list(range(4000, 5000))
    train_indices = list(range(0, 4000)) + list(range(5000, 25000))

    train_split = selected_train.select(train_indices)
    val_split = selected_train.select(val_indices)

    # Keep the same 1,000 test reviews for a final comparison.
    test_split = dataset["test"].shuffle(seed=42).select(range(1000))

    train_texts = train_split["text"]
    train_labels = train_split["label"]

    val_texts = val_split["text"]
    val_labels = val_split["label"]

    test_texts = test_split["text"]
    test_labels = test_split["label"]

    print("Building vocabulary from training reviews...")
    vocab = build_vocabulary(train_texts)

    train_dataset = ReviewDataset(train_texts, train_labels, vocab)
    val_dataset = ReviewDataset(val_texts, val_labels, vocab)
    test_dataset = ReviewDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = prepare_data()

    reviews, labels = next(iter(train_loader))

    print("\nData preparation successful!")
    print("Training reviews:", len(train_loader.dataset))
    print("Validation reviews:", len(val_loader.dataset))
    print("Test reviews:", len(test_loader.dataset))
    print("Vocabulary size:", len(vocab))
    print("Batch review tensor shape:", reviews.shape)
    print("Batch label tensor shape:", labels.shape)
    print("First 5 labels:", labels[:5].tolist())
