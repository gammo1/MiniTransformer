
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline


def main():
    print("Loading IMDb dataset...")

    dataset = load_dataset("stanfordnlp/imdb")

    # Reproduce the exact split used by the Transformer.
    shuffled_train = dataset["train"].shuffle(seed=42)

    train_indices = list(range(0, 4000)) + list(range(5000, 25000))
    val_indices = list(range(4000, 5000))

    train_data = shuffled_train.select(train_indices)
    val_data = shuffled_train.select(val_indices)

    test_data = dataset["test"].shuffle(seed=42).select(range(1000))

    X_train = train_data["text"]
    y_train = train_data["label"]

    X_val = val_data["text"]
    y_val = val_data["label"]

    X_test = test_data["text"]
    y_test = test_data["label"]

    print(f"Training reviews: {len(X_train)}")
    print(f"Validation reviews: {len(X_val)}")
    print(f"Test reviews: {len(X_test)}")

    # Same TF-IDF and Logistic Regression settings as the original baseline.
    baseline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english",
            ),
        ),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    baseline.fit(X_train, y_train)

    # Validation is reported separately; it is not used to fit the model.
    val_predictions = baseline.predict(X_val)
    print(f"\nValidation accuracy: {accuracy_score(y_val, val_predictions):.4f}")

    # Evaluate once on the held-out test set.
    test_predictions = baseline.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)

    print("\n===== TF-IDF BASELINE TEST RESULTS =====")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Positive-class F1-score: {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Negative", "Positive"],
        )
    )


if __name__ == "__main__":
    main()
