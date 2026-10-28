
import sys
from pathlib import Path

import streamlit as st
import torch

# Make modules inside src/ importable.
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data import encode_text
from model import SentimentClassifier


CHECKPOINT_PATH = PROJECT_DIR / "models" / "sentiment_model_10epochs.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
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
    token_ids = encode_text(review, vocab, config["max_length"])
    input_tensor = torch.tensor(
        [token_ids], dtype=torch.long, device=DEVICE
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_class = int(probabilities.argmax().item())

    return (
        "Positive" if predicted_class == 1 else "Negative",
        float(probabilities[predicted_class].item()),
    )


# ---------- Page configuration ----------
st.set_page_config(
    page_title="MiniTransformer | Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #11111b 0%, #19172b 100%);
        color: #f4f1ff;
    }
    h1, h2, h3 {
        color: #eee9ff;
    }
    .subtitle {
        color: #b8b3ce;
        margin-top: -12px;
        margin-bottom: 28px;
    }
    .result-card {
        padding: 22px;
        border: 1px solid #39334f;
        border-radius: 16px;
        background: rgba(39, 34, 59, 0.8);
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 MiniTransformer")
st.markdown(
    '<p class="subtitle">AI-powered movie review sentiment analysis</p>',
    unsafe_allow_html=True,
)

st.markdown("### Analyze a review")
st.write(
    "Enter a movie review and let the Transformer predict "
    "whether its sentiment is positive or negative."
)

example = st.selectbox(
    "Try an example",
    [
        "Write my own review",
        "Positive example",
        "Negative example",
    ],
)

examples = {
    "Positive example": (
        "This movie was fantastic, with excellent acting, "
        "a compelling story, and a wonderful ending."
    ),
    "Negative example": (
        "The movie was boring and disappointing. "
        "The acting was poor and the story made no sense."
    ),
}

default_text = examples.get(example, "")

if "review_input" not in st.session_state:
    st.session_state.review_input = default_text
elif example in examples:
    st.session_state.review_input = default_text

review = st.text_area(
    "Movie review",
    key="review_input",
    height=180,
    placeholder="Paste or type a movie review here...",
)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            with st.spinner("Analyzing review..."):
                model, vocab, config = load_model()
                sentiment, confidence = predict_sentiment(
                    review, model, vocab, config
                )

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            if sentiment == "Positive":
                st.success("😊 Positive sentiment")
            else:
                st.error("😞 Negative sentiment")

            st.metric("Model confidence", f"{confidence:.2%}")
            st.progress(confidence)

            st.caption(
                "Confidence reflects the model's predicted probability, "
                "not a guarantee that the prediction is correct."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Could not analyze the review: {exc}")

with st.expander("About this model"):
    st.write("**Architecture:** MiniTransformer encoder with multi-head attention")
    st.write("**Dataset:** IMDb movie reviews")
    st.write("**Vocabulary size:** 10,000")
    st.write("**Maximum sequence length:** 200 tokens")
    st.write("**Test accuracy:** 82.5% on the project's 1,000-review test subset")
    st.write("**Framework:** PyTorch")

st.divider()
st.caption("MiniTransformer · A machine learning portfolio project")
