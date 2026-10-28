
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_length, d_model = x.shape

        Q = self.query(x).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = self.key(x).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = self.value(x).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention scores.
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Ignore padding tokens as attention keys.
        if key_padding_mask is not None:
            # Shape: [batch, 1, 1, sequence_length]
            key_mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attended = weights @ V

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(batch_size, seq_length, d_model)

        return self.output_projection(attended)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x, key_padding_mask=None):
        # Attention + residual connection + normalization.
        x = self.norm1(
            x + self.attention(x, key_padding_mask)
        )

        # Feed-forward network + residual connection + normalization.
        x = self.norm2(x + self.feed_forward(x))

        return x


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_dim, max_length):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_length, d_model)

        self.encoder = TransformerEncoderBlock(
            d_model, num_heads, ff_dim
        )

        self.classifier = nn.Linear(d_model, 2)

    def forward(self, token_ids):
        # True for real tokens, False for padding.
        mask = token_ids != 0

        positions = torch.arange(
            token_ids.shape[1], device=token_ids.device
        ).unsqueeze(0)

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)

        # Pass the padding mask into attention.
        x = self.encoder(x, key_padding_mask=mask)

        # Average real-token representations only.
        mask_float = mask.unsqueeze(-1).to(x.dtype)
        x = (x * mask_float).sum(dim=1)
        x = x / mask_float.sum(dim=1).clamp(min=1)

        return self.classifier(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = SentimentClassifier(
        vocab_size=100,
        d_model=16,
        num_heads=4,
        ff_dim=32,
        max_length=20
    )

    token_ids = torch.tensor([
        [12, 7, 35, 9, 22],
        [18, 4, 0, 0, 0]
    ])

    logits = model(token_ids)

    print("Input shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print("Logits:\n", logits)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters())
    )
