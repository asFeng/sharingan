"""Attention analysis metrics."""

from __future__ import annotations

import numpy as np


def compute_entropy(attention: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute attention entropy.

    Entropy measures how distributed the attention is:
    - High entropy = attention spread across many tokens
    - Low entropy = attention focused on few tokens

    Args:
        attention: Attention weights (any shape, probabilities along axis)
        axis: Axis along which to compute entropy

    Returns:
        Entropy values with axis dimension removed
    """
    eps = 1e-10
    attn = np.clip(attention, eps, 1.0)
    return -np.sum(attn * np.log(attn), axis=axis)


def find_attention_sinks(
    attention: np.ndarray,
    threshold: float = 0.1,
    min_attending: int = 3,
) -> list[dict]:
    """Find attention sink positions.

    Attention sinks are positions that receive high attention from many positions.
    These are often special tokens (BOS, EOS) or punctuation.

    Args:
        attention: Attention matrix [seq, seq] (aggregated across layers/heads)
        threshold: Minimum average attention to be considered a sink
        min_attending: Minimum positions that must attend to qualify

    Returns:
        List of sink dicts sorted by attention received
    """
    seq_len = attention.shape[0]

    # Attention received by each position (column sums, normalized)
    attn_received = attention.mean(axis=0)

    sinks = []
    for pos in range(seq_len):
        if attn_received[pos] > threshold:
            # Count positions attending above threshold
            attending = np.sum(attention[:, pos] > 0.05)
            if attending >= min_attending:
                sinks.append(
                    {
                        "position": pos,
                        "attention_received": float(attn_received[pos]),
                        "attending_positions": int(attending),
                        "attending_ratio": float(attending / seq_len),
                    }
                )

    return sorted(sinks, key=lambda x: -x["attention_received"])


def compute_token_importance(
    attention: np.ndarray,
    method: str = "received",
) -> np.ndarray:
    """Compute token importance scores from attention.

    Args:
        attention: Attention matrix [seq, seq]
        method: Scoring method
            - "received": Total attention received (column sum)
            - "given": Total attention given (row sum)
            - "both": Geometric mean of received and given
            - "pagerank": PageRank-style iterative scoring

    Returns:
        Importance scores [seq_len], normalized to [0, 1]
    """
    if method == "received":
        scores = attention.sum(axis=0)
    elif method == "given":
        scores = attention.sum(axis=1)
    elif method == "both":
        received = attention.sum(axis=0)
        given = attention.sum(axis=1)
        scores = np.sqrt(received * given)
    elif method == "pagerank":
        scores = _attention_pagerank(attention)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to [0, 1]
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def _attention_pagerank(
    attention: np.ndarray,
    damping: float = 0.85,
    iterations: int = 20,
) -> np.ndarray:
    """Compute PageRank-style importance from attention matrix.

    Args:
        attention: Attention matrix [seq, seq]
        damping: Damping factor (probability of following attention)
        iterations: Number of iterations

    Returns:
        PageRank scores [seq_len]
    """
    n = attention.shape[0]

    # Normalize attention to transition probabilities
    row_sums = attention.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transition = attention / row_sums

    # Initialize uniform
    scores = np.ones(n) / n

    # Iterate
    for _ in range(iterations):
        scores = (1 - damping) / n + damping * (transition.T @ scores)

    return scores


def compute_attention_distance(
    attention: np.ndarray,
) -> dict[str, float]:
    """Compute attention distance statistics.

    Measures how far tokens attend on average.

    Args:
        attention: Attention matrix [seq, seq]

    Returns:
        Dict with distance statistics
    """
    seq_len = attention.shape[0]

    # Create distance matrix
    positions = np.arange(seq_len)
    distance_matrix = np.abs(positions[:, None] - positions[None, :])

    # Weighted average distance (weighted by attention)
    mean_distance = np.sum(attention * distance_matrix) / np.sum(attention)

    # Local vs distant attention
    local_mask = distance_matrix <= 10
    local_attention = np.sum(attention * local_mask) / np.sum(attention)

    return {
        "mean_distance": float(mean_distance),
        "local_attention_ratio": float(local_attention),
        "distant_attention_ratio": float(1 - local_attention),
    }


def layer_head_similarity(
    attention: np.ndarray,
) -> np.ndarray:
    """Compute similarity between attention patterns across layers/heads.

    Args:
        attention: Attention [layers, heads, seq, seq]

    Returns:
        Similarity matrix [layers*heads, layers*heads]
    """
    n_layers, n_heads, seq_len, _ = attention.shape

    # Flatten each layer-head attention to vector
    patterns = attention.reshape(n_layers * n_heads, -1)

    # Normalize
    norms = np.linalg.norm(patterns, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    patterns = patterns / norms

    # Cosine similarity
    similarity = patterns @ patterns.T

    return similarity
