"""
Model inference script
"""

import random
import numpy as np

from tokeniser import BOS, vocab_size, uchars
from model_arch import n_layer, gpt, softmax, block_size

temperature = 0.3  # in (0, 1], control the "creativity" of generated text, low to high


def inference():
    print("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(20):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")


def forest_inference():
    """
    The inference mechanism to create a branching tree structure of potential outcomes.

    Returns:
        A series of tuples, token IDs to probabilities of the most likely token at each stage
    """
    # Initialise empty kv
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

    # Start with beginning of string token
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])

        # Get max index and element in probs
        weights = [prob.data for prob in probs]
        token_id = random.choices(range(vocab_size), weights=weights)
        prob = weights[token_id]
        if token_id == BOS:
            sample.append((None, prob))
            break
        sample.append((uchars[token_id], prob))
    return sample
