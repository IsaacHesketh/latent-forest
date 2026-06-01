"""
Training script
"""

from model_arch import params, block_size, gpt, softmax
from tokeniser import docs, BOS, uchars
from model_arch import n_layer


def train():
    # Let there be Adam, the blessed optimizer and its buffers
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)  # first moment buffer
    v = [0.0] * len(params)  # second moment buffer

    # Repeat in sequence
    num_steps = 250  # Number of training steps
    for step in range(num_steps):

        # Take a single document, tokenise, surround with BOS token on both sides
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building the computation graph all the way to the loss
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(
            losses
        )  # final average loss over the document sequence. May yours be low.

        # Backward the loss, calculating the gradients with respect to all model parameters.
        loss.backward()

        # Adam optimizer update: update the model parameters based on the corresponding gradients.
        lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0

        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
