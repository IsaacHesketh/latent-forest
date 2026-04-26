# Latent Forest
**Latent Forest** is a visualisation toolkit for exploring the stochastic nature of language models.

Modern language models do not produce a single deterministic continuation of text. At every step, they evaluate a probability distribution over thousands of possible next tokens. Each generated sentence represents just **one path** through a vast forest of alternative possibilities.

Latent forest makes this hidden process visible.

By expanding generation into a branching tree of candidate tokens, the project allows you to:

- Visualise how probabilities evolve during generation
- Explore alternative continuations
- Understand sampling, temperature and randomness intuitively
- Inspect how small choices lead to divergent outputs
- experiment with the dynamics of autoregressive language models

---

## Why this exists
Language models operate within a high dimensional latent space where meaning, structure, and context are encoded numerically. The text we see is only the surface trace of a trajectory through that hidden space.

Latent forest aims to provide an interactive _microscope_ for language model behaviour - helping curious users explore how probabilistic generation actually unfolds.

---

### Core Idea
At each generation step:
1. A model produces a probability distribution over a set of next tokens.
2. Multipl plausible continuations exist simultaneously.
3. Sampling selects one path - while all the others are abandoned

Latent Forest expands these unseen paths into a navigable structure:
```
prompt
├── token A
│    ├── token A1
│    └── token A2
├── token B
│    └── token B1
└── token C
```
The result is a living map of possibility rather than a single output.

---

### Status
🚧Early experimental project - still ironing out the kinks!

## Design Decisions
### Model
The model is inspired by Andrej Karpathy's [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) with some necessary modifications.

### Tokensiser
> **Initial State:** Andrej Karpathy's MicroGPT uses a very simple letter to token mapping with each token being a lowercase letter from the latin alphabet and one additional token *BOS* to mark the beginning of each sequence.
> 
>This is only a placeholder to accelerate the development of the Latent Forest core logic and I will implement a more complex tokeniser at a later date.

Due to the nature of Latent Forest as an experiment and explainability tool, rather than a typical performance based use case, it uses word level tokenisation.

Word level tokenisation splits text into individual words based on whitespace and punctuation. 

**Advantages:** High semantic richness per token and human-readable output.
**Disadvantages:** Inability to handle unseen words (OOV).

### Training Data
> **Initial State:** Similar to the tokeniser, we will directly reuse the dataset format from MicroGPT. The dataset consists of a list of 32,000 names in a simple text file, one per line.

The end state for the dataset is still to be determined. Given my compute constraints and the branching nature of the model, texts should be kept incredibly short by modern GPT standards.