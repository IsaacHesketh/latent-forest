"""
Tokeniser functions
"""
from dataset_loader import load_data
def char_tokeniser():
    pass

docs = load_data()

uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
