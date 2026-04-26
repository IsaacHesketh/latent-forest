"""
Simple dataset loader
"""

import os
import random


def load_data(
        source: str = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
) -> list[str]:
    """
    Loads data from the source URL or the input.txt file
    Args:
        source: Web URL to pull data from

    Returns:
        List of strings on which to train
    """
    if not os.path.exists('input.txt'):
        import urllib.request
        import ssl
        import certifi

        # Set SSL context for certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        # Open url and load to buffer
        with urllib.request.urlopen(source, context=ssl_context) as response:
            with open("input.txt", "wb") as f:
                f.write(response.read())

    # Load buffer and separate by line into a list of lower case names
    docs = [
        l.strip()
        for l in (
            open('input.txt')
            .read()
            .strip()
            .split('\n')
        )
        if l.strip()
    ] # list[str] of documents

    # Randomise order
    random.shuffle(docs)
    return docs

if __name__ == "__main__":
    print(f"num docs: {len(load_data())}")