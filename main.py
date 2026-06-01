"""
Main orchestrator script
"""

from dataset_loader import load_data
from train import train
from inference import inference, forest_inference

print("Loading data...")
load_data()

print("Training model...")
train()

print("Running inference...")
inference()

print("Running amended inference")
for i in range(20):
    forest = forest_inference()
    print(f"Sample {i+1}:\n",forest,"\n",''.join([token[0] for token in forest if token[0]]),"\n")
