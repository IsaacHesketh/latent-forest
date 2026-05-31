"""
Main orchestrator script
"""

from dataset_loader import load_data
from train import train
from inference import inference

print("Loading data...")
load_data()

print("Training model...")
train()

print("Running inference...")
inference()