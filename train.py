"""
Training Script — Hierarchical Taxonomy Classifier
====================================================

What this script does:
    1. Loads product title + category dataset
    2. Tokenizes all titles using DistilBERT tokenizer
    3. Builds the model with 3 classification heads
    4. Trains for N epochs using combined cross-entropy loss
    5. Evaluates accuracy at each taxonomy level
    6. Saves the trained model to disk

Loss function:
    total_loss = loss_l1 + loss_l2 + loss_l3
    We sum the losses from all 3 heads and backpropagate once.
    This forces the shared BERT encoder to learn features 
    useful for ALL levels simultaneously.

Usage:
    python train.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_builder import build_dataframe, get_label_maps
from models.bert_classifier import HierarchicalTaxonomyClassifier


# ─────────────────────────────────────────────
# HYPERPARAMETERS — easy to tune from one place
# ─────────────────────────────────────────────
CONFIG = {
    "model_name": "distilbert-base-uncased",   # smaller BERT, CPU-friendly
    "max_length": 64,                           # max token length per title
    "batch_size": 4,                            # small batch for CPU
    "epochs": 15,                                # increase for better accuracy
    "learning_rate": 2e-5,                      # standard for fine-tuning BERT
    "dropout_rate": 0.3,
    "test_size": 0.2,                           # 20% held out for evaluation
    "random_state": 42,
}


class ProductDataset(Dataset):
    """
    PyTorch Dataset — wraps our DataFrame so DataLoader can batch it.

    What is a Dataset?
        PyTorch needs data in a specific format.
        A Dataset tells PyTorch: "here's how to get item number N."
        The DataLoader then automatically batches and shuffles.
    """

    def __init__(self, titles: list, labels_l1: list, labels_l2: list,
                 labels_l3: list, tokenizer, max_length: int):
        self.encodings = tokenizer(
            titles,
            max_length=max_length,
            padding="max_length",   # pad short titles to max_length
            truncation=True,        # truncate long titles to max_length
            return_tensors="pt"
        )
        self.labels_l1 = torch.tensor(labels_l1, dtype=torch.long)
        self.labels_l2 = torch.tensor(labels_l2, dtype=torch.long)
        self.labels_l3 = torch.tensor(labels_l3, dtype=torch.long)

    def __len__(self):
        return len(self.labels_l1)

    def __getitem__(self, idx):
        """Returns one sample as a dict of tensors."""
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label_l1": self.labels_l1[idx],
            "label_l2": self.labels_l2[idx],
            "label_l3": self.labels_l3[idx],
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs ONE full pass through the training data.

    For each batch:
        1. Forward pass → get predictions
        2. Compute loss (how wrong were we?)
        3. Backward pass → compute gradients
        4. Optimizer step → update weights
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Move data to device (CPU in our case)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_l1 = batch["label_l1"].to(device)
        labels_l2 = batch["label_l2"].to(device)
        labels_l3 = batch["label_l3"].to(device)

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass
        logits_l1, logits_l2, logits_l3 = model(input_ids, attention_mask)

        # Compute loss for each level
        loss_l1 = criterion(logits_l1, labels_l1)
        loss_l2 = criterion(logits_l2, labels_l2)
        loss_l3 = criterion(logits_l3, labels_l3)

        # Combined loss — all 3 levels trained simultaneously
        total_batch_loss = loss_l1 + loss_l2 + loss_l3

        # Backward pass
        total_batch_loss.backward()

        # Gradient clipping — prevents exploding gradients (common in BERT training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += total_batch_loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluates the model on a dataset (no weight updates).
    Returns accuracy at each taxonomy level.
    """
    model.eval()

    all_preds_l1, all_preds_l2, all_preds_l3 = [], [], []
    all_true_l1, all_true_l2, all_true_l3 = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_l1, logits_l2, logits_l3 = model(input_ids, attention_mask)

            # Get predictions (argmax = index of highest logit)
            all_preds_l1.extend(torch.argmax(logits_l1, dim=1).cpu().tolist())
            all_preds_l2.extend(torch.argmax(logits_l2, dim=1).cpu().tolist())
            all_preds_l3.extend(torch.argmax(logits_l3, dim=1).cpu().tolist())

            all_true_l1.extend(batch["label_l1"].tolist())
            all_true_l2.extend(batch["label_l2"].tolist())
            all_true_l3.extend(batch["label_l3"].tolist())

    acc_l1 = accuracy_score(all_true_l1, all_preds_l1)
    acc_l2 = accuracy_score(all_true_l2, all_preds_l2)
    acc_l3 = accuracy_score(all_true_l3, all_preds_l3)

    return acc_l1, acc_l2, acc_l3


def main():
    print("=" * 60)
    print("TAXONOMY CLASSIFIER — TRAINING")
    print("=" * 60)

    device = torch.device("cpu")  # CPU-only
    print(f"Device: {device}")

    # ── 1. Load data ─────────────────────────────────────────
    print("\nLoading dataset...")
    df = build_dataframe()
    label_maps = get_label_maps(df)

    print(f"  Products: {len(df)}")
    print(f"  Level 1 categories ({label_maps['num_l1']}): {list(label_maps['l1_to_idx'].keys())}")
    print(f"  Level 2 categories ({label_maps['num_l2']}): {list(label_maps['l2_to_idx'].keys())}")
    print(f"  Level 3 categories ({label_maps['num_l3']}): {list(label_maps['l3_to_idx'].keys())}")

    # Convert string labels to integers
    df["label_l1"] = df["level_1"].map(label_maps["l1_to_idx"])
    df["label_l2"] = df["level_2"].map(label_maps["l2_to_idx"])
    df["label_l3"] = df["level_3"].map(label_maps["l3_to_idx"])

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=df["label_l1"]
    )
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # ── 2. Tokenizer ─────────────────────────────────────────
    print(f"\nLoading tokenizer: {CONFIG['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # ── 3. Datasets & DataLoaders ─────────────────────────────
    train_dataset = ProductDataset(
        train_df["title"].tolist(),
        train_df["label_l1"].tolist(),
        train_df["label_l2"].tolist(),
        train_df["label_l3"].tolist(),
        tokenizer, CONFIG["max_length"]
    )

    test_dataset = ProductDataset(
        test_df["title"].tolist(),
        test_df["label_l1"].tolist(),
        test_df["label_l2"].tolist(),
        test_df["label_l3"].tolist(),
        tokenizer, CONFIG["max_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ── 4. Model ──────────────────────────────────────────────
    print(f"\nBuilding model...")
    model = HierarchicalTaxonomyClassifier(
        num_l1=label_maps["num_l1"],
        num_l2=label_maps["num_l2"],
        num_l3=label_maps["num_l3"],
        model_name=CONFIG["model_name"],
        dropout_rate=CONFIG["dropout_rate"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── 5. Loss & Optimizer ───────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # AdamW is the standard optimizer for fine-tuning transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # ── 6. Training Loop ──────────────────────────────────────
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    print("-" * 60)

    best_l3_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc_l1, acc_l2, acc_l3 = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}/{CONFIG['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"L1 Acc: {acc_l1:.2%} | "
              f"L2 Acc: {acc_l2:.2%} | "
              f"L3 Acc: {acc_l3:.2%}")

        # Save best model based on Level 3 accuracy (hardest task)
        if acc_l3 > best_l3_acc:
            best_l3_acc = acc_l3
            os.makedirs("models/saved", exist_ok=True)
            torch.save(model.state_dict(), "models/saved/best_model.pt")

    print(f"\nBest Level 3 accuracy: {best_l3_acc:.2%}")

    # ── 7. Save everything needed for inference ───────────────
    tokenizer.save_pretrained("models/saved/tokenizer")

    with open("models/saved/label_maps.json", "w") as f:
        json.dump(label_maps, f, indent=2)

    with open("models/saved/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n✅ Training complete! Saved to models/saved/")
    print("   Now run: python demo.py")


if __name__ == "__main__":
    main()
