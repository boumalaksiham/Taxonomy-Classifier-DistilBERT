"""
Demo — Taxonomy Classifier
===========================
Loads the trained model and lets you type any product title
to see its predicted category path.

Usage:
    python demo.py

Run train.py first to generate the saved model.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.bert_classifier import HierarchicalTaxonomyClassifier, TaxonomyPredictor

SAVE_DIR = "models/saved"


def load_model():
    """Loads saved model, tokenizer, and label maps from disk."""
    if not os.path.exists(f"{SAVE_DIR}/best_model.pt"):
        print("❌ No saved model found. Please run train.py first.")
        sys.exit(1)

    with open(f"{SAVE_DIR}/label_maps.json") as f:
        label_maps = json.load(f)

    with open(f"{SAVE_DIR}/config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(f"{SAVE_DIR}/tokenizer")

    model = HierarchicalTaxonomyClassifier(
        num_l1=label_maps["num_l1"],
        num_l2=label_maps["num_l2"],
        num_l3=label_maps["num_l3"],
        model_name=config["model_name"]
    )
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pt", map_location="cpu"))

    predictor = TaxonomyPredictor(model, tokenizer, label_maps, max_length=config["max_length"])
    return predictor


def print_prediction(result: dict):
    """Pretty prints a single prediction result."""
    print("\n" + "─" * 60)
    print(f"  Title     : {result['title']}")
    print(f"  Level 1   : {result['level_1']:<25} (confidence: {result['confidence']['level_1']:.2%})")
    print(f"  Level 2   : {result['level_2']:<25} (confidence: {result['confidence']['level_2']:.2%})")
    print(f"  Level 3   : {result['level_3']:<25} (confidence: {result['confidence']['level_3']:.2%})")
    print(f"  Full Path : {result['full_path']}")
    print("─" * 60)


def run_preset_examples(predictor):
    """Runs interesting test cases, including some tricky ones."""
    print("\n" + "=" * 60)
    print("  PRESET EXAMPLES")
    print("=" * 60)

    examples = [
        # Clear cases
        "Apple AirPods Pro 2nd Generation",
        "Levi's 501 Slim Taper Men's 32x30 Medium Wash",
        "LEGO Technic McLaren Formula 1 Race Car 42141",
        "Instant Pot Duo Plus 9-in-1 Electric Pressure Cooker 8Qt",
        "Manduka PRO Yoga Mat 6mm Long",

        # Tricky cases — can the model handle these?
        "Sony WF-1000XM5 True Wireless Earbuds Black",   # earbuds = headphones?
        "Apple Watch Ultra 2 GPS 49mm Titanium",          # smartwatch
        "Dyson V15 Detect Absolute Cordless Vacuum",      # home appliance
        "KitchenAid Classic 4.5Qt Tilt-Head Stand Mixer Aqua Sky",
        "Adidas Ultraboost 23 Running Shoe Women's White Size 8",
    ]

    for title in examples:
        result = predictor.predict(title)
        print_prediction(result)


def interactive_mode(predictor):
    """Interactive CLI for typing any product title."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE — Type any product title")
    print("  (Press Ctrl+C to quit)")
    print("=" * 60)

    while True:
        try:
            print()
            title = input("Product Title: ").strip()
            if not title:
                continue
            result = predictor.predict(title)
            print_prediction(result)
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break


def main():
    print("=" * 60)
    print("  TAXONOMY CLASSIFIER DEMO")
    print("  Fine-tuned DistilBERT — 3-Level Hierarchical Classification")
    print("=" * 60)

    print("\nLoading model...")
    predictor = load_model()
    print("Model loaded!")

    run_preset_examples(predictor)
    interactive_mode(predictor)


if __name__ == "__main__":
    main()
