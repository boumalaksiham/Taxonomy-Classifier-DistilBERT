"""
Hierarchical Taxonomy Classifier — Fine-tuned BERT
====================================================

Architecture:
    Input: product title (string)
        ↓
    [BERT Encoder]  ← pretrained transformer, frozen or fine-tuned
        ↓
    [CLS token embedding]  ← 768-dimensional vector representing the whole title
        ↓
    ┌──────────────┬──────────────┬──────────────┐
    [Level 1 Head] [Level 2 Head] [Level 3 Head]   ← three separate classifiers
    Electronics    Audio          Headphones
    Clothing       Phones         Jeans
    Home & Garden  ...            ...
    ...

Why three heads?
    Each taxonomy level is a separate classification task.
    We train all three simultaneously — the shared BERT encoder learns 
    features useful for all three levels.

Why BERT?
    BERT understands context. "Apple" in "Apple iPhone 15 Pro" means a 
    company, not a fruit. "Air" in "Nike Air Max" means a shoe, not weather.
    Simple bag-of-words models miss this.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np


class HierarchicalTaxonomyClassifier(nn.Module):
    """
    A BERT-based model with three classification heads:
        - Head 1: Level 1 category (e.g. Electronics, Clothing, ...)
        - Head 2: Level 2 category (e.g. Audio, Phones, Computers, ...)
        - Head 3: Level 3 category (e.g. Headphones, Smartphones, Jeans, ...)
    """

    def __init__(self, num_l1: int, num_l2: int, num_l3: int,
                 model_name: str = "distilbert-base-uncased",
                 dropout_rate: float = 0.3):
        """
        Args:
            num_l1: Number of Level 1 categories
            num_l2: Number of Level 2 categories
            num_l3: Number of Level 3 categories
            model_name: Pretrained transformer to use.
                        'distilbert-base-uncased' is smaller & faster than BERT
                        but almost as accurate. Great for CPU.
            dropout_rate: Regularization to prevent overfitting
        """
        super().__init__()

        self.model_name = model_name

        # ── SHARED ENCODER ─────────────────────────────────────
        # DistilBERT is 40% smaller than BERT but retains 97% of performance
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for BERT, 768 for DistilBERT

        self.dropout = nn.Dropout(dropout_rate)

        # ── THREE CLASSIFICATION HEADS ─────────────────────────
        # Each head is a small feedforward network that maps
        # the 768-dim BERT output to category logits
        self.classifier_l1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_l1)
        )

        self.classifier_l2 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_l2)
        )

        self.classifier_l3 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_l3)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            input_ids:      Token IDs from the tokenizer (shape: batch_size x seq_len)
            attention_mask: 1 for real tokens, 0 for padding (shape: batch_size x seq_len)

        Returns:
            Tuple of (logits_l1, logits_l2, logits_l3)
            Each is shape (batch_size x num_categories_at_that_level)
        """
        # Step 1: Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Step 2: Extract the [CLS] token representation
        # The [CLS] token (index 0) is a summary of the entire sequence
        # It's what we use for classification tasks
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)
        cls_embedding = self.dropout(cls_embedding)

        # Step 3: Pass through each classification head
        logits_l1 = self.classifier_l1(cls_embedding)
        logits_l2 = self.classifier_l2(cls_embedding)
        logits_l3 = self.classifier_l3(cls_embedding)

        return logits_l1, logits_l2, logits_l3


class TaxonomyPredictor:
    """
    Wraps the trained model for easy inference.
    Call predict(title) and get the full category path back.
    """

    def __init__(self, model: HierarchicalTaxonomyClassifier,
                 tokenizer, label_maps: dict,
                 max_length: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length
        self.model.eval()

    def predict(self, title: str) -> dict:
        """
        Predicts all 3 taxonomy levels for a single product title.

        Returns:
            {
                "title": "Nike Air Max 90 White Size 10",
                "level_1": "Clothing",
                "level_2": "Shoes",
                "level_3": "Sneakers",
                "full_path": "Clothing > Shoes > Sneakers",
                "confidence": { "level_1": 0.97, "level_2": 0.89, "level_3": 0.92 }
            }
        """
        # Tokenize the input title
        encoding = self.tokenizer(
            title,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"   # "pt" = return PyTorch tensors
        )

        with torch.no_grad():   # no_grad saves memory — we're not training
            logits_l1, logits_l2, logits_l3 = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )

        # Convert logits to probabilities using softmax
        probs_l1 = torch.softmax(logits_l1, dim=1)[0]
        probs_l2 = torch.softmax(logits_l2, dim=1)[0]
        probs_l3 = torch.softmax(logits_l3, dim=1)[0]

        # Get the highest-probability class
        pred_l1 = torch.argmax(probs_l1).item()
        pred_l2 = torch.argmax(probs_l2).item()
        pred_l3 = torch.argmax(probs_l3).item()

        label_l1 = self.label_maps["idx_to_l1"][str(pred_l1)]
        label_l2 = self.label_maps["idx_to_l2"][str(pred_l2)]
        label_l3 = self.label_maps["idx_to_l3"][str(pred_l3)]

        return {
            "title": title,
            "level_1": label_l1,
            "level_2": label_l2,
            "level_3": label_l3,
            "full_path": f"{label_l1} > {label_l2} > {label_l3}",
            "confidence": {
                "level_1": round(probs_l1[pred_l1].item(), 4),
                "level_2": round(probs_l2[pred_l2].item(), 4),
                "level_3": round(probs_l3[pred_l3].item(), 4),
            }
        }

    def predict_batch(self, titles: list[str]) -> list[dict]:
        """Predict categories for a list of product titles."""
        return [self.predict(t) for t in titles]
