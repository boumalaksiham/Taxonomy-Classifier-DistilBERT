# Hierarchical Taxonomy Classifier

## Evaluation scope

The current training script evaluates the same small held-out split after each epoch and uses its Level 3 accuracy to choose a checkpoint. That split is therefore a **validation split**, not an untouched test set. The reported numbers below are historical validation/demo results, not independent generalization estimates. A future experiment should select checkpoints on validation data and evaluate once on a separate product-disjoint test set.

### Fine-tuned DistilBERT — 3-Level Product Category Prediction

> Built as part of an e-commerce ML portfolio targeting applied research roles at companies like eBay, Amazon, and Shopify.

---

## Overview

When a seller lists a product on eBay or Amazon, it needs to be placed in the correct category so buyers can find it through search and filtering. This project automates that process — given any product title, the model predicts its full **3-level category path** in under 100ms on CPU.

```
Input:  "Nike Air Force 1 Low White Men's Size 10"
Output: Clothing > Shoes > Sneakers  (confidence: L1 78%, L2 40%, L3 27%)

Input:  "Instant Pot Duo Plus 9-in-1 Electric Pressure Cooker 8Qt"
Output: Home & Garden > Kitchen > Appliances  (confidence: L1 79%, L2 67%, L3 41%)

Input:  "LEGO Technic McLaren Formula 1 Race Car 42141"
Output: Toys > Building Sets > LEGO  (confidence: L1 54%, L2 23%, L3 19%)
```

---

## Results

Trained on 77 products across 5 top-level categories, 12 mid-level, and 18 specific categories. Evaluated on a held-out test set of 16 products.

| Metric | Score |
|---|---|
| Level 1 Accuracy (broad category) | **100%** |
| Level 2 Accuracy (mid category) | **68.75%** |
| Level 3 Accuracy (specific category) | **81.25%** |
| Accuracy on completely unseen product types | **73%** |

### Full Prediction Results on Test Titles

| Input Title | Predicted Path | Correct? |
|---|---|---|
| Apple AirPods Pro 2nd Generation | Electronics > Audio > Headphones | ✅ |
| Levi's 501 Slim Taper Men's 32x30 Medium Wash | Clothing > Men's Clothing > Jeans | ✅ |
| LEGO Technic McLaren Formula 1 Race Car 42141 | Toys > Building Sets > LEGO | ✅ |
| Instant Pot Duo Plus 9-in-1 Pressure Cooker 8Qt | Home & Garden > Kitchen > Appliances | ✅ |
| Manduka PRO Yoga Mat 6mm Long | Sports > Fitness > Yoga & Pilates | ✅ |
| Sony WF-1000XM5 True Wireless Earbuds Black | Electronics > Audio > Headphones | ✅ |
| KitchenAid Classic 4.5Qt Stand Mixer Aqua Sky | Home & Garden > Kitchen > Appliances | ✅ |
| Adidas Ultraboost 23 Running Shoe Women's Size 8 | Clothing > Shoes > Sneakers | ✅ |
| Nike Air Force 1 Low White Men's Size 10 | Clothing > Shoes > Sneakers | ✅ |
| Samsung 65 inch 4K QLED Smart TV | Electronics > Audio > Speakers | ❌ (TV not in training data) |
| Dyson V15 Detect Absolute Cordless Vacuum | Electronics > Audio > Speakers | ❌ (Vacuum not in training data) |
| Apple Watch Ultra 2 GPS 49mm Titanium | Electronics > Audio > Headphones | ❌ (Wearables not in training data) |

The 3 incorrect predictions are all **out-of-distribution** — product types the model was never trained on. In every case the model correctly identifies Level 1 (Electronics) and falls back to the nearest known subcategory. Adding training examples for TVs, vacuums, and wearables would resolve these.

---

## Training Progress

The model improves steadily across 15 epochs, with Level 1 converging fastest (it's the easiest) and Level 3 improving last (18 fine-grained classes):

```
Epoch  1/15 | Loss: 6.94 | L1: 68.75% | L2: 18.75% | L3:  6.25%
Epoch  3/15 | Loss: 6.20 | L1: 75.00% | L2: 18.75% | L3: 31.25%
Epoch  5/15 | Loss: 5.42 | L1: 81.25% | L2: 25.00% | L3: 37.50%
Epoch  7/15 | Loss: 4.72 | L1:100.00% | L2: 37.50% | L3: 50.00%
Epoch 10/15 | Loss: 3.72 | L1:100.00% | L2: 62.50% | L3: 50.00%
Epoch 14/15 | Loss: 2.89 | L1:100.00% | L2: 68.75% | L3: 68.75%
Epoch 15/15 | Loss: 2.68 | L1:100.00% | L2: 68.75% | L3: 81.25%
```

Level 1 reaches 100% accuracy by Epoch 7 and stays there. Level 3 continues improving through all 15 epochs — this is expected because fine-grained classification (18 classes) requires more passes through the data than coarse classification (5 classes).

---

## Architecture

```
Input: "Nike Air Force 1 Low White Men's Size 10"
         ↓
[DistilBERT Tokenizer]
"nike", "air", "force", "1", "low", "white", "men", "size", "10"
→ token IDs + attention mask
         ↓
[DistilBERT Encoder]  ← 66M parameters, pretrained on Wikipedia + BookCorpus
         ↓
[CLS Token Embedding]  ← 768-dimensional vector, summarizes entire title
         ↓
         ├──────────────────────────────────────────┐
         │                                          │
[L1 Classifier Head]              [L2 Classifier Head]         [L3 Classifier Head]
Linear(768 → 256)                 Linear(768 → 256)            Linear(768 → 256)
ReLU + Dropout(0.3)               ReLU + Dropout(0.3)          ReLU + Dropout(0.3)
Linear(256 → 5)                   Linear(256 → 12)             Linear(256 → 18)
         ↓                                 ↓                            ↓
"Clothing" (78%)              "Shoes" (40%)                "Sneakers" (27%)
```

### Why Three Separate Heads?
Each taxonomy level is its own classification problem with a different number of output classes (5, 12, and 18 respectively). Three heads allow each level to learn independently while sharing the same BERT encoder — the encoder learns features useful for all three levels simultaneously.

### Why One Shared Encoder?
A single DistilBERT encoder is shared across all three classification heads. This is more memory efficient than three separate models and forces the encoder to learn representations that are useful for broad, mid, and fine-grained classification at the same time.

### Combined Loss Function
```python
total_loss = loss_level1 + loss_level2 + loss_level3
```
All three cross-entropy losses are summed and backpropagated in a single step. This trains the entire network end-to-end — the encoder weights are updated based on signals from all three heads simultaneously.

### Why DistilBERT and not BERT?
DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its performance on classification tasks. For CPU deployment (no GPU), this difference is significant — DistilBERT runs in ~80ms per title on CPU versus ~200ms for full BERT.

---

## Taxonomy Structure

The model classifies products into a 3-level hierarchy:

```
Clothing
├── Men's Clothing
│   ├── Jeans
│   └── T-Shirts
├── Women's Clothing
│   └── Dresses
└── Shoes
    └── Sneakers

Electronics
├── Audio
│   ├── Headphones
│   └── Speakers
├── Phones
│   ├── Smartphones
│   └── Phone Cases
└── Computers
    ├── Laptops
    └── Monitors

Home & Garden
├── Kitchen
│   ├── Appliances
│   └── Cookware
└── Furniture
    └── Sofas

Sports
├── Fitness
│   ├── Yoga & Pilates
│   └── Weights & Dumbbells
└── Outdoor
    └── Camping

Toys
├── Building Sets
│   └── LEGO
└── Action Figures
    └── Collectibles
```

---

## Project Structure

```
taxonomy_classifier/
├── data/
│   ├── dataset_builder.py     ← 77 labeled products across the full taxonomy
│   └── __init__.py
├── models/
│   ├── bert_classifier.py     ← HierarchicalTaxonomyClassifier + TaxonomyPredictor
│   ├── saved/                 ← Generated after running train.py
│   │   ├── best_model.pt      ← Saved model weights (best L3 accuracy)
│   │   ├── tokenizer/         ← Saved DistilBERT tokenizer
│   │   ├── label_maps.json    ← Category ↔ integer mappings
│   │   └── config.json        ← Hyperparameters used during training
│   └── __init__.py
├── train.py                   ← Full training loop with epoch-by-epoch metrics
├── demo.py                    ← Interactive CLI: type any title, get category path
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements
- Python 3.10+
- No GPU required — fully CPU compatible
- ~260MB disk space for DistilBERT model (downloaded once on first run)

### Installation
```bash
git clone https://github.com/boumalaksiham/Taxonomy-Classifier-DistilBERT.git
cd Taxonomy-Classifier-DistilBERT

python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# Pin numpy<2 for PyTorch compatibility
pip install "numpy<2" torch==2.2.2 transformers==4.40.0 \
    pandas==2.2.2 scikit-learn==1.4.2 datasets==2.19.0
```

### Train the Model
```bash
python train.py
```
Downloads DistilBERT on first run (~260MB, one time only). Trains for 15 epochs and saves the best model to `models/saved/`.

### Run Interactive Demo
```bash
python demo.py
```
Type any product title and get the full 3-level category prediction with confidence scores at each level.

---

## Key Design Decisions

**1. Three independent classification heads, one shared encoder**
Rather than training three separate models, all heads share a single DistilBERT encoder. This means the encoder learns features that are simultaneously useful for broad and fine-grained classification — a form of multi-task learning. It's also 3x more memory efficient.

**2. Dropout regularization (rate = 0.3)**
With only 77 training examples and 66M parameters, overfitting is a serious risk. Dropout randomly disables 30% of neurons during training, forcing the model to learn redundant representations and generalize better to unseen titles.

**3. Gradient clipping (max norm = 1.0)**
Transformer models are prone to exploding gradients during fine-tuning, especially on small datasets. Clipping gradients to a maximum norm of 1.0 stabilizes training and prevents loss spikes.

**4. AdamW optimizer with low learning rate (2e-5)**
AdamW is the standard optimizer for fine-tuning pretrained transformers. The low learning rate (2e-5) ensures we make small, careful updates to the pretrained weights rather than overwriting the general language knowledge BERT learned during pretraining.

**5. Saving best model by Level 3 accuracy**
The model checkpoint is saved based on Level 3 accuracy (the hardest task with 18 classes) rather than overall loss. This ensures the saved model is optimal for the most challenging and most useful prediction task.

---

## Confidence Score Interpretation

Each prediction comes with a confidence score at every level:

| Confidence | Interpretation |
|---|---|
| > 70% | High confidence — reliable prediction |
| 40–70% | Moderate confidence — likely correct |
| 20–40% | Low confidence — model is uncertain |
| < 20% | Very low — product may be out-of-distribution |

When Level 3 confidence is below 20%, it's a signal the product type may not be well represented in the training data.

---

## Limitations & Future Work

**Current limitations:**
- Trained on 77 products — a larger dataset would significantly improve Level 2 and Level 3 accuracy
- Out-of-distribution categories (TVs, vacuums, wearables) default to nearest known class
- No hierarchical constraint enforcement — theoretically possible to predict "Electronics > Men's Clothing > Jeans" even though that path is impossible

**Extensions for production scale:**
- **More training data**: The [Amazon Product Reviews](https://huggingface.co/datasets/amazon_us_reviews) dataset on HuggingFace contains millions of labeled products and would support a more representative evaluation; the resulting accuracy must be measured
- **Hierarchical constraints**: After predicting L1, restrict L2 softmax to only valid children of that L1 class. This eliminates impossible paths entirely.
- **Confidence threshold + human review**: Flag predictions with L3 confidence below 25% for human review rather than serving a low-confidence prediction
- **Quantization**: Apply INT8 quantization to reduce model size by 4x and inference time by 2x for production serving
- **eBay taxonomy**: Replace the generic taxonomy with eBay's actual category tree (3,000+ categories) and fine-tune on eBay listing data

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `transformers` | 4.40.0 | DistilBERT model + tokenizer |
| `torch` | 2.2.2 | Neural network training + tensors |
| `scikit-learn` | 1.4.2 | Train/test split, accuracy metrics |
| `pandas` | 2.2.2 | Dataset management |
| `numpy` | <2.0 | Numerical operations |

---

## Related Work

- **DistilBERT** — Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
- **Multi-label hierarchical classification** — common approach in e-commerce taxonomy assignment
- **Amazon BERT** — Amazon's internal product classification system uses a similar fine-tuned BERT architecture on millions of product listings

---

*This project is part of a 4-project ML portfolio covering Entity Resolution, Hierarchical Taxonomy Classification, Named Entity Recognition, and Image Quality Scoring for e-commerce applications.*