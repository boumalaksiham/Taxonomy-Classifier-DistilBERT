# Hierarchical Product Taxonomy Classification

A multi-task text-classification prototype that predicts three product-category levels from a title using one shared DistilBERT encoder and three classification heads.

## Architecture and data

[data/dataset_builder.py](data/dataset_builder.py) contains **77 curated product titles** spanning **5 broad categories, 12 intermediate categories, and 18 detailed categories**. Label mappings are generated from the dataset and saved for inference.

The encoder is `distilbert-base-uncased`. Three cross-entropy losses are summed and optimized jointly. Independent heads can predict an inconsistent category path; the implementation does not enforce parent-child constraints.

| Training setting | Current value |
|---|---|
| Maximum tokens | 64 |
| Batch size | 4 |
| Epochs | 15 |
| Learning rate | 0.00002 |
| Dropout | 0.3 |
| Split | 61 training / 16 validation titles, stratified by Level 1 |
| Split seed | 42 |

The split is named `test` in the code, but it is used every epoch to select the best Level 3 checkpoint. Its scores are **validation scores**, not untouched-test performance. The split seed does not fully fix model-training randomness.

## Setup

Use Python 3.10 or 3.11 for the repository's pinned PyTorch 2.2.2 environment. Run commands from the repository root. The first model load downloads pretrained weights; an internet connection is needed unless the model cache is populated. A GPU is not required by the current scripts. Runtime depends on hardware and is not benchmarked here.

```bash
git clone https://github.com/boumalaksiham/Taxonomy-Classifier-DistilBERT.git
cd Taxonomy-Classifier-DistilBERT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. If installation reports an unsupported wheel, check Python version and architecture before changing the pinned environment.

## Train and predict

```bash
python train.py
python demo.py
```

Training writes `models/saved/best_model.pt`, `label_maps.json`, `config.json`, and tokenizer files under `models/saved/tokenizer/`. The demo expects these artifacts; checkpoints are not included in the source checkout. Use a trusted checkpoint compatible with the saved label maps and configuration.

## Repository map

| File | Purpose |
|---|---|
| [data/dataset_builder.py](data/dataset_builder.py) | Titles, taxonomy and label mappings |
| [models/bert_classifier.py](models/bert_classifier.py) | Shared encoder and classification heads |
| [train.py](train.py) | Training loop and checkpoint selection |
| [demo.py](demo.py) | Interactive category predictions |

## Interpreting results

Previously reported accuracy values are not reproduced or certified by this README. A credible final report should include sample counts, per-level accuracy, full-path consistency, per-class errors, and predictions on a separate test set. Softmax confidence does not establish calibrated probability or reliable behavior on unknown product types.

## Limitations and next steps

The dataset is small and hand-curated; fine-grained classes have few examples. There is no independent final test set, production deployment, or validated latency benchmark. Add product-disjoint train/validation/test partitions, hierarchical constraints or consistency reporting, stronger baselines, error analysis, and a saved evaluation report before making broader performance claims.
