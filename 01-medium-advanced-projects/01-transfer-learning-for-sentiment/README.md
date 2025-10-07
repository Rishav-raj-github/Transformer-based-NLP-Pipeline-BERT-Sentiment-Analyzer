# Module 1: Transfer Learning for Sentiment Analysis

Status: Active | Difficulty: Medium-Advanced | Est. Time: 6–10 hours

Objective
- Fine-tune BERT-family models for sentiment classification (binary/multi-class) with robust evaluation and export-ready artifacts.

Learning Outcomes
- Apply transfer learning using BERT/mBERT/XLM-R on domain datasets
- Implement reproducible training with configs and seeds
- Run hyperparameter optimization with Optuna
- Evaluate with accuracy, F1, ROC-AUC, confusion matrices
- Export models to TorchScript and ONNX for deployment

Structure
```
01-transfer-learning-for-sentiment/
├── data/
│   ├── raw/               # Original datasets (IMDb, SST-2, custom)
│   ├── processed/         # Tokenized and split data
│   └── README.md          # Data sources and licenses
├── notebooks/
│   ├── 01_prepare_data.ipynb
│   ├── 02_finetune_bert.ipynb
│   └── 03_evaluate_export.ipynb
├── src/
│   ├── datasets.py        # Dataset loaders and tokenization
│   ├── train.py           # Training loop, metrics
│   ├── evaluate.py        # Evaluation routines
│   └── export.py          # TorchScript/ONNX export
├── configs/
│   ├── train.yaml         # Hyperparameters & paths (Hydra)
│   └── hpo.yaml           # Optuna search space
└── README.md
```

Recommended Datasets
- IMDb (binary), SST-2 (binary), Yelp (multi-class), Amazon Reviews (domain)
- For multilingual: XNLI or custom translated reviews

Quickstart
```bash
# Install deps
pip install -r requirements.txt

# Prepare data (example: IMDb)
python -m src.datasets --dataset imdb --output data/processed/imdb

# Train baseline
python -m src.train \
  model.name=bert-base-uncased \
  data.path=data/processed/imdb \
  training.batch_size=16 training.max_epochs=3 \
  optim.lr=2e-5 seed=42

# Hyperparameter search
python -m src.train +hpo=optuna n_trials=25

# Evaluate and export
python -m src.evaluate --ckpt models/finetuned/best.pt --data data/processed/imdb
python -m src.export --ckpt models/finetuned/best.pt --format onnx --out models/onnx/best.onnx
```

Best Practices
- Use stratified splits and consistent seeds
- Track experiments in MLflow (metrics, params, artifacts)
- Save label maps and tokenizer configs with model
- Validate generalization with cross-domain evaluation

Next Steps
- Move to Module 2: Multilingual Sentiment Analysis
- Integrate FastAPI endpoint from Module 4 for serving
- Benchmark ONNX vs PyTorch inference latencies
