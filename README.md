# 7015

## Medical Visual Question Answering on VQA-RAD

This repository contains two Med-VQA implementations:

1. CNN-LSTM baseline with frozen ResNet50
2. BLIP-based Vision-Language Model with discriminative fine-tuning

## 📊 Data Preparation
To ensure a fair comparison between the CNN-LSTM baseline and the BLIP model, we implemented a strictly image-disjoint split strategy.

- Method: Grouped questions by unique image MD5 hash.
- Ratio: Train (~65%) / Val (~15%) / Test (~20%).
- Reproducibility: A fixed random seed (`SEED=42`) is used across all scripts.
- Why: This prevents data leakage (same patient's image appearing in both train and test) and guarantees both models are evaluated on the exact same held-out test set.

### Dataset
- VQA-RAD
- https://huggingface.co/datasets/flaviagiammarino/vqa-rad/viewer/default/train?p=1&views%5B%5D=train

### Files
- cnn_lstm_vqa.ipynb
- blip_vqa.ipynb

### Environment
- PyTorch
- Transformers
- Kaggle GPU
