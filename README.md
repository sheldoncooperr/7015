# 7015


### Dataset
- VQA-RAD
- https://huggingface.co/datasets/flaviagiammarino/vqa-rad/viewer/default/train?p=1&views%5B%5D=train

  
## Medical Visual Question Answering on VQA-RAD

This repository contains two Med-VQA implementations:

- CNN-LSTM Baseline: A classical approach utilizing a frozen ResNet50 for visual feature extraction combined with an LSTM for text processing.

- BLIP-Based VLM: A modern Vision-Language Model adapted for discriminative fine-tuning, featuring a frozen vision encoder and a custom classification head for closed-set VQA tasks."


## 📊 Data Preparation
To ensure a fair comparison between the CNN-LSTM baseline and the BLIP model, we implemented a strictly image-disjoint split strategy.

- Method: Grouped questions by unique image MD5 hash.
- Ratio: Train (~65%) / Val (~15%) / Test (~20%).
- Reproducibility: A fixed random seed (`SEED=42`) is used across all scripts.
- Why: This prevents data leakage (same patient's image appearing in both train and test) and guarantees both models are evaluated on the exact same held-out test set.

### Files
- cnn_lstm_vqa.ipynb
- blip_vqa.ipynb

### Environment
- PyTorch
- Transformers
- Kaggle GPU
