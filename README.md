# 7015 Medical Visual Question Answering on VQA-RAD

This repository contains two Med-VQA implementations developed for the VQA-RAD dataset. The project compares a classical deep learning baseline against a modern Vision-Language Model to analyze their performance in the medical domain.

## 🧠 Models Implemented

1.  **CNN-LSTM Baseline**
    * A classical approach utilizing a **frozen ResNet50** (pre-trained on ImageNet) for visual feature extraction.
    * Uses a Bidirectional LSTM for text processing.
    * Features a projection layer to bridge the dimension gap between visual (2048) and text (512) embeddings.

2.  **BLIP-Based VLM**
    * A modern Vision-Language Model (`Salesforce/blip-vqa-base`) adapted for **discriminative fine-tuning**.
    * Features a **frozen vision encoder** to retain pre-trained features and a fine-tuned text encoder with Cross-Attention.
    * Uses a custom classification head for closed-set VQA tasks.

## 📂 Dataset

**VQA-RAD** (Visual Question Answering in Radiology)
* Source: [HuggingFace Dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad/viewer/default/train?p=1&views%5B%5D=train)
* A dataset of clinically generated visual questions and answers about radiology images.

## 📊 Data Preparation (Crucial)

To ensure a rigorously fair comparison between the CNN-LSTM baseline and the BLIP model, we implemented a strictly **image-disjoint split strategy** rather than using random splits.

* **Method:** Grouped questions by unique image MD5 hash.
* **Ratio:** Train (~65%) / Val (~15%) / Test (~20%).
* **Reproducibility:** A fixed random seed (`SEED=42`) is used across all scripts.
* **Why:** This prevents data leakage (ensuring the same patient's image never appears in both training and testing sets) and guarantees both models are evaluated on the **exact same held-out test set**.

## 🏆 Results

Both models were evaluated on the held-out test set defined above.

| Model | Overall Accuracy | Closed (Yes/No) | Open-Ended | F1 Score |
| **CNN-LSTM** | **60.25%** | **66.14%** | **39.44%** | **0.593** |
| BLIP (Frozen Vision) | 56.21% | 64.94% | 25.35% | 0.534 |

### 💡 Key Findings
* **The Accuracy Paradox:** The simpler CNN-LSTM baseline outperformed the pre-trained BLIP model, particularly on Open-Ended questions (39.44% vs 25.35%).
* **Domain Gap:** Since the BLIP vision encoder was **frozen** and pre-trained on natural images (COCO), it struggled to extract meaningful features from grayscale medical scans.
* **Qualitative Difference:** Although BLIP had lower accuracy, qualitative analysis showed it produced more linguistically coherent errors (e.g., predicting incorrect organ names) compared to the CNN's tendency to guess "Yes/No".

## 🛠️ Environment & Usage

### Requirements
* Python 3.x
* PyTorch
* Transformers (HuggingFace)
* **Hardware:** Recommended to run on **Kaggle GPU (P100/T4)** or similar.

### Files
* `cnn_lstm_vqa.ipynb`: Complete training and evaluation pipeline for the CNN-LSTM model.
* `blip_vqa.ipynb`: Fine-tuning and evaluation pipeline for the BLIP model.

To reproduce the results, simply open the notebooks in Kaggle or Colab and run all cells. The dataset will be automatically downloaded via the HuggingFace API.

## 👥 Authors
* CR - Model Implementation & Analysis
* LR - Experimental Design & Interpretation
