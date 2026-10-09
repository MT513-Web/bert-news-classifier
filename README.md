BERT-Based News Classification System

## Project Overview
This project implements a production-grade news classification dashboard using fine-tuned BERT (Bidirectional Encoder Representations from Transformers). The model categorizes real-time breaking news headlines into four distinct domains with high contextual accuracy.

## Performance & System Specs
* **Architecture:** Fine-Tuned BERT Base (Uncased)
* **Benchmark Accuracy:** 93.8% Evaluation F1 / 94.5% Validation Accuracy
* **Inference Engine:** PyTorch (CPU-Optimized)
* **Interface:** Streamlit Dashboard (Dark Mode UI)
* **Embedding Depth:** 768-Dimensional Contextual Vectors

## Key Features
* **Transformer-Powered Inference:** Deep contextual classification via Hugging Face & PyTorch.
* **Instant Sample Testing:** Quick test buttons for Tech, Sports, Business, and World news headlines.
* **Confidence & Uncertainty Handling:** Dynamic probability distribution breakdown and confidence grading.
* **Latency Benchmarking:** Displays real-time CPU inference speed per request (~100–200ms).

## Project Structure
* `app.py`: Main Streamlit web application and inference pipeline.
* `requirements.txt`: Python package dependencies.
* `README.md`: Technical documentation and setup instructions.

## Setup & Installation

### 1. Download Model Weights
Due to GitHub file size limits, model weights (~420MB) are hosted externally:
* **Download Link:** [Download Model Weights from Google Drive](https://drive.google.com/file/d/1s_jaTa_7LvUbkE-EASBdh4lTjSakDiLy/view?usp=drive_link)
* **Instruction:** Extract folder `news_classifier_bert_v1` into the project root directory.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
