# Bert-arabic-fake-news-rbf
## Overview
Bert-arabic-fake-news-rbf is a project aimed at detecting fake news in Arabic using BERT embeddings combined with two different classification approaches: a basic dense neural network head and an SVM with RBF kernel. This project leverages the power of BERT (CAMeL-Lab/bert-base-arabic-camelbert-msa) for natural language understanding and compares two classification approaches.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To train both models using the pre-configured script:
```bash
bash run_scripts.sh
```

Or train models individually:

**Basic Model:**
```bash
python train.py --device cuda --EPOCHS 10 --BATCH_SIZE 32 --model_id CAMeL-Lab/bert-base-arabic-camelbert-msa --model_type basic
```

**RBF/SVM Model:**
```bash
python train.py --device cuda --EPOCHS 10 --BATCH_SIZE 32 --model_id CAMeL-Lab/bert-base-arabic-camelbert-msa --model_type rbf
```

## Dataset
**Name:** SANAD: Single-Label Arabic News Articles Dataset
**Source:** Kaggle (haithemhermessi/sanad-dataset)

The dataset consists of Arabic news articles categorized into multiple classes. The data is automatically downloaded and prepared using the kagglehub API.

**Data Split:**
- Training set: 70%
- Validation set: 15%
- Testing set: 15%
- Stratified splitting is used to maintain class distribution

## Model Architecture

### Basic Model
The basic model architecture consists of:
1. **BERT Embeddings** (frozen): CAMeL-Lab/bert-base-arabic-camelbert-msa converts text into 768-dimensional vectors
2. **Dense Classification Head**:
   - Linear layer: 768 → 384
   - Dropout: 0.25
   - Linear layer: 384 → num_classes

### RBF/SVM Model
The RBF model architecture consists of:
1. **BERT Embeddings** (frozen): CAMeL-Lab/bert-base-arabic-camelbert-msa extracts 768-dimensional features
2. **SVM Classifier**: Support Vector Machine with RBF kernel (GPU-accelerated using cuML)

## Training

### Basic Model Training
- **Optimizer:** AdamW
- **Learning Rate:** 5e-5
- **Loss Function:** CrossEntropyLoss
- **Training Process:** End-to-end gradient-based training for the classification head
- **Epochs:** 10 (configurable)
- **Batch Size:** 32 (configurable)

### RBF/SVM Model Training
- **Feature Extraction:** BERT embeddings extracted in batches
- **Training:** SVM fit on pre-computed embeddings
- **GPU Acceleration:** cuML's GPU-accelerated SVM implementation

## Hyperparameters

### Reproducibility
- **Random Seed:** 42 (set for Python random, NumPy, PyTorch, and Transformers)

### Basic Model Hyperparameters
- **Learning Rate:** 5e-5
- **Optimizer:** AdamW
- **Epochs:** 10
- **Batch Size:** 32
- **Dropout:** 0.25
- **Architecture:** BERT with Dense Classification Head

### RBF/SVM Model Hyperparameters
- **Kernel:** rbf
- **Degree:** 6
- **Gamma:** scale
- **C (Regularization):** 9
- **Batch Size:** 32 (for embedding extraction)
- **Architecture:** BERT with SVM Classification Head

### Training Environment
Experiments were conducted on an NVIDIA GPU using CUDA for acceleration. The CAMeL-Lab/bert-base-arabic-camelbert-msa model is used as the base BERT model, specifically designed for Modern Standard Arabic (MSA).

## Results
Detailed results and performance metrics are tracked using Weights & Biases (wandb). Each model logs:
- Training/validation loss and accuracy per epoch
- Overall test accuracy
- Per-class precision, recall, F1-score, and support
- Training time and inference time
- Classification reports as wandb Tables

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
