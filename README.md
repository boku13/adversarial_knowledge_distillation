# Adversarial Knowledge Distillation (AKD) Implementation

This project implements Data-Free Adversarial Knowledge Distillation (AKD) using PyTorch on the CIFAR-100 dataset.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the script - it will automatically download the teacher model weights:
```bash
python main.py
```

## Project Structure

- `main.py`: Main implementation file containing models and training code
- `requirements.txt`: Python dependencies
- `weights/`: Directory for storing teacher model weights (created and populated automatically)
- `results/`: Directory for storing output visualizations (created automatically)

## Models

1. Teacher Model: ResNet-34 (pre-trained)
   - Parameters: ~21.3M
   - Test Accuracy: 85.12%

2. Student Models:
   - Small Student (~10% of teacher parameters)
   - Medium Student (~20% of teacher parameters)

## Training

The implementation uses Adversarial Knowledge Distillation where:
1. A generator creates synthetic images
2. The student model learns from the teacher's predictions on these synthetic images
3. No direct access to training data is used for the student models

## Evaluation

The models are evaluated on:
- 20% test data split
- 10% test data split

Metrics reported:
- Test Accuracy
- Confusion Matrix
- Parameter counts
- Generated images samples 