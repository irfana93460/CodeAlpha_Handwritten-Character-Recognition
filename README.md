# CodeAlpha ML Task 3: Handwritten Character Recognition

## Objective
CNN model to classify handwritten digits 0-9 using MNIST dataset for CodeAlpha ML Internship.

## Dataset
**MNIST** - 70,000 grayscale images, 28x28 pixels
- Training: 60,000 images
- Testing: 10,000 images

## Model Architecture
3-layer Convolutional Neural Network:
1. Conv2D (32 filters) + MaxPooling
2. Conv2D (64 filters) + MaxPooling  
3. Conv2D (64 filters)
4. Dense layers with Dropout

## Results
| Metric | Score |
| --- | --- |
| Test Accuracy | 99.1%+ |
| Test Loss | <0.04 |

## Tech Stack
Python, TensorFlow, Keras, NumPy, Matplotlib

## How to Run
```bash
pip install tensorflow matplotlib
python Handwritten_recognition.py
