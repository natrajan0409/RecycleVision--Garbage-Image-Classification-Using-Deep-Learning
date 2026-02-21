# 🧠 CNN & Deep Learning Reference Guide
**RecycleVision Project — Concepts & Code Reference**  
*Prepared for future study*

---

## 📋 Table of Contents
1. [Classification vs Regression](#1-classification-vs-regression)
2. [Why CNN for Images](#2-why-cnn-for-images)
3. [Convolution — How it Works](#3-convolution)
4. [Convolution Output Formula](#4-convolution-output-formula)
5. [Max Pooling](#5-max-pooling)
6. [ReLU, Flatten, Softmax](#6-relu-flatten-softmax)
7. [Full CNN Pipeline](#7-full-cnn-pipeline)
8. [Channels in Images](#8-channels)
9. [Data Augmentation](#9-data-augmentation)
10. [Transfer Learning](#10-transfer-learning)
11. [Forward Pass & Backward Pass](#11-forward--backward-pass)
12. [model.train() vs model.eval()](#12-train-vs-eval-mode)
13. [Criterion & Optimizer](#13-criterion--optimizer)
14. [Famous CNN Models](#14-famous-cnn-models)
15. [SENet](#15-senet)
16. [Edge Detection](#16-edge-detection)
17. [Project Code Reference](#17-project-code-reference)

---

## 1. Classification vs Regression

| | Classification ✅ | Regression |
|--|------------------|------------|
| Output | Category label ("plastic") | Continuous number (3.7) |
| Loss | CrossEntropyLoss | MSE |
| Final layer | `Dense(n, softmax)` | `Dense(1, linear)` |
| Our project | ✅ 12 garbage classes | ❌ |

**Rule:** Discrete labels → Classification | Continuous values → Regression

---

## 2. Why CNN for Images

| Method | Accuracy on Images | Why |
|--------|-------------------|-----|
| KNN / Decision Tree | ~40–55% | No spatial awareness |
| Bagging / Boosting | ~55–65% | Tabular data only |
| **CNN (EfficientNetB0)** | **~95%** | Learns spatial patterns ✅ |

**Images need CNNs** — they understand spatial patterns (edges, textures, shapes).

---

## 3. Convolution

A filter **slides across the image** detecting patterns:

```
Input (5×5)       Filter (3×3)       Feature Map (3×3)
pixel values   ×  learned weights  =  detected features
```

**Step-by-step:**
1. Place filter on image patch
2. Multiply filter × image patch element-wise
3. Sum all values → single output number
4. Slide filter by stride → repeat

```python
def convolution(input_, kernel, stride=1):
    h, w = input_.shape
    kh, kw = kernel.shape
    output_h = (h - kh) // stride + 1
    output_w = (w - kw) // stride + 1
    output = np.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            region = input_[i:i+kh, j:j+kw]
            output[i,j] = np.sum(region * kernel)  # ← multiply by kernel!
    return output
```

**What each layer learns:**
- Early layers → edges, lines
- Middle layers → textures, patterns
- Deep layers → shapes, objects

---

## 4. Convolution Output Formula

```
Output = ((W - F + 2P) / S) + 1
```

| Letter | Meaning |
|--------|---------|
| W | Input size |
| F | Filter size |
| P | Padding |
| S | Stride |

**Examples:**

| Input | Filter | Padding | Stride | Output |
|-------|--------|---------|--------|--------|
| 5×5 | 3×3 | 0 | 1 | **3×3** |
| 32×32 | 3×3 | 0 | 1 | **30×30** |
| 4×4 | 3×3 | 1 | 1 | **4×4** (same!) |
| 15×15 | 3×3 | 0 | 1 | **13×13** |

**Padding=1 → Output same size as input (Same Padding)**

---

## 5. Max Pooling

Takes the **maximum value** from each region — reduces spatial size:

```python
def maxpool2d(input_, size=2, stride=2):
    h, w = input_.shape
    output_h = (h - size) // stride + 1
    output_w = (w - size) // stride + 1
    output = np.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            region = input_[i*stride:i*stride+size, j*stride:j*stride+size]
            output[i,j] = np.max(region)  # Take MAX
    return output
```

**Formula:** `((W - size) / stride) + 1`

| Input | Pool size | Stride | Output |
|-------|-----------|--------|--------|
| 32×32 | 2×2 | 2 | **16×16** |
| 15×15 | 2×2 | 2 | **7×7** (floor) |
| 5×5 | 2×2 | 2 | **2×2** |
| 3×3 | 2×2 | 2 | **1×1** |

> MaxPool **halves** the size when size=2, stride=2 ✅

---

## 6. ReLU, Flatten, Softmax

### ReLU — "Kill negatives"
```python
def relu(x):
    return np.maximum(0, x)

# Example:
relu(np.array([-3, -1, 0, 2, 5]))
# Output: [0, 0, 0, 2, 5]
```

### Flatten — "2D → 1D"
```python
def flatten(x):
    return x.reshape(-1)

# Example:
# (3×3 feature map) → 9 values
```

### Softmax — "Scores → Probabilities (sum=1)"
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for stability
    return e_x / e_x.sum()

# Example for 12 classes:
# [0.01, 0.02, 0.92, ...] → sums to 1.0 ✅
```

**Order:** Flatten → Dense → ReLU → Dense → **Softmax** → class prediction

---

## 7. Full CNN Pipeline

Step-by-step example (32×32 RGB image):

```
Input             →  (3,   32, 32)   ← 3 channels (RGB)
Conv(32,  3, 1)   →  (32,  30, 30)   ← 32 feature maps
MaxPool(2, 2)     →  (32,  15, 15)   ← halved
Conv(64,  3, 1)   →  (64,  13, 13)
Conv(64,  3, 1)   →  (64,  11, 11)
MaxPool(2, 2)     →  (64,   5,  5)   ← halved
Conv(128, 3, 1)   →  (128,  3,  3)
MaxPool(2, 2)     →  (128,  1,  1)   ← 1×1
Flatten           →   128 values
Dense(64)+ReLU    →   64 values
Dense(12)+Softmax →   12 class probs
                  →   "plastic" ✅
```

---

## 8. Channels

`(3, 32, 32)` means:
- **3** = channels (R, G, B)
- **32** = height
- **32** = width

| Channels | Meaning |
|----------|---------|
| 1 | Grayscale |
| 3 | RGB colour ✅ (our images) |
| 32, 64, 128 | Feature maps after conv layers |

After Conv2d, channels **increase** (3 → 32 → 64 → 128) — more filters = richer features.

---

## 9. Data Augmentation

Creates **artificial variety** from existing images to prevent overfitting:

```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),         # Mirror left↔right
    transforms.RandomRotation(20),             # Rotate ±20°
    transforms.ColorJitter(brightness=0.2,     # Change brightness
                           contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

> ⚠️ **Only apply augmentation to training data. NEVER to validation/test!**

---

## 10. Transfer Learning

Use a model **pretrained on large dataset** (ImageNet) and adapt to your task:

```python
# Load pretrained EfficientNetB0 (trained on 1.2M images, 1000 classes)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Phase 1: Freeze base, train only classifier
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, 256),   # Fully Connected 1
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 12)      # Fully Connected 2 → 12 classes
)
# Train → 92.65% accuracy

# Phase 2: Unfreeze top layers, fine-tune
for param in model.features[-3:].parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(..., lr=1e-5)  # Very small lr
# Train → 95.42% accuracy 🏆
```

**Why Transfer Learning?**
- Pretrained model already knows edges, textures, shapes
- Only need to teach it garbage-specific patterns
- Much faster training, much less data needed

---

## 11. Forward & Backward Pass

```python
for images, labels in train_loader:
    optimizer.zero_grad()

    outputs = model(images)              # FORWARD PASS  ← data flows forward
    loss = criterion(outputs, labels)    # Measure error

    loss.backward()                      # BACKWARD PASS ← gradients flow back
    optimizer.step()                     # Update weights
```

| Pass | Direction | Purpose |
|------|-----------|---------|
| Forward | Input → Output | Get predictions |
| Backward | Output → Input | Calculate gradients |
| Update | — | Fix weights using gradients |

---

## 12. Train vs Eval Mode

```python
model.train()   # During training
# - Dropout ON (randomly drops neurons → prevents overfitting)
# - BatchNorm uses batch statistics

model.eval()    # During validation/testing
# - Dropout OFF (all neurons active → stable predictions)
# - BatchNorm uses running average

# Always use torch.no_grad() with eval():
with torch.no_grad():
    outputs = model(val_images)
```

---

## 13. Criterion & Optimizer

```python
criterion = nn.CrossEntropyLoss()   # Measures how wrong prediction is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Fixes weights

# Used in training loop:
loss = criterion(outputs, labels)   # ← criterion: computes error
loss.backward()
optimizer.step()                    # ← optimizer: corrects weights
```

| | Role |
|--|------|
| **Criterion** | "How wrong are we?" → loss value |
| **Optimizer** | "How to fix it?" → update weights |

---

## 14. Famous CNN Models

| Year | Model | Key Innovation | Accuracy |
|------|-------|---------------|----------|
| 2012 | **AlexNet** | First deep CNN, ReLU, Dropout | 84.7% |
| 2014 | **VGGNet** | Deep (16-19 layers), only 3×3 | 92.3% |
| 2014 | **GoogLeNet** | Inception modules | 93.3% |
| 2015 | **ResNet** | Skip connections (residual) | 96.4% |
| 2017 | **MobileNet** | Lightweight, mobile-friendly | 90% |
| 2019 | **EfficientNet** | Compound scaling | 93.3% |
| 2020 | **ViT** | Transformers, no convolution | 94%+ |

**Why EfficientNetB0 for our project:**
- Best accuracy/size ratio (5.3M params vs ResNet's 25M)
- Compound scaling: width + depth + resolution balanced
- Already uses SE blocks (channel attention)
- Pretrained on ImageNet → fast convergence
- Our result: **95.42%** ✅

---

## 15. SENet

**Squeeze-and-Excitation Network** — channel attention mechanism:

```
Feature Map (C × H × W)
    ↓
Squeeze: GlobalAvgPool → (C × 1 × 1)  "summarize each channel"
    ↓
Excitation: FC → ReLU → FC → Sigmoid  "score each channel 0-1"
    ↓
Scale: Multiply scores × feature map   "boost important, suppress noise"
```

> EfficientNetB0 includes SE blocks internally — we got it for free! ✅

---

## 16. Edge Detection

```python
# Sobel filters (fixed — not learned)
sobel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])  # Horizontal edges

sobel_y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])  # Vertical edges

# Combined edge strength
edges = np.sqrt(edges_x**2 + edges_y**2)
```

> CNN's first conv layer learns filters similar to Sobel — but optimized automatically!

---

## 17. Project Code Reference

### Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision streamlit pillow scikit-learn seaborn
```

### Data Loading & Augmentation
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
```

### Model
```python
import torch.nn as nn
from torchvision import models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, 256),   # FC1
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 12)      # FC2 → 12 classes
)
```

### Training Loop
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)            # Forward pass
    loss = criterion(outputs, labels)
    loss.backward()                    # Backward pass
    optimizer.step()
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
```

### Save & Load
```python
torch.save(model.state_dict(), "best_model.pth")
model.load_state_dict(torch.load("best_model.pth"))
```

### Run App
```bash
streamlit run app.py
```

---

## 📊 Final Project Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | **95.42%** 🏆 |
| Test Accuracy | **94%** |
| Target | >85% ✅ |
| Classes | 12 garbage types |
| Dataset | ~15,150 images |
| Model | EfficientNetB0 |

---

*GitHub: [RecycleVision](https://github.com/natrajan0409/RecycleVision--Garbage-Image-Classification-Using-Deep-Learning)*
