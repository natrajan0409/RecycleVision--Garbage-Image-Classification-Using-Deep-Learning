# ♻️ RecycleVision — Project Walkthrough
**Garbage Image Classification Using Deep Learning**  
**Tech Stack:** Python · PyTorch · EfficientNetB0 · Streamlit  
**Final Accuracy: 95.42% Validation | 94% Test**

---

## 🗂️ Project Structure

```
GUVI_Recycle/
├── Data/
│   └── garbage_classification/   ← 12-class dataset (~15,150 images)
│       ├── battery/
│       ├── biological/
│       ├── brown-glass/
│       ├── cardboard/
│       ├── clothes/
│       ├── green-glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       ├── shoes/
│       ├── trash/
│       └── white-glass/
├── models/
│   └── best_model.pth            ← Trained EfficientNetB0 weights
├── app.py                        ← Streamlit application
├── requirements.txt              ← Dependencies
└── venv/                         ← Virtual environment
```

---

## ✅ STEP 1 — Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Register Jupyter kernel
venv\Scripts\python.exe -m pip install ipykernel
venv\Scripts\python.exe -m ipykernel install --user --name=guvi_recycle --display-name "Python (GUVI Recycle)"
```
> In Jupyter: **Kernel → Change Kernel → Python (GUVI Recycle)**

---

## ✅ STEP 2 — Dataset

- **Source:** [Kaggle — Garbage Classification (12 Classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Classes:** `battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass`
- **Total Images:** ~15,150

---

## ✅ STEP 3 — EDA

```python
import os, matplotlib.pyplot as plt
from PIL import Image

data_dir = r"D:\workspace\GUVI_Recycle\Data\garbage_classification"

class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls)))
                for cls in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, cls))}

# Bar chart
plt.figure(figsize=(12, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='steelblue')
plt.title("Images per Class"); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

# Sample images grid (3 rows × 4 cols = 12 classes)
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, cls in zip(axes.flatten(), sorted(class_counts)):
    img = Image.open(os.path.join(data_dir, cls, os.listdir(os.path.join(data_dir, cls))[0]))
    ax.imshow(img); ax.set_title(cls); ax.axis('off')
plt.tight_layout(); plt.show()
```

---

## ✅ STEP 4 — Preprocessing & Data Augmentation

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Training: with augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # Augmentation
    transforms.RandomRotation(20),           # Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/Test: NO augmentation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split: 80% Train / 10% Val / 10% Test
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
total = len(full_dataset)
train_size, val_size = int(0.8*total), int(0.1*total)
test_size = total - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=2)
```

---

## ✅ STEP 5 — Model Building & Training

### Architecture: EfficientNetB0 + Custom Head

```python
import torch, torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False   # Freeze base layers

model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 12)   # 12 classes
)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
```

### Training Function

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)            # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()                    # Backward pass
            optimizer.step()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        train_acc = 100. * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = model(images).max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
```

### Phase 1 — Frozen Base (lr=0.001)
```python
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
# 🏆 Best Val Accuracy: 92.65%
```

### Phase 2 — Fine-Tuning (lr=1e-5)
```python
for param in model.features[-3:].parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
# 🏆 Best Val Accuracy: 95.42%
```

> 💡 Trained on **Google Colab T4 GPU** (~1–2 min/epoch)

---

## ✅ STEP 6 — Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class_names = ['battery','biological','brown-glass','cardboard','clothes',
               'green-glass','metal','paper','plastic','shoes','trash','white-glass']

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        _, preds = model(images.to(device)).max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
            yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()
```

### 📊 Final Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **94%** ✅ |
| Macro Avg F1 | 92% |

| Class | F1 | Class | F1 |
|-------|----|-------|----|
| battery | 0.92 | metal | 0.88 |
| biological | 0.93 | paper | 0.90 |
| brown-glass | 0.94 | plastic | 0.85 |
| cardboard | 0.94 | shoes | 0.95 |
| **clothes** | **0.98** 🥇 | trash | 0.92 |
| green-glass | 0.94 | white-glass | 0.86 |

---

## ✅ STEP 7 — Download & Deploy Model

```python
# In Google Colab
from google.colab import files
files.download("best_model.pth")
```
Place at: `models/best_model.pth`

---

## ✅ STEP 8 — Streamlit App

```bash
venv\Scripts\activate
streamlit run app.py
```
Opens at: `http://localhost:8501`

**Features:** Upload image → Predict class → Show confidence + Top 3 + Recycling tip

---

## 🛠️ Key Decisions

| Decision | Reason |
|----------|--------|
| PyTorch over TensorFlow | Python 3.14 incompatible with TF |
| EfficientNetB0 | Best accuracy/speed ratio |
| 2-phase training | Phase 1 learns task; Phase 2 fine-tunes |
| 12-class dataset | More data → better generalization |
| Google Colab T4 GPU | 10–20x faster than local CPU |
| ImageNet normalization | Required for pretrained model weights |
