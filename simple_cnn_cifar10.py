"""
Simple CNN on CIFAR-10  —  Built from Scratch (No Pretrained Weights)
======================================================================
Architecture  :  Custom 3-block CNN
Dataset       :  CIFAR-10  (60 000 RGB 32x32 images, 10 classes)
Framework     :  PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import time
from collections import Counter

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR    = "./data"
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 1e-3
NUM_CLASSES = 10
CKPT_PATH   = "./simple_cnn_cifar10.pth"

CLASSES = ["plane","car","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

sep = lambda t: print(f"\n{'='*62}\n  {t}\n{'='*62}")


# ══════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must be importable at module level on Windows)
# ══════════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    """
    Custom CNN for CIFAR-10  (32x32 RGB -> 10 classes)
    Input -> Block1 -> Block2 -> Block3 -> Classifier
    Each block: Conv->BN->ReLU->Conv->BN->ReLU->MaxPool->Dropout
    Classifier: Flatten->FC(512)->BN->ReLU->Dropout->FC(10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.3),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64,  128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════
# TRAINING UTILITIES  (defined at module level — importable)
# ══════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, scaler, scheduler=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type,
                                 enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()   # OneCycleLR steps once per BATCH
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        loss  = criterion(out, labels)
        preds = out.argmax(1)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (preds == labels).sum().item()
        total    += imgs.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return (loss_sum / total,
            100.0 * correct / total,
            torch.cat(all_preds),
            torch.cat(all_labels))


# ══════════════════════════════════════════════════════════════════
# MAIN — required on Windows for multiprocessing in DataLoader
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Dataset download & exploration ────────────────────────
    sep("1. DATASET — CIFAR-10")

    raw      = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                             download=True, transform=T.ToTensor())
    test_raw = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                             download=True, transform=T.ToTensor())
    img0, lbl0 = raw[0]
    print(f"Train samples   :  {len(raw):,}")
    print(f"Test  samples   :  {len(test_raw):,}")
    print(f"Image shape     :  {tuple(img0.shape)}   (C x H x W)")
    print(f"Classes ({NUM_CLASSES})     :  {CLASSES}")
    print(f"Device          :  {device}")

    print("\nClass distribution:")
    counts = Counter(raw.targets)
    for i, cls in enumerate(CLASSES):
        bar = "=" * (counts[i] // 200)
        print(f"  {i}  {cls:<7}  {counts[i]:,}  {bar}")

    # ── 2. DataLoaders ───────────────────────────────────────────
    sep("2. DATA AUGMENTATION & DATALOADERS")

    train_tfm = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.2)),
    ])
    val_tfm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                             download=False, transform=train_tfm)
    val_ds   = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                             download=False, transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    imgs, lbls = next(iter(train_loader))
    print(f"Train batches  :  {len(train_loader)}")
    print(f"Val   batches  :  {len(val_loader)}")
    print(f"Batch shape    :  {tuple(imgs.shape)}")

    # ── 3. Model ─────────────────────────────────────────────────
    sep("3. SIMPLE CNN ARCHITECTURE")

    model       = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal parameters     :  {total_p:,}")
    print(f"Trainable parameters :  {trainable_p:,}")
    print(f"Model size (FP32)    :  {total_p * 4 / 1e6:.2f} MB")

    # ── 4. Training setup ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10,
        epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # ── 5. Training loop ─────────────────────────────────────────
    sep("4. TRAINING LOOP")

    history  = dict(tr_loss=[], tr_acc=[], va_loss=[], va_acc=[])
    best_acc = 0.0

    print(f"\n{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Acc':>8}  "
          f"{'Va Loss':>8}  {'Va Acc':>8}  {'LR':>9}  {'Time':>6}")
    print("-" * 66)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc       = train_epoch(model, train_loader,
                                            criterion, optimizer, scaler, scheduler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["va_loss"].append(va_loss)
        history["va_acc"].append(va_acc)

        flag = " *" if va_acc > best_acc else ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"epoch": epoch, "state": model.state_dict(),
                        "val_acc": va_acc}, CKPT_PATH)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>7.2f}%  "
              f"{va_loss:>8.4f}  {va_acc:>7.2f}%  "
              f"{lr_now:>9.6f}  {time.time()-t0:>5.1f}s{flag}")

    print(f"\nBest validation accuracy :  {best_acc:.2f}%")

    # ── 6. Per-class accuracy ─────────────────────────────────────
    sep("5. PER-CLASS ACCURACY")

    _, _, all_preds, all_labels = evaluate(model, val_loader, criterion)
    cls_correct = torch.zeros(NUM_CLASSES)
    cls_total   = torch.zeros(NUM_CLASSES)
    for p, l in zip(all_preds, all_labels):
        cls_correct[l] += (p == l).item()
        cls_total[l]   += 1

    print(f"\n{'Class':<8}  {'Correct':>8}  {'Total':>6}  {'Acc':>8}   Bar")
    print("-" * 55)
    for i, cls in enumerate(CLASSES):
        acc = 100.0 * cls_correct[i] / cls_total[i]
        bar = "=" * int(acc / 5)
        print(f"{cls:<8}  {int(cls_correct[i]):>8}  "
              f"{int(cls_total[i]):>6}  {acc:>7.2f}%   {bar}")

    # ── 7. Confusion matrix ───────────────────────────────────────
    sep("6. CONFUSION MATRIX  (rows=true, cols=predicted)")

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for p, l in zip(all_preds, all_labels):
        conf[l][p] += 1

    print(f"{'':>7}" + "".join(f"{c[:5]:>6}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = f"{cls[:6]:>7}"
        for j in range(NUM_CLASSES):
            v = conf[i][j].item()
            row += f"[{v:>3}]" if i == j else f" {v:>4} "
        print(row)

    # ── 8. Single-image inference ─────────────────────────────────
    sep("7. SINGLE IMAGE INFERENCE  (first 10 test images)")

    model.eval()
    print(f"\n{'#':>3}  {'True':>8}  {'Predicted':>10}  {'Confidence':>11}  Result")
    print("-" * 48)
    with torch.no_grad():
        for i in range(10):
            img_t, true_lbl = val_ds[i]
            logits = model(img_t.unsqueeze(0).to(device))
            probs  = F.softmax(logits, dim=1).squeeze()
            pred   = probs.argmax().item()
            conf_p = probs[pred].item() * 100
            symbol = "OK" if pred == true_lbl else "X"
            print(f"{i+1:>3}  {CLASSES[true_lbl]:>8}  "
                  f"{CLASSES[pred]:>10}  {conf_p:>10.2f}%   {symbol}")

    # ── 9. Training history ───────────────────────────────────────
    sep("8. TRAINING HISTORY")

    print(f"\n{'Ep':>4}  {'Train Acc':>10}  {'Val Acc':>10}  Progress")
    print("-" * 55)
    for ep, (tr, va) in enumerate(zip(history["tr_acc"], history["va_acc"]), 1):
        bar = "=" * int(va / 5)
        print(f"{ep:>4}  {tr:>9.2f}%  {va:>9.2f}%  {bar}")

    print(f"\nFinal  train acc  :  {history['tr_acc'][-1]:.2f}%")
    print(f"Final  val   acc  :  {history['va_acc'][-1]:.2f}%")
    print(f"Best   val   acc  :  {best_acc:.2f}%")
    print(f"Overfit gap       :  "
          f"{history['tr_acc'][-1] - history['va_acc'][-1]:.2f}%")

    # ── 10. Save & reload ─────────────────────────────────────────
    sep("9. SAVE & RELOAD BEST CHECKPOINT")

    ckpt   = torch.load(CKPT_PATH, map_location=device)
    model2 = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    model2.load_state_dict(ckpt["state"])
    model2.eval()

    _, reload_acc, _, _ = evaluate(model2, val_loader, criterion)
    print(f"Saved at epoch    :  {ckpt['epoch']}")
    print(f"Saved val acc     :  {ckpt['val_acc']:.2f}%")
    print(f"Reloaded val acc  :  {reload_acc:.2f}%  OK")

    # ── 11. Architecture summary ──────────────────────────────────
    sep("ARCHITECTURE SUMMARY")

    summary = [
        ("Input",        "—",                                "3x32x32",   "—"),
        ("Block 1",      "Conv(3->32) BN ReLU x2 MaxPool",  "32x16x16",  "~18K"),
        ("Block 2",      "Conv(32->64) BN ReLU x2 MaxPool", "64x8x8",    "~74K"),
        ("Block 3",      "Conv(64->128) BN ReLU x2 MaxPool","128x4x4",   "~295K"),
        ("Flatten",      "—",                                "2048",      "—"),
        ("FC(2048->512)","Linear BN ReLU Dropout(0.5)",      "512",       "~1.0M"),
        ("FC(512->10)",  "Linear",                           "10",        "~5K"),
    ]
    print(f"\n{'Stage':<16} {'Layers':<38} {'Output':<12} {'Params'}")
    print("-" * 78)
    for stage, layers, out, params in summary:
        print(f"{stage:<16} {layers:<38} {out:<12} {params}")

    print(f"\nTotal trainable parameters :  {trainable_p:,}  "
          f"(~{trainable_p/1e6:.2f} M)")
    print("\n" + "=" * 62)
    print("  Simple CNN + CIFAR-10 complete")
    print("=" * 62)
