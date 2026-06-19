import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DATASET_DIR = "dataset/Train"
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── transforms ────────────────────────────────────────────────────────────
# Training set gets augmentation (helps the model generalize to webcam
# conditions: different lighting, angles, slight blur, hand near face, etc.)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/test set: no augmentation, just resize + normalize
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── load dataset twice (once per transform), then split with same seed ─────
full_dataset = datasets.ImageFolder(DATASET_DIR)
print("Detected classes:", full_dataset.class_to_idx)

train_size = int(0.85 * len(full_dataset))
test_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_indices, test_indices = random_split(
    range(len(full_dataset)), [train_size, test_size], generator=generator
)

train_ds_raw = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
test_ds_raw = datasets.ImageFolder(DATASET_DIR, transform=eval_transform)

train_ds = torch.utils.data.Subset(train_ds_raw, train_indices.indices)
test_ds = torch.utils.data.Subset(test_ds_raw, test_indices.indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

# ── model ────────────────────────────────────────────────────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze early layers, but unfreeze the LAST few feature blocks so the
# model can adapt its high-level features to mask/no-mask specifically,
# not just rely on generic ImageNet features.
for p in model.features.parameters():
    p.requires_grad = False
for p in model.features[-2:].parameters():
    p.requires_grad = True

model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# ── training loop with validation tracking + best-model checkpoint ─────────
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{num_batches} | Running Loss: {running_loss/(batch_idx+1):.4f}")

    scheduler.step()

    # ── validation pass each epoch ──
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "mask_detector.pth")
        print(f"  -> New best model saved ({val_acc:.2f}%)")

print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
print("Saved mask_detector.pth")