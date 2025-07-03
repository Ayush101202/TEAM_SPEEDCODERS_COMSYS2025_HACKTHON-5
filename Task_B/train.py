import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#! Dataset 
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_folder):
                continue
            #! Original images
            for img_path in glob.glob(os.path.join(cls_folder, '*.jpg')):
                self.samples.append((img_path, self.class_to_idx[cls]))
            #! Distortion images
            dist_folder = os.path.join(cls_folder, 'distortion')
            if os.path.isdir(dist_folder):
                for img_path in glob.glob(os.path.join(dist_folder, '*.jpg')):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

#! Transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#! Prepare Datasets and DataLoader
train_dir = '/Task_B_Dataset/train'
val_dir   = '/Task_B_Dataset/val'

train_dataset = FaceDataset(train_dir, transform=transform)
val_dataset   = FaceDataset(val_dir, transform=transform)

batch_size = 32
num_workers = 4

data_loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers, pin_memory=True)

num_classes = len(train_dataset.class_to_idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#! Model, Loss, Optimizer 
model = models.efficientnet_b0(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler()

#! Training
num_epochs = 10
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

history = {
    'train_loss': [],
    'val_loss': [],
    'train_top1': [],
    'val_top1': [],
    'val_macro_f1': []
}
print("Training Started...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_top1 = running_corrects / total

    #! Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_corrects += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss_epoch = val_loss / val_total
    val_top1 = val_corrects / val_total
    val_macro_f1 = f1_score(all_labels, all_preds, average='macro')

    scheduler.step(val_loss_epoch)

    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    #! Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss_epoch)
    history['train_top1'].append(train_top1)
    history['val_top1'].append(val_top1)
    history['val_macro_f1'].append(val_macro_f1)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f}, Top-1 Acc: {train_top1:.4f} | "
        f"Val Loss: {val_loss_epoch:.4f}, Top-1 Acc: {val_top1:.4f}, Macro F1: {val_macro_f1:.4f}"
    )
print("Training Ended...")
#! Plot Metrics 
plt.figure(figsize=(12,5))
#! Loss
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

#! Top-1 Accuracy
plt.subplot(1,2,2)
plt.plot(history['train_top1'], label='Train Top-1 Accuracy')
plt.plot(history['val_top1'], label='Val Top-1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Top-1 Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
