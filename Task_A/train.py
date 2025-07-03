import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, Normalize, RandomBrightnessContrast, CoarseDropout, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

#! Face Detection & Cropping
try:
    from facenet_pytorch import MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True,
                  post_process=True, device=device)
    def detect_and_crop(image):
        boxes, _ = mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].astype(int)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return image[y1:y2, x1:x2]
        h, w = image.shape[:2]
        md = min(h, w); t=(h-md)//2; l=(w-md)//2
        return image[t:t+md, l:l+md]
except ImportError:
    print("Warning: facenet_pytorch not found. Skipping face detection.")
    def detect_and_crop(image):
        h, w = image.shape[:2]
        md = min(h, w); t=(h-md)//2; l=(w-md)//2
        return image[t:t+md, l:l+md]

#! Dataset
class GenderDataset(Dataset):
    def __init__(self, image_paths, labels, augmentations=None):
        self.image_paths = image_paths
        self.labels = labels
        self.augmentations = augmentations
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        import matplotlib.pyplot as plt
        image = plt.imread(self.image_paths[idx])
        image = detect_and_crop(image)
        if self.augmentations:
            image = self.augmentations(image=image)['image']
        return image, self.labels[idx]

#! Augmentations
train_transforms = Compose([
    Resize(224,224), HorizontalFlip(p=0.5), RandomBrightnessContrast(p=0.3),
    CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
])
val_transforms = Compose([
    Resize(224,224), Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
])

#! Model
def get_model(name='efficientnet_v2_s', pretrained=True, num_classes=2):
    model = getattr(models, name)(pretrained=pretrained)
    try:
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Linear(in_f,512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512,num_classes))
    except:
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_f,512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512,num_classes))
    return model

#! Focal loss and training

def focal_loss(inputs, targets, alpha=1, gamma=2):
    ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1-pt)**gamma * ce).mean()

def train_fold(fold, train_idx, val_idx, image_paths, labels):
    #! Balanced sampler
    subset_labels = [labels[i] for i in train_idx]
    counts = torch.tensor([subset_labels.count(c) for c in [0,1]], dtype=torch.float)
    weights = 1.0/counts
    sample_weights = [weights[labels[i]] for i in train_idx]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), True)

    train_ds = GenderDataset([image_paths[i] for i in train_idx], [labels[i] for i in train_idx], train_transforms)
    val_ds = GenderDataset([image_paths[i] for i in val_idx], [labels[i] for i in val_idx], val_transforms)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 5
    scaler = GradScaler()

    #! Metrics storage
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, 16):  # 15 epochs
        # !Training 
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(): outputs = model(imgs); loss = focal_loss(outputs, targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item(); total += targets.size(0)
            all_preds.extend(preds.cpu().tolist()); all_labels.extend(targets.cpu().tolist())
        train_loss = epoch_loss / total; train_acc = correct/total
        train_losses.append(train_loss); train_accs.append(train_acc)

        # !Validation 
        model.eval(); v_loss, v_correct, v_total = 0, 0, 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                # TTA flip
                flipped = torch.flip(imgs, [3]); out += model(flipped)
                loss = focal_loss(out, targets)
                v_loss += loss.item() * targets.size(0)
                preds = out.argmax(dim=1)
                v_correct += (preds==targets).sum().item(); v_total += targets.size(0)
                v_preds.extend(preds.cpu().tolist()); v_labels.extend(targets.cpu().tolist())
        val_loss = v_loss/v_total; val_acc = v_correct/v_total
        val_losses.append(val_loss); val_accs.append(val_acc)

        # !Metrics 
        print(f"Fold {fold} Epoch {epoch} | "
              f"TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.4f} | "
              f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}")
        print(classification_report(v_labels, v_preds, target_names=['male','female'], digits=4))

        if epoch > swa_start:
            swa_model.update_parameters(model)

    #! Update BN for SWA
    def bn_loader(loader):
        for imgs, _ in loader: yield imgs.to(device)
    torch.optim.swa_utils.update_bn(bn_loader(train_loader), swa_model)
    torch.save(swa_model.module.state_dict(), f"swa_fold_{fold}.pth") # model.pth (renamed)

    #! Plotting 
    epochs = list(range(1, len(train_accs)+1))
    plt.figure(); plt.plot(epochs, train_accs, label='Train Acc'); plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.show()
    plt.figure(); plt.plot(epochs, train_losses, label='Train Loss'); plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()

    return max(val_accs)


def main(image_paths, labels):
    n = len(image_paths)
    n_splits = min(5, n) if n>=2 else 1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for fold, (ti, vi) in enumerate(skf.split(image_paths,labels),1): accs.append(train_fold(fold, ti, vi, image_paths, labels))
    print('Fold Accuracies:', accs, 'Mean:', sum(accs)/len(accs))

if __name__ == '__main__':
    base = '/Task_A_Dataset'
    paths, labs = [], []
    for split in ['train','val']:
        for cls,lbl in {'male':0,'female':1}.items():
            d = os.path.join(base, split, cls)
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg','.png')): paths.append(os.path.join(d,f)); labs.append(lbl)
    main(paths, labs)
