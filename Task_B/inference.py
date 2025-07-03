import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

#! Load Class Mapping
train_folder = '/Task_B_Dataset/train'
class_names = sorted(os.listdir(train_folder))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

#! Load Model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

#! Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#! Load Test Images 
test_folder = '/Task_B_Dataset/test'
image_paths = glob.glob(os.path.join(test_folder, '*.jpg'))
true_labels = []
pred_labels = []
images_for_plot = []

for img_path in image_paths:
    img_name = os.path.basename(img_path)

    #! IMPORTANT: Adjust this line future me
    #! If your classes are like '037_frontal', '023_frontal' and your filenames are '037_frontal_xxx.jpg':
    true_label = "_".join(img_name.split('_')[:2])

    #! If the classes are just '037' and filenames are '037_xxx.jpg':
    #! true_label = img_name.split('_')[0]

    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = idx_to_class[pred_idx]

    true_labels.append(true_label)
    pred_labels.append(pred_label)
    images_for_plot.append((image, pred_label, true_label))

#! Debug: Show labels 
print("True labels extracted:", true_labels)
print("Classes in training:", list(class_to_idx.keys()))

#! Encode Labels Robustly 
true_encoded = []
pred_encoded = []

for t, p in zip(true_labels, pred_labels):
    if t in class_to_idx and p in class_to_idx:
        true_encoded.append(class_to_idx[t])
        pred_encoded.append(class_to_idx[p])

if len(true_encoded) == 0:
    print("No valid labels matched; cannot compute metrics.")
    acc = 0.0
    f1 = 0.0
else:
    acc = accuracy_score(true_encoded, pred_encoded)
    f1 = f1_score(true_encoded, pred_encoded, average='macro')

print(f"Top-1 Accuracy: {acc:.4f}")
print(f"Macro F1 Score: {f1:.4f}")

#! Plot Grid of Predictions 
n_images = min(len(images_for_plot), 16)
rows, cols = 4, 4
plt.figure(figsize=(12, 12))
for i in range(n_images):
    img, pred_label, true_label = images_for_plot[i]
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()
