import torch
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report
import glob

import math
import matplotlib.pyplot as plt
import os
import numpy as np

#! Enter test directory path here (test -> 'male' folder, 'female' folder)
test_dir = "/Task_A_Dataset/test"


#! Load model
model = models.efficientnet_v2_s(pretrained=False)
in_f = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_f,512), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(512,2)
)
model.load_state_dict(torch.load("model.pth"))
model.eval()

#! Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

#! Prepare test paths and labels automatically from folder names

image_paths, labels = [], []
for cls, lbl in [('male', 0), ('female', 1)]:
    for path in glob.glob(os.path.join(test_dir, cls, '*')):
        image_paths.append(path)
        labels.append(lbl)

#! Inference
preds, trues = [], []
for path, label in zip(image_paths, labels):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)
    out = model(img)
    pred = out.argmax(1).item()
    preds.append(pred)
    trues.append(label)

print(classification_report(trues, preds, target_names=['male','female'], digits=4))

#! Plot images with Actual vs Predicted labels

print("Please wait for the plots...\n\n")

n = len(image_paths)
cols = min(4, n)
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

for idx, (path, true, pred) in enumerate(zip(image_paths, trues, preds)):
    ax = axes[idx]
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.axis('off')
    filename = os.path.basename(path)
    true_label = 'male' if true == 0 else 'female'
    pred_label = 'male' if pred == 0 else 'female'
    title = f"{filename} | true→{true_label} | pred→{pred_label}"
    ax.set_title(title, fontsize=10, pad=6)

#! Hide any extra subplots
for ax in axes[n:]:
    ax.axis('off')

plt.tight_layout()
plt.show()