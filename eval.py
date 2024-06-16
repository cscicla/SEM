import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer
import cv2 as cv
from skimage.io import imsave
from model import UNet_PV
from train_test import get_mask

# Define your UNet_PV model here or import it if defined in another module
# from model import UNet_PV

# Check if cuda is available
use_cuda = torch.cuda.is_available()
# Set proper device based on cuda availability
device = torch.device("cuda" if use_cuda else "cpu")

model = UNet_PV().to(device)
model.load_state_dict(torch.load('/path/to/final_model_filled2.pth', map_location=device))
model.eval()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = ["background", "silver", "glass", "silicon", "void", "interfacial void"]
f1_dict = {class_name: [] for class_name in classes}
precisions = {class_name: [] for class_name in classes}
recalls = {class_name: [] for class_name in classes}
f1_scores = {class_name: [] for class_name in classes}

img_dir = Path('/home/crcvreu.student10/SEM/train/data')
mask_dir = Path('/home/crcvreu.student10/SEM/train/output')
output_dir = Path('/path/to/output')

for filename in img_dir.iterdir():
    if filename.suffix in {'.png', '.tif', '.jpg'}:
        print(filename.name)
        image = Image.open(filename)
        image = image.resize((1024, 768))
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        OG_mask_path = mask_dir / f"{filename.stem}.png"
        print(OG_mask_path)
        OG_Mask = Image.open(OG_mask_path)
        OG_Mask = OG_Mask.resize((1024, 768))
        OG_Mask = np.array(OG_Mask).astype(np.uint8)
        
        plt.imshow(OG_Mask)
        plt.axis('off')
        plt.show()

        image = image.convert('L')  # Convert to grayscale
        image = trans(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        print("IMAGE DIMS: ", image.shape)

        with torch.no_grad():
            pred = model(image)
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze()

        pred_mask = get_mask(pred)
        pred_mask = cv.cvtColor(pred_mask, cv.COLOR_BGR2RGB)
        
        plt.imshow(pred_mask)
        plt.axis('off')
        plt.show()

        imsave(output_dir / f"{filename.stem}.png", pred_mask)
        plt.clf()

        label = np.load(mask_dir / f"{filename.stem}_label.npy")
        m = MultiLabelBinarizer().fit(label)
        f1 = f1_score(m.transform(label), m.transform(pred), average=None)

        if len(f1) != 6:
            continue
        for i, class_name in enumerate(classes):
            f1_dict[class_name].append(f1[i])
        
        precision, recall, _ = precision_recall_curve(m.transform(label).ravel(), m.transform(pred).ravel())
        for i, class_name in enumerate(classes):
            precisions[class_name].append(precision[i])
            recalls[class_name].append(recall[i])
            f1_scores[class_name].append(f1[i])

# Plot precision, recall, and F1 score versus cutoff value
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
for class_name in classes:
    plt.plot(recalls[class_name], precisions[class_name], label=class_name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 3, 2)
for class_name in classes:
    plt.plot(recalls[class_name], f1_scores[class_name], label=class_name)
plt.xlabel('Recall')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(1, 3, 3)
for class_name in classes:
    plt.plot(precisions[class_name], f1_scores[class_name], label=class_name)
plt.xlabel('Precision')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

for class_name in classes:
    print(class_name, np.mean(f1_dict[class_name]))