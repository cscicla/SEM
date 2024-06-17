import os
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import random
from torchvision import transforms
import numpy as np

class PVDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir # (NumPy)
        self.image_list = glob.glob(img_dir + '*.png')
        self.image_list.extend(glob.glob(img_dir + '*.jpg'))
        self.image_list.extend(glob.glob(img_dir + '*.tif'))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert("L") # grayscale
        image = image.resize((1024, 768))
        label_name = self.label_dir + os.path.basename(os.path.splitext(img_name)[0]) + '_label.npy'
        label_np = np.load(label_name)
        label = torch.from_numpy(label_np) # convert label to tensor

        if self.transform: # random cropping and flipping to help generalizability
            image = self.transform(image)
            if random.random() > 0.5:
              image = transforms.functional.hflip(image)
              label = transforms.functional.hflip(label)
            if random.random() > 0.5:
              i, j, h, w = transforms.RandomCrop.get_params(
              image, output_size=(698, 364))
              image = transforms.functional.crop(image, i, j, h, w)
              label = transforms.functional.crop(label, i, j, h, w)
        return image, label