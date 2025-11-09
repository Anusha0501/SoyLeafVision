import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SoyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform or T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image
