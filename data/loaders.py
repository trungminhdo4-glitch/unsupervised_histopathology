# data/loaders.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from medmnist import BreastMNIST
from PIL import Image
from config import config

class BreastMNISTLoader:
    """Lädt BreastMNIST mit domänenspezifischer Augmentation"""
    
    def __init__(self, augment=True):
        self.dataset = BreastMNIST(
            split='train', 
            download=True, 
            transform=None,
            size=28
        )
        self.augment = augment
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # MedMNIST liefert NumPy-Arrays – konvertiere zu PIL Image
        img = self.dataset.imgs[idx]  # Shape: (28, 28, 1)
        img = img.astype('uint8').squeeze()  # Entferne Kanal-Dimension
        img = Image.fromarray(img, mode='L')  # Graustufen-PIL-Bild
        
        # Lade Transformations-Pipeline
        from data.stain_augmentation import StainAugmentation
        transform = StainAugmentation.get_transforms(augment=self.augment)
        
        return transform(img)  # PIL → Tensor
    
    def get_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
    
    def get_labels(self):
        """Gibt die wahren Labels für Evaluation zurück"""
        return self.dataset.labels.flatten()