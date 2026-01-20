# data/stain_augmentation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from torchvision import transforms

class StainAugmentation:
    """Simuliert unterschiedliche Färbungsvariationen in histopathologischen Bildern"""
    
    @staticmethod
    def get_transforms(augment=True):
        """Gibt domänenspezifische Transformations-Pipeline zurück"""
        if augment:
            return transforms.Compose([
                # Standard-Augmentationen für PIL-Bilder
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),  # Konvertiert PIL → Tensor
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor()
            ])