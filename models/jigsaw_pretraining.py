# models/jigsaw_pretraining.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class JigsawPuzzleSolver(nn.Module):
    """Löst Jigsaw Puzzles zur Lernt räumlicher Beziehungen zwischen Zellen"""
    
    def __init__(self, encoder, grid_size=2):
        super().__init__()
        self.encoder = encoder
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_permutations = 24
        
        # Klassifikator für Permutationen
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_permutations)
        )
        
        # Generiere feste Permutationen für Konsistenz
        self.permutations = self._generate_permutations()
    
    def _generate_permutations(self):
        """Generiert feste Permutationen für das Training"""
        base = list(range(self.num_patches))
        perms = []
        
        # Generiere zufällige Permutationen
        while len(perms) < self.num_permutations:
            random.shuffle(base)
            perm = base.copy()
            if perm not in perms:
                perms.append(perm)
        
        return perms
    
    def _create_jigsaw(self, image):
        """Zerlegt Bild in Patches und erstellt zufällige Permutation"""
        batch_size = image.size(0)
        h, w = image.size(2), image.size(3)
        patch_h, patch_w = h // self.grid_size, w // self.grid_size
        
        patches = []
        labels = []
        
        for i in range(batch_size):
            img_patches = []
            # Zerlege Bild in Patches
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    patch = image[i, :, row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w]
                    img_patches.append(patch)
            
            # Wähle zufällige Permutation
            perm_idx = random.randint(0, self.num_permutations - 1)
            perm = self.permutations[perm_idx]
            shuffled_patches = [img_patches[i] for i in perm]
            
            # Füge Patches wieder zu einem Bild zusammen
            rows = []
            for r in range(self.grid_size):
                row_patches = shuffled_patches[r*self.grid_size:(r+1)*self.grid_size]
                row_tensor = torch.cat(row_patches, dim=2)
                rows.append(row_tensor)
            
            jigsaw_img = torch.cat(rows, dim=1)
            patches.append(jigsaw_img)
            labels.append(perm_idx)
        
        return torch.stack(patches), torch.tensor(labels)
    
    def forward(self, x):
        # Erstelle Jigsaw-Puzzle
        jigsaw_images, labels = self._create_jigsaw(x)
        
        # Extrahiere Features
        with torch.no_grad():
            features = self.encoder(jigsaw_images)
        
        # Klassifiziere Permutation
        outputs = self.classifier(features)
        return outputs, labels