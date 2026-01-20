# models/prototype_learning.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

class PrototypeLearner(nn.Module):
    """Lernt Prototypen für Gewebetypen aus wenigen Beispielen"""
    
    def __init__(self, latent_dim=16, num_prototypes=2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
    
    def forward(self, z):
        """Berechnet Abstand zu Prototypen"""
        distances = torch.cdist(z, self.prototypes)
        return torch.softmax(-distances, dim=1)  # Wahrscheinlichkeiten
    
    def update_prototypes(self, embeddings, labels):
        """Aktualisiert Prototypen basierend auf Labels"""
        for i in range(self.prototypes.shape[0]):
            class_embeddings = embeddings[labels == i]
            if len(class_embeddings) > 0:
                self.prototypes.data[i] = class_embeddings.mean(dim=0)