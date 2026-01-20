# utils/biological_features.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from scipy import ndimage

class BiologicalFeatureExtractor:
    """Extrahiert biologisch relevante Merkmale aus histopathologischen Bildern"""
    
    @staticmethod
    def extract_nuclear_features(image_tensor):
        """
        Extrahiert Zellkern-Merkmale aus Graustufenbildern:
        - Kerngröße (Fläche)
        - Kernform (Circularity)
        - Kerndichte (Anzahl pro Fläche)
        """
        # Konvertiere Tensor zu numpy Bild
        if image_tensor.dim() == 3:
            image_tensor = image_tensor[0]  # Entferne Kanal-Dimension
        
        img = (image_tensor.numpy() * 255).astype(np.uint8)
        
        # Adaptive Schwellenwertbestimmung für Kernsegmentierung
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Entferne Rauschen
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Finde zusammenhängende Komponenten (Kerne)
        num_labels, labels_im = cv2.connectedComponents(thresh)
        
        if num_labels <= 1:  # Keine Kerne gefunden
            return {
                "nuclear_size": 0.0,
                "nuclear_circularity": 0.0,
                "nuclear_density": 0.0
            }
        
        # Analysiere jedes Objekt
        nuclear_sizes = []
        circularities = []
        
        for label in range(1, num_labels):
            # Erstelle Maske für dieses Objekt
            obj_mask = (labels_im == label).astype(np.uint8)
            
            # Berechne Fläche und Umfang
            area = np.sum(obj_mask)
            perimeter = cv2.arcLength(cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
            
            if area > 10 and perimeter > 0:  # Filtere kleine Artefakte
                nuclear_sizes.append(area)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                circularities.append(circularity)
        
        # Berechne aggregierte Merkmale
        avg_size = np.mean(nuclear_sizes) if nuclear_sizes else 0
        avg_circularity = np.mean(circularities) if circularities else 0
        density = len(nuclear_sizes) / (img.shape[0] * img.shape[1]) * 1000  # Kerne pro 1000 Pixel
        
        return {
            "nuclear_size": float(avg_size),
            "nuclear_circularity": float(avg_circularity),
            "nuclear_density": float(density)
        }
    
    @staticmethod
    def calculate_biological_constraint_loss(latent_vector, biological_features):
        """
        Berechnet Loss-Term, der die Korrelation zwischen latenter Dimension
        und biologischen Merkmalen maximiert
        """
        # Erste latente Dimension soll mit Zellkern-Größe korrelieren
        latent_dim_0 = latent_vector[:, 0].cpu().numpy()
        nuclear_size = np.array([feat["nuclear_size"] for feat in biological_features])
        
        # Normalisiere Werte
        latent_norm = (latent_dim_0 - latent_dim_0.mean()) / (latent_dim_0.std() + 1e-8)
        size_norm = (nuclear_size - nuclear_size.mean()) / (nuclear_size.std() + 1e-8)
        
        # Maximiere absolute Korrelation
        correlation = np.abs(np.mean(latent_norm * size_norm))
        return torch.tensor(1.0 - correlation, device=latent_vector.device)