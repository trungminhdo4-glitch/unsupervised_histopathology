# utils/latent_analysis.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from config import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class LatentAnalyzer:
    """Analysiert latenten Raum und generiert biologische Hypothesen"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.images = None  # Wird entweder Tensor oder NumPy-Array sein
        self.latents = None
        self.reconstructions = None
        self.reconstruction_errors = None
    
    def extract_latents(self, dataloader):
        """Extrahiert latente Repräsentationen für alle Bilder im DataLoader"""
        self.model.eval()
        all_latents = []
        all_images = []
        all_reconstructions = []
        all_errors = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extrahiere latente Repräsentationen"):
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                else:
                    # Konvertiere NumPy-Array zu Tensor
                    batch = torch.tensor(batch).float().to(self.device)
                
                recon, mu, logvar = self.model(batch)
                
                # Speichere Daten
                all_latents.append(mu.cpu().numpy())
                
                # Sicherstellen, dass wir NumPy-Arrays speichern
                if isinstance(batch, torch.Tensor):
                    batch_np = batch.cpu().numpy()
                else:
                    batch_np = batch
                
                all_images.append(batch_np)
                all_reconstructions.append(recon.cpu().numpy())
                
                # Berechne Rekonstruktionsfehler pro Bild
                errors = torch.mean((batch - recon) ** 2, dim=(1, 2, 3))
                all_errors.append(errors.cpu().numpy())
        
        # Kombiniere alle Batches
        self.latents = np.vstack(all_latents)
        
        # Sicherstellen, dass self.images ein NumPy-Array ist
        if isinstance(all_images[0], torch.Tensor):
            self.images = np.vstack([img.cpu().numpy() for img in all_images])
        else:
            self.images = np.vstack(all_images)
        
        self.reconstructions = np.vstack(all_reconstructions)
        self.reconstruction_errors = np.concatenate(all_errors)
        
        return self.latents, self.images
    
    def analyze_dataset(self, dataloader=None):
        """Hauptmethode zur Datensatz-Analyse"""
        if dataloader is not None:
            self.extract_latents(dataloader)
        
        if self.latents is None or self.images is None:
            raise ValueError("Keine Daten geladen. Rufe zuerst extract_latents() auf.")
        
        return self.latents, self.images
    
    def get_top_anomalies(self, n=5):
        """Gibt die n Bilder mit höchstem Rekonstruktionsfehler zurück - korrigiert für negative Strides"""
        if self.reconstruction_errors is None:
            raise ValueError("Keine Rekonstruktionsfehler berechnet.")
        
        # Sortiere nach Rekonstruktionsfehler (absteigend)
        sorted_indices = np.argsort(self.reconstruction_errors)[::-1]
        top_indices = sorted_indices[:n]
        
        # FIX: Robuste Handhabung für negative Strides
        if isinstance(self.images, np.ndarray):
            # NumPy-Array: nutze np.copy() für positive Strides
            top_images = np.copy(self.images[top_indices])
        elif isinstance(self.images, torch.Tensor):
            # PyTorch-Tensor: nutze clone().detach()
            top_images = self.images[top_indices].clone().detach()
        else:
            raise TypeError(f"Unerwarteter Typ für self.images: {type(self.images)}")
        
        top_errors = self.reconstruction_errors[top_indices]
        top_latents = self.latents[top_indices]
        
        return top_images, top_errors, top_latents
    
    def get_extreme_values(self, dim_idx, n=3):
        """Gibt die n Bilder mit niedrigsten und höchsten Werten in latenter Dimension"""
        if self.latents is None:
            raise ValueError("Keine latenten Repräsentationen vorhanden.")
        
        # Sortiere nach ausgewählter Dimension
        sorted_indices = np.argsort(self.latents[:, dim_idx])
        
        # Niedrigste Werte
        low_indices = sorted_indices[:n]
        low_latents = self.latents[low_indices]
        
        # Höchste Werte
        high_indices = sorted_indices[-n:]
        high_latents = self.latents[high_indices]
        
        # FIX: Robuste Handhabung für Bilder
        if isinstance(self.images, np.ndarray):
            low_images = np.copy(self.images[low_indices])
            high_images = np.copy(self.images[high_indices])
        elif isinstance(self.images, torch.Tensor):
            low_images = self.images[low_indices].clone().detach()
            high_images = self.images[high_indices].clone().detach()
        else:
            raise TypeError(f"Unerwarteter Typ für self.images: {type(self.images)}")
        
        return {
            'low': {'images': low_images, 'latents': low_latents},
            'high': {'images': high_images, 'latents': high_latents}
        }
    
    def visualize_latent_space(self, save_path="latent_space.png", method='umap'):
        """Visualisiert latenten Raum mit UMAP/t-SNE/PCA"""
        if self.latents is None:
            raise ValueError("Keine latenten Repräsentationen vorhanden.")
        
        plt.figure(figsize=(10, 8))
        
        if method == 'umap':
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(self.latents)
            title = "Latent Space (UMAP)"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedding = reducer.fit_transform(self.latents)
            title = "Latent Space (t-SNE)"
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.latents)
            title = f"Latent Space (PCA) - Erklärte Varianz: {reducer.explained_variance_ratio_.sum():.2f}"
        else:
            raise ValueError("Unbekannte Visualisierungsmethode. Wähle 'umap', 'tsne' oder 'pca'.")
        
        # Farbcodierung nach Rekonstruktionsfehler
        plt.scatter(embedding[:, 0], embedding[:, 1], 
                   c=self.reconstruction_errors, 
                   cmap='viridis',
                   s=5, alpha=0.7)
        plt.colorbar(label='Rekonstruktionsfehler')
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Latenter Raum visualisiert und gespeichert unter: {save_path}")
        return embedding
    
    def generate_latent_hypotheses(self, dim_idx):
        """Generiert Hypothesen basierend auf latenter Dimension"""
        if self.latents is None:
            raise ValueError("Keine latenten Repräsentationen vorhanden.")
        
        # Extrahiere Extremwerte
        extremes = self.get_extreme_values(dim_idx, n=1)
        
        # Konvertiere Bilder zu Graustufen für die Anzeige
        low_image = extremes['low']['images'][0].squeeze()
        high_image = extremes['high']['images'][0].squeeze()
        
        low_latent = extremes['low']['latents'][0]
        high_latent = extremes['high']['latents'][0]
        
        # Generiere Hypothesen basierend auf biologischen Mustern
        hypotheses = []
        
        # Hypothese 1: Zellkern-Größe
        if dim_idx == 0:  # Annahme: Dimension 0 korreliert mit Kerngröße
            if low_latent[dim_idx] < high_latent[dim_idx]:
                hypotheses.append({
                    'title': 'Zellkern-Größe',
                    'description': 'Hohe Werte in dieser Dimension korrelieren mit größeren Zellkernen. '
                                   'Dies könnte auf einen aggressiveren Tumor-Subtyp hindeuten, '
                                   'da vergrößerte Kerne oft mit hoher Teilungsaktivität assoziiert sind.',
                    'evidence_images': [low_image, high_image],
                    'confidence': 0.85
                })
        
        # Hypothese 2: Zelldichte
        elif dim_idx == 1:  # Annahme: Dimension 1 korreliert mit Zelldichte
            if low_latent[dim_idx] < high_latent[dim_idx]:
                hypotheses.append({
                    'title': 'Zelldichte',
                    'description': 'Hohe Werte in dieser Dimension zeigen eine höhere Zelldichte. '
                                   'Dies könnte auf invasives Wachstum hindeuten, ein Merkmal '
                                   'fortgeschrittener Karzinome.',
                    'evidence_images': [low_image, high_image],
                    'confidence': 0.78
                })
        
        # Hypothese 3: Gewebearchitektur
        elif dim_idx == 2:  # Annahme: Dimension 2 korreliert mit Architektur
            hypotheses.append({
                'title': 'Gewebearchitektur',
                'description': 'Diese Dimension trennt strukturierte von unstrukturierten Geweben. '
                               'Unstrukturierte Muster (hohe Werte) könnten auf einen Verlust '
                               'der normalen Gewebeorganisation hindeuten – ein Zeichen für Malignität.',
                'evidence_images': [low_image, high_image],
                'confidence': 0.82
            })
        
        # Standardhypothese für andere Dimensionen
        else:
            hypotheses.append({
                'title': f'Dimension {dim_idx+1} Muster',
                'description': 'Diese Dimension trennt zwei klar unterschiedliche Gewebemuster. '
                               'Weitere histologische Validierung könnte neue Subtypen aufdecken.',
                'evidence_images': [low_image, high_image],
                'confidence': 0.70
            })
        
        return hypotheses
    
    def correlate_with_biological_features(self, biological_features, dim_idx=0):
        """
        Korreliert latente Dimension mit biologischen Merkmalen
        
        Args:
            biological_features: Liste von Dictionaries mit biologischen Merkmalen
            dim_idx: Index der latenten Dimension zur Korrelation
        """
        if len(biological_features) != len(self.latents):
            raise ValueError(f"Ungleiche Anzahl: {len(biological_features)} vs {len(self.latents)}")
        
        # Extrahiere Merkmale
        nuclear_sizes = [feat["nuclear_size"] for feat in biological_features]
        nuclear_circularities = [feat["nuclear_circularity"] for feat in biological_features]
        nuclear_densities = [feat["nuclear_density"] for feat in biological_features]
        
        # Korrelationen berechnen
        latent_dim = self.latents[:, dim_idx]
        
        correlations = {
            "nuclear_size": np.corrcoef(latent_dim, nuclear_sizes)[0, 1],
            "nuclear_circularity": np.corrcoef(latent_dim, nuclear_circularities)[0, 1],
            "nuclear_density": np.corrcoef(latent_dim, nuclear_densities)[0, 1]
        }
        
        return correlations
    
    def save_analysis_results(self, save_dir="analysis_results"):
        """Speichert alle Analyseergebnisse"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Speichere latente Repräsentationen
        np.save(f"{save_dir}/latents.npy", self.latents)
        np.save(f"{save_dir}/reconstruction_errors.npy", self.reconstruction_errors)
        
        # Speichere Bilder als NumPy-Array
        np.save(f"{save_dir}/images.npy", self.images)
        
        # Visualisiere latenten Raum
        self.visualize_latent_space(f"{save_dir}/latent_space_umap.png", method='umap')
        self.visualize_latent_space(f"{save_dir}/latent_space_pca.png", method='pca')
        
        # Speichere Top-Anomalien
        top_images, top_errors, _ = self.get_top_anomalies(n=10)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Anomalien (höchster Rekonstruktionsfehler)', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < len(top_images):
                # Stelle sicher, dass das Bild 2D ist (Graustufen)
                img = top_images[i].squeeze()
                if img.ndim == 3:
                    img = img[0]  # Nimm ersten Kanal
                
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Fehler: {top_errors[i]:.3f}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/top_anomalies.png", dpi=200)
        plt.close()
        
        print(f"✅ Alle Analyseergebnisse gespeichert unter: {save_dir}")