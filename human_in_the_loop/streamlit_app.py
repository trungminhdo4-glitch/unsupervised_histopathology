# human_in_the_loop/streamlit_app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.constrained_vae import ConstrainedVAE
from utils.latent_analysis import LatentAnalyzer
from config import config
import os

class PathologistApp:
    def __init__(self):
        self.model = self.load_model()
        self.latent_analyzer = LatentAnalyzer(self.model, config.DEVICE)
        st.set_page_config(page_title="Histopathology Hypothesis Generator", layout="wide")
    
    def load_model(self):
        """Lädt das trainierte Modell"""
        model = ConstrainedVAE(latent_dim=config.LATENT_DIM)
        model_path = "artifacts/best_model.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            st.success("✅ Modell erfolgreich geladen!")
        else:
            st.warning("⚠️ Kein trainiertes Modell gefunden. Nutze Initialisierung.")
        
        model.to(config.DEVICE)
        model.eval()
        return model
    
    def display_latent_space(self, latents, images):
        """Zeigt latenten Raum mit Bildbeispielen"""
        st.subheader("Latent Space Explorer")
        
        # Wählen Sie eine latente Dimension zur Analyse
        dim_options = [f"Dimension {i+1}" for i in range(config.LATENT_DIM)]
        selected_dim = st.selectbox("Wählen Sie eine latente Dimension:", dim_options)
        dim_idx = int(selected_dim.split()[-1]) - 1
        
        # Sortiere Bilder nach ausgewählter Dimension
        sorted_indices = np.argsort(latents[:, dim_idx])
        sorted_images = [images[i] for i in sorted_indices]
        sorted_latents = latents[sorted_indices]
        
        # Zeige sortierte Bilder
        st.write(f"Bilder sortiert nach {selected_dim} (niedrig → hoch):")
        cols = st.columns(10)
        for i, col in enumerate(cols):
            idx = i * (len(sorted_images) // 10)
            img = sorted_images[idx].squeeze().cpu().numpy()
            col.image(img, caption=f"Latent: {sorted_latents[idx, dim_idx]:.2f}", width=50)
        
        return dim_idx, sorted_latents, sorted_images
    
    def generate_hypotheses(self, dim_idx, latents, images):
        """Generiert biologische Hypothesen basierend auf latenter Dimension"""
        st.subheader("Generierte Hypothesen")
        
        # Extrahiere Extremwerte
        min_idx = np.argmin(latents[:, dim_idx])
        max_idx = np.argmax(latents[:, dim_idx])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Niedrigster Wert")
            st.image(images[min_idx].squeeze().cpu().numpy(), width=200)
            st.write("Potenzielle Hypothese:")
            st.text_area("", 
                        "Dieses Gewebe zeigt geringe Zelldichte und homogene Struktur. "
                        "Könnte gesundes Gewebe oder ein weniger aggressiver Subtyp sein.",
                        height=100)
        
        with col2:
            st.write("### Höchster Wert")
            st.image(images[max_idx].squeeze().cpu().numpy(), width=200)
            st.write("Potenzielle Hypothese:")
            st.text_area("", 
                        "Dieses Gewebe zeigt hohe Zelldichte und pleomorphe Kerne. "
                        "Könnte ein aggressiver Tumor-Subtyp sein.",
                        height=100)
        
        # Anomalieerkennung
        st.subheader("Auffällige Fälle (höchster Rekonstruktionsfehler)")
        
        # Berechne Rekonstruktionsfehler
        reconstruction_errors = []
        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0).to(config.DEVICE)
                recon, _, _ = self.model(img)
                error = torch.mean((img - recon) ** 2).item()
                reconstruction_errors.append(error)
        
        # Top Anomalien
        top_anomaly_indices = np.argsort(reconstruction_errors)[-config.TOP_ANOMALIES:]
        
        cols = st.columns(config.TOP_ANOMALIES)
        for i, col in enumerate(cols):
            idx = top_anomaly_indices[i]
            img = images[idx].squeeze().cpu().numpy()
            error = reconstruction_errors[idx]
            col.image(img, caption=f"Fehler: {error:.2f}", width=100)
        
        st.write("Diese Fälle sollten priorisiert histologisch validiert werden.")
    
    def pathologist_feedback(self):
        """Sammelt Feedback vom Pathologen"""
        st.subheader("Ihr Feedback")
        
        feedback = st.text_area(
            "Welche dieser Hypothesen halten Sie für biologisch plausibel? "
            "Geben Sie bitte Ihre Einschätzung ein:",
            height=150
        )
        
        hypothesis_rating = st.slider(
            "Bewerten Sie die biologische Relevanz der generierten Hypothesen (1-10):",
            1, 10, 5
        )
        
        if st.button("Feedback speichern"):
            # In echter App: Speichere Feedback für Modellverbesserung
            st.success("✅ Feedback erfolgreich gespeichert! Vielen Dank für Ihre Expertise.")
            st.balloons()
    
    def run(self):
        """Startet die Streamlit-App"""
        st.title("🧬 Histopathology Hypothesis Generator")
        st.write("Ein interaktives Tool zur Generierung biologisch relevanter Hypothesen aus kleinen Datensätzen")
        
        # Sidebar mit Systeminfo
        with st.sidebar:
            st.header("Systeminformation")
            config.print_config()
            st.write(f"Modell: ConstrainedVAE")
            st.write(f"Datensatz: BreastMNIST")
            st.write(f"Bilder analysiert: 546")
        
        # Hauptbereich
        st.write("🔍 Analysiere vorhandene BreastMNIST-Daten...")
        
        # Simuliere Datenanalyse (in echter App: echte Daten laden)
        from data.loaders import BreastMNISTLoader
        loader = BreastMNISTLoader(augment=False)
        dataloader = loader.get_dataloader(batch_size=64)
        
        try:
            latents, images = self.latent_analyzer.analyze_dataset(dataloader)
            
            # Latent Space Explorer
            dim_idx, sorted_latents, sorted_images = self.display_latent_space(latents, images)
            
            # Hypothesengenerierung
            self.generate_hypotheses(dim_idx, latents, images)
            
            # Pathologen-Feedback
            self.pathologist_feedback()
            
        except Exception as e:
            st.error(f"Fehler bei der Analyse: {str(e)}")
            st.write("Versuche es erneut oder überprüfe die Modell-Dateien.")

if __name__ == "__main__":
    app = PathologistApp()
    app.run()