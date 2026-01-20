# main.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from train import main as train_model
from human_in_the_loop.streamlit_app import PathologistApp
from utils.latent_analysis import LatentAnalyzer
from config import config
import traceback

def run_full_pipeline():
    """Führt die gesamte Pipeline aus: Training → Analyse → Interaktive App"""
    print("🚀 Starte vollständige Pipeline für unsupervised Histopathology...")
    
    # Setze Seed für Reproduzierbarkeit
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # Schritt 1: Training
        print("\n=== 1. TRAININGSPHASE ===")
        model, dataloader = train_model()
    except Exception as e:
        print(f"❌ Fehler im Training: {str(e)}")
        print("Versuche mit minimalem Modell fortzufahren...")
        from models.constrained_vae import ConstrainedVAE
        model = ConstrainedVAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
        from data.loaders import BreastMNISTLoader
        loader = BreastMNISTLoader(augment=False)
        dataloader = loader.get_dataloader()
    
    try:
        # Schritt 2: Latente Raum Analyse
        print("\n=== 2. LATENTE RAUM ANALYSE ===")
        analyzer = LatentAnalyzer(model, config.DEVICE)
        latents, images = analyzer.analyze_dataset(dataloader)
        
        # Generiere Hypothesen für die ersten 3 Dimensionen
        print("\n🧬 Generiere biologische Hypothesen:")
        for dim in range(3):
            hypotheses = analyzer.generate_latent_hypotheses(dim)
            for hyp in hypotheses:
                print(f"\nDimension {dim+1} - {hyp['title']}:")
                print(f"   🔍 {hyp['description']}")
                print(f"   💯 Konfidenz: {hyp['confidence']:.0%}")
        
        # Speichere Analyseergebnisse
        analyzer.save_analysis_results()
        
        # Schritt 3: Interaktive App
        print("\n=== 3. INTERAKTIVE PATHOLOGEN-APP ===")
        print("Starte Streamlit-App für Pathologen-Feedback...")
        print("Öffne im Browser: http://localhost:8501")
        
        app = PathologistApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Kritischer Fehler: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("\n💡 Tipp: Überprüfe, ob alle Ordner existieren:")
        print("   - artifacts/")
        print("   - analysis_results/")
        print("   - Stelle sicher, dass alle Module korrekt importiert werden")

if __name__ == "__main__":
    run_full_pipeline()