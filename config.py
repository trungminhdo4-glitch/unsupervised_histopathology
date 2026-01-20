# config.py
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class Config:
    # Dataset
    DATASET = "breastmnist"
    IMAGE_SIZE = (28, 28)
    CHANNELS = 1
    
    # Model
    LATENT_DIM = 16
    BIOLOGICAL_CONSTRAINT_WEIGHT = 0.3
    
    # Training
    BATCH_SIZE = 32
    PRETRAIN_EPOCHS = 20
    FINETUNE_EPOCHS = 15
    LEARNING_RATE = 1e-3
    
    # Hypothesis Generation
    TOP_ANOMALIES = 5
    CLUSTERS_FOR_HYPOTHESIS = 3
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def print_config(cls):
        print("🔧 System-Konfiguration:")
        print(f"   Dataset: {cls.DATASET}")
        print(f"   Latent Dim: {cls.LATENT_DIM}")
        print(f"   Biological Constraint Weight: {cls.BIOLOGICAL_CONSTRAINT_WEIGHT}")
        print(f"   Device: {cls.DEVICE}")

config = Config()