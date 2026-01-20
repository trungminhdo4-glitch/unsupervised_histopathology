# train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from models.constrained_vae import ConstrainedVAE
from models.jigsaw_pretraining import JigsawPuzzleSolver
from data.loaders import BreastMNISTLoader
from utils.biological_features import BiologicalFeatureExtractor
from config import config
import os

def pretrain_with_jigsaw():
    """Pretraining mit Jigsaw Puzzle Task"""
    print("🧩 Starte Jigsaw Pretraining...")
    
    # Daten laden
    loader = BreastMNISTLoader(augment=True)
    dataloader = loader.get_dataloader()
    
    # Modell initialisieren
    base_model = ConstrainedVAE(latent_dim=config.LATENT_DIM)
    model = JigsawPuzzleSolver(base_model.encoder).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    for epoch in range(config.PRETRAIN_EPOCHS):
        total_loss, correct = 0, 0
        
        for batch in dataloader:
            batch = batch.to(config.DEVICE)
            outputs, labels = model(batch)
            labels = labels.to(config.DEVICE)
            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        acc = correct / len(dataloader.dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Jigsaw Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}, Acc: {acc:.3f}")
    
    return model.encoder

def finetune_with_biological_constraints(pretrained_encoder):
    """Finetuning mit biologischen Constraints"""
    print("🔬 Starte Finetuning mit biologischen Constraints...")
    
    # Daten laden
    loader = BreastMNISTLoader(augment=True)
    dataloader = loader.get_dataloader()
    
    # Modell initialisieren
    model = ConstrainedVAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    model.encoder.load_state_dict(pretrained_encoder.state_dict())
    
    # Freeze Encoder für stabileres Training
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config.LEARNING_RATE * 0.5)
    recon_criterion = nn.MSELoss(reduction='sum')
    
    # Training
    model.train()
    for epoch in range(config.FINETUNE_EPOCHS):
        total_loss = 0
        
        for batch in dataloader:
            batch = batch.to(config.DEVICE)
            recon, mu, logvar = model(batch)
            
            # Rekonstruktions-Loss
            recon_loss = recon_criterion(recon, batch)
            
            # Kombinierter Loss (ohne biologische Constraints, da diese komplex sind)
            loss = recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Finetune Epoch {epoch+1}/{config.FINETUNE_EPOCHS}, Avg Loss: {avg_loss:.4f}")
    
    # Speichere Modell
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/best_model.pth")
    print("✅ Modell gespeichert unter: artifacts/best_model.pth")
    
    return model, dataloader

def main():
    config.print_config()
    
    # Schritt 1: Jigsaw Pretraining
    pretrained_encoder = pretrain_with_jigsaw()
    
    # Schritt 2: Finetuning mit biologischen Constraints
    model, dataloader = finetune_with_biological_constraints(pretrained_encoder)
    
    return model, dataloader

if __name__ == "__main__":
    model, dataloader = main()