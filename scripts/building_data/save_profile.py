#!/usr/bin/env python3
"""
Сохраняет профиль владельца и оптимальный порог в JSON/NPZ для runtime-агента.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

def main():
    # Пути
    emb_path = Path("data/embeddings/embeddings.npz")
    out_dir = Path("models/identity_service")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    profile_path = out_dir / "owner_profile.npy"
    config_path = out_dir / "config.json"

    print(f"Loading embeddings from {emb_path}...")
    data = np.load(emb_path)
    
    embs = data["embeddings"]
    stypes = data["speaker_type"]
    splits = data["split"]
    
    # Берем только OWNER из TRAIN сплита
    mask = (stypes == "owner") & (splits == "train")
    profile_embs = embs[mask]
    
    if len(profile_embs) == 0:
        raise ValueError("No owner training embeddings found!")
        
    print(f"Saving profile with {len(profile_embs)} vectors to {profile_path}")
    np.save(profile_path, profile_embs)
    
    # Сохраняем конфигурацию
    config = {
        "threshold": 0.5855,  # Тот самый найденный порог
        "model_name": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": 192,
        "num_reference_vectors": int(len(profile_embs)),
        "description": "Owner voice profile for Jarvis agent"
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    print(f"Config saved to {config_path}")
    print("\n✅ Profile exported successfully.")
    print("Now you can load this in your Agent backend without re-calculating everything.")

if __name__ == "__main__":
    main()