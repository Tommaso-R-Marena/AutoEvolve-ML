#!/usr/bin/env python
"""Smoke test - Quick end-to-end test of the training system"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import AdvancedDataGenerator, SelfImprovingModel, TrainingManager

def smoke_test():
    print("=" * 60)
    print("Running smoke test - minimal training cycle")
    print("=" * 60)
    
    # 1. Data generation
    print("\n1. Generating synthetic data...")
    gen = AdvancedDataGenerator(complexity_level=1)
    X, y = gen.generate_data(n_samples=200)
    print(f"   Generated: X={X.shape}, y={y.shape}")
    assert X.shape == (200, 10), "Data shape incorrect"
    
    # 2. Train/val split
    print("\n2. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 3. Create model
    print("\n3. Creating model...")
    manager = TrainingManager()
    model = manager.load_or_create_model()
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Quick training (5 epochs)
    print("\n4. Running minimal training (10 epochs)...")
    train_loss, val_loss = manager.train_epoch(
        X_train, y_train, X_val, y_val, 
        epochs=10, 
        lr=0.001
    )
    print(f"   Final - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 5. Save checkpoint
    print("\n5. Saving checkpoint...")
    manager.history['loss'].append(float(train_loss))
    manager.history['val_loss'].append(float(val_loss))
    manager.history['epochs'].append(1)
    manager.history['complexity'].append(1)
    manager.save_checkpoint(complexity=1)
    print("   Checkpoint saved")
    
    # 6. Save metrics
    print("\n6. Saving metrics...")
    import json
    
    with open('metrics.json', 'w') as f:
        json.dump(manager.history, f, indent=2)
    
    with open('improvement_metrics.json', 'w') as f:
        json.dump({
            'improvement_percentage': 0.0,
            'current_val_loss': float(val_loss),
            'best_val_loss': float(val_loss),
            'cycle': 1,
            'timestamp': '2026-03-01T00:00:00'
        }, f, indent=2)
    
    print("   Metrics saved")
    
    # 7. Verify outputs
    print("\n7. Verifying outputs...")
    assert os.path.exists('model_checkpoint.pth'), "Checkpoint not created"
    assert os.path.exists('metrics.json'), "Metrics not created"
    assert os.path.exists('improvement_metrics.json'), "Improvement metrics not created"
    print("   All outputs verified")
    
    print("\n" + "=" * 60)
    print("✅ SMOKE TEST PASSED")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = smoke_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
