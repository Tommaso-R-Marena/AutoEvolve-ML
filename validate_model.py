import json
import torch
import numpy as np
from train import SelfImprovingModel, DataGenerator, TrainingManager

def validate_model_performance():
    """Validate that model meets performance thresholds"""
    print("Validating model performance...")
    
    # Load model and history
    manager = TrainingManager()
    model = manager.load_or_create_model()
    
    try:
        with open('metrics.json', 'r') as f:
            history = json.load(f)
    except:
        history = {'val_loss': [float('inf')]}
    
    # Generate validation data
    complexity = len(history.get('loss', [])) // 10 + 1
    data_gen = DataGenerator(complexity_level=complexity)
    X, y = data_gen.generate_data(n_samples=1000)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        predictions = model(X_tensor)
        val_loss = torch.nn.MSELoss()(predictions, y_tensor).item()
    
    print(f"Current validation loss: {val_loss:.4f}")
    
    # Check for NaN/Inf
    if np.isnan(val_loss) or np.isinf(val_loss):
        print("❌ FAIL: Model producing NaN or Inf values")
        return False
    
    # Check for catastrophic performance degradation
    if history.get('val_loss'):
        best_historical_loss = min(history['val_loss'])
        if val_loss > best_historical_loss * 3.0:  # 3x worse than best
            print(f"❌ FAIL: Performance degraded significantly (current: {val_loss:.4f}, best: {best_historical_loss:.4f})")
            return False
    
    print("✅ PASS: Model performance acceptable")
    return True

if __name__ == "__main__":
    success = validate_model_performance()
    exit(0 if success else 1)
