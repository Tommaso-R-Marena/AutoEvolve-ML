import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    """Ensemble of models for improved performance and robustness"""
    
    def __init__(self, max_models=5):
        self.models = []
        self.model_weights = []
        self.max_models = max_models
        self.performance_history = []
        
    def add_model(self, model, performance_score):
        """Add a model to the ensemble with its performance weight"""
        if len(self.models) >= self.max_models:
            # Remove worst performing model
            worst_idx = np.argmin(self.model_weights)
            if performance_score > self.model_weights[worst_idx]:
                self.models.pop(worst_idx)
                self.model_weights.pop(worst_idx)
            else:
                return False
        
        self.models.append(model)
        self.model_weights.append(performance_score)
        return True
    
    def predict(self, X):
        """Make ensemble prediction using weighted average"""
        if not self.models:
            return None
        
        predictions = []
        weights = []
        
        for model, weight in zip(self.models, self.model_weights):
            try:
                model.eval()
                with torch.no_grad():
                    pred = model(X)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                print(f"Model prediction failed: {e}")
                continue
        
        if not predictions:
            return None
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        return ensemble_pred
    
    def evaluate_ensemble(self, X_val, y_val, criterion):
        """Evaluate ensemble performance"""
        if not self.models:
            return float('inf')
        
        pred = self.predict(X_val)
        if pred is None:
            return float('inf')
        
        loss = criterion(pred, y_val).item()
        return loss
    
    def get_diversity_score(self):
        """Calculate diversity among ensemble members"""
        if len(self.models) < 2:
            return 0.0
        
        # Diversity based on architecture differences
        architectures = []
        for model in self.models:
            arch = [layer.out_features for layer in model.layers if hasattr(layer, 'out_features')]
            architectures.append(arch)
        
        # Calculate pairwise differences
        diversity = 0.0
        count = 0
        for i in range(len(architectures)):
            for j in range(i+1, len(architectures)):
                diff = sum(abs(a - b) for a, b in zip(architectures[i], architectures[j]))
                diversity += diff
                count += 1
        
        return diversity / count if count > 0 else 0.0
    
    def save_ensemble(self, path_prefix='ensemble_model'):
        """Save ensemble models"""
        try:
            for i, (model, weight) in enumerate(zip(self.models, self.model_weights)):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'weight': weight,
                    'timestamp': datetime.now().isoformat()
                }, f'{path_prefix}_{i}.pth')
            
            # Save ensemble metadata
            with open(f'{path_prefix}_meta.json', 'w') as f:
                json.dump({
                    'num_models': len(self.models),
                    'weights': self.model_weights,
                    'diversity': float(self.get_diversity_score()),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Could not save ensemble: {e}")
    
    def load_ensemble(self, path_prefix='ensemble_model', model_class=None):
        """Load ensemble models"""
        try:
            with open(f'{path_prefix}_meta.json', 'r') as f:
                meta = json.load(f)
            
            self.models = []
            self.model_weights = []
            
            for i in range(meta['num_models']):
                checkpoint = torch.load(f'{path_prefix}_{i}.pth', map_location=torch.device('cpu'))
                if model_class:
                    model = model_class()
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.models.append(model)
                    self.model_weights.append(checkpoint['weight'])
            
            print(f"Loaded ensemble: {len(self.models)} models")
        except Exception as e:
            print(f"Could not load ensemble: {e}")
