import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_system import ModelEnsemble
from train import SelfImprovingModel

class TestEnsemble:
    
    def test_ensemble_initialization(self):
        """Test ensemble can be initialized"""
        ensemble = ModelEnsemble(max_models=3)
        assert ensemble is not None
        assert ensemble.max_models == 3
        assert len(ensemble.models) == 0
    
    def test_add_model(self):
        """Test adding model to ensemble"""
        ensemble = ModelEnsemble(max_models=3)
        model = SelfImprovingModel()
        
        result = ensemble.add_model(model, performance_score=0.9)
        assert result == True
        assert len(ensemble.models) == 1
    
    def test_max_models_limit(self):
        """Test ensemble respects max_models limit"""
        ensemble = ModelEnsemble(max_models=2)
        
        # Add 3 models
        ensemble.add_model(SelfImprovingModel(), 0.5)
        ensemble.add_model(SelfImprovingModel(), 0.7)
        ensemble.add_model(SelfImprovingModel(), 0.9)  # Should replace worst
        
        assert len(ensemble.models) == 2
        assert min(ensemble.model_weights) >= 0.7  # Worst (0.5) should be removed
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        ensemble = ModelEnsemble(max_models=2)
        ensemble.add_model(SelfImprovingModel(), 0.8)
        ensemble.add_model(SelfImprovingModel(), 0.9)
        
        X = torch.randn(5, 10)
        pred = ensemble.predict(X)
        
        assert pred is not None
        assert pred.shape == (5, 1)
