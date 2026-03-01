import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import SelfImprovingModel

class TestModel:
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = SelfImprovingModel()
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test model forward pass works"""
        model = SelfImprovingModel()
        X = torch.randn(10, 10)
        output = model(X)
        
        assert output.shape == (10, 1), f"Expected (10, 1), got {output.shape}"
    
    def test_model_no_nans(self):
        """Test model output contains no NaN values"""
        model = SelfImprovingModel()
        X = torch.randn(10, 10)
        output = model(X)
        
        assert not torch.isnan(output).any(), "Model output contains NaN"
    
    def test_custom_architecture(self):
        """Test model with custom architecture"""
        model = SelfImprovingModel(hidden_sizes=[32, 64, 32])
        assert len(model.layers) == 3
        X = torch.randn(5, 10)
        output = model(X)
        assert output.shape == (5, 1)
    
    def test_model_trainable(self):
        """Test model parameters are trainable"""
        model = SelfImprovingModel()
        params = list(model.parameters())
        
        assert len(params) > 0, "Model has no parameters"
        assert all(p.requires_grad for p in params), "Some parameters not trainable"
