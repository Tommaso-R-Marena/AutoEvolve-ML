import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import AdvancedDataGenerator, SelfImprovingModel, TrainingManager
from sklearn.model_selection import train_test_split

class TestIntegration:
    
    @pytest.mark.timeout(120)
    def test_full_pipeline_minimal(self):
        """Test complete training pipeline with minimal data"""
        # Generate data
        gen = AdvancedDataGenerator(complexity_level=1)
        X, y = gen.generate_data(n_samples=100)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create model
        model = SelfImprovingModel(hidden_sizes=[32, 32])
        
        # Quick training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()
        
        assert loss.item() < 100, "Loss should decrease to reasonable value"
    
    def test_data_to_model_compatibility(self):
        """Test generated data is compatible with model"""
        gen = AdvancedDataGenerator(complexity_level=1)
        X, y = gen.generate_data(n_samples=50)
        
        model = SelfImprovingModel()
        X_t = torch.FloatTensor(X)
        
        output = model(X_t)
        assert output.shape[0] == 50
    
    def test_training_manager_workflow(self):
        """Test training manager initialization and model creation"""
        manager = TrainingManager()
        model = manager.load_or_create_model('test_checkpoint.pth')
        
        assert model is not None
        assert isinstance(model, SelfImprovingModel)
