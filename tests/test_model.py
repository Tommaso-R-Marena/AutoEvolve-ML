import pytest
import torch
import numpy as np
from train import SelfImprovingModel, DataGenerator, TrainingManager

def test_model_initialization():
    """Test that model initializes correctly"""
    model = SelfImprovingModel()
    assert model is not None
    assert len(model.layers) == 3
    print("✓ Model initialization test passed")

def test_model_forward_pass():
    """Test forward pass produces correct output shape"""
    model = SelfImprovingModel()
    x = torch.randn(32, 10)  # Batch of 32, 10 features
    output = model(x)
    assert output.shape == (32, 1)
    assert not torch.isnan(output).any()
    print("✓ Forward pass test passed")

def test_data_generator():
    """Test data generation"""
    gen = DataGenerator(complexity_level=1)
    X, y = gen.generate_data(n_samples=100)
    assert X.shape == (100, 10)
    assert y.shape == (100, 1)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    print("✓ Data generation test passed")

def test_training_manager_initialization():
    """Test training manager setup"""
    manager = TrainingManager()
    assert manager is not None
    assert manager.best_loss == float('inf')
    print("✓ Training manager test passed")

def test_model_no_nan_gradients():
    """Test that training doesn't produce NaN gradients"""
    model = SelfImprovingModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), "NaN gradient detected"
    
    print("✓ Gradient sanity test passed")

def test_model_output_bounded():
    """Test that model outputs are reasonable"""
    model = SelfImprovingModel()
    model.eval()
    
    x = torch.randn(100, 10)
    with torch.no_grad():
        output = model(x)
    
    # Check outputs are finite and within reasonable bounds
    assert torch.isfinite(output).all()
    assert output.abs().max() < 1000, "Output values too large"
    print("✓ Output bounds test passed")

def test_increasing_complexity():
    """Test that higher complexity generates more challenging data"""
    gen1 = DataGenerator(complexity_level=1)
    gen2 = DataGenerator(complexity_level=5)
    
    X1, y1 = gen1.generate_data(n_samples=1000)
    X2, y2 = gen2.generate_data(n_samples=1000)
    
    # Higher complexity should have higher variance
    var1 = np.var(y1)
    var2 = np.var(y2)
    
    assert var2 > var1, "Higher complexity should increase variance"
    print("✓ Complexity scaling test passed")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
