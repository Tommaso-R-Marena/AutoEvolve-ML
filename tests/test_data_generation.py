import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import AdvancedDataGenerator

class TestDataGeneration:
    
    def test_data_generator_initialization(self):
        """Test data generator can be initialized"""
        gen = AdvancedDataGenerator(complexity_level=1)
        assert gen is not None
        assert gen.complexity == 1
    
    def test_generate_data_shape(self):
        """Test generated data has correct shape"""
        gen = AdvancedDataGenerator(complexity_level=1)
        X, y = gen.generate_data(n_samples=100)
        
        assert X.shape == (100, 10), f"Expected (100, 10), got {X.shape}"
        assert y.shape == (100, 1), f"Expected (100, 1), got {y.shape}"
    
    def test_generate_data_types(self):
        """Test generated data has correct types"""
        gen = AdvancedDataGenerator(complexity_level=1)
        X, y = gen.generate_data(n_samples=50)
        
        assert X.dtype == np.float32
        assert y.dtype == np.float32
    
    def test_generate_data_no_nans(self):
        """Test generated data contains no NaN values"""
        gen = AdvancedDataGenerator(complexity_level=1)
        X, y = gen.generate_data(n_samples=100)
        
        assert not np.isnan(X).any(), "X contains NaN values"
        assert not np.isnan(y).any(), "y contains NaN values"
    
    def test_generate_data_no_infs(self):
        """Test generated data contains no infinite values"""
        gen = AdvancedDataGenerator(complexity_level=1)
        X, y = gen.generate_data(n_samples=100)
        
        assert not np.isinf(X).any(), "X contains infinite values"
        assert not np.isinf(y).any(), "y contains infinite values"
    
    def test_complexity_levels(self):
        """Test different complexity levels work"""
        for level in [1, 2, 3, 5]:
            gen = AdvancedDataGenerator(complexity_level=level)
            X, y = gen.generate_data(n_samples=50)
            assert X.shape == (50, 10)
            assert y.shape == (50, 1)
