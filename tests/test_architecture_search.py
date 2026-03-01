import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture_search import NeuralArchitectureSearch

class TestArchitectureSearch:
    
    def test_nas_initialization(self):
        """Test NAS can be initialized"""
        nas = NeuralArchitectureSearch()
        assert nas is not None
        assert nas.search_history == []
    
    def test_generate_candidates(self):
        """Test candidate architecture generation"""
        nas = NeuralArchitectureSearch()
        base = [64, 128, 64]
        candidates = nas.generate_candidate_architectures(base, n_candidates=3)
        
        assert len(candidates) == 3
        for name, arch in candidates:
            assert isinstance(name, str)
            assert isinstance(arch, list)
            assert len(arch) > 0
    
    def test_evaluate_architecture(self):
        """Test architecture evaluation"""
        nas = NeuralArchitectureSearch()
        arch = [64, 128, 64]
        metrics = {'loss': 0.5, 'val_loss': 0.6}
        
        record = nas.evaluate_architecture(arch, metrics)
        
        assert record is not None
        assert record['architecture'] == arch
        assert record['loss'] == 0.5
        assert len(nas.search_history) == 1
    
    def test_get_best_architecture(self):
        """Test getting best architecture"""
        nas = NeuralArchitectureSearch()
        
        # Add some architectures
        nas.evaluate_architecture([64, 128, 64], {'val_loss': 0.8})
        nas.evaluate_architecture([128, 256, 128], {'val_loss': 0.5})
        nas.evaluate_architecture([32, 64, 32], {'val_loss': 0.9})
        
        best = nas.get_best_architecture()
        assert best == [128, 256, 128], "Should return architecture with lowest val_loss"
