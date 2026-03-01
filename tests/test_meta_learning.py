import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_learning import MetaLearner

class TestMetaLearning:
    
    def test_meta_learner_initialization(self):
        """Test meta-learner can be initialized"""
        ml = MetaLearner()
        assert ml is not None
        assert ml.learning_history == []
        assert ml.strategy_performance == {}
    
    def test_record_episode(self):
        """Test recording learning episodes"""
        ml = MetaLearner()
        episode = ml.record_learning_episode(
            strategy='adamw_cosine',
            initial_loss=1.0,
            final_loss=0.5,
            epochs=10,
            lr=0.001,
            architecture=[64, 128, 64]
        )
        
        assert len(ml.learning_history) == 1
        assert episode['improvement'] == 0.5
        assert episode['strategy'] == 'adamw_cosine'
    
    def test_recommend_strategy(self):
        """Test strategy recommendation"""
        ml = MetaLearner()
        
        # Record multiple episodes
        for i in range(5):
            ml.record_learning_episode(
                strategy='adamw_cosine',
                initial_loss=1.0,
                final_loss=0.5 - i*0.05,
                epochs=10,
                lr=0.001,
                architecture=[64, 128, 64]
            )
        
        strategy = ml.recommend_strategy()
        assert strategy is not None
        assert isinstance(strategy, str)
    
    def test_recommend_hyperparameters(self):
        """Test hyperparameter recommendation"""
        ml = MetaLearner()
        
        # Record multiple episodes
        for i in range(5):
            ml.record_learning_episode(
                strategy='adamw_cosine',
                initial_loss=1.0,
                final_loss=0.5,
                epochs=10,
                lr=0.001 * (i+1),
                architecture=[64, 128, 64]
            )
        
        params = ml.recommend_hyperparameters()
        assert 'lr' in params
        assert params['lr'] > 0
