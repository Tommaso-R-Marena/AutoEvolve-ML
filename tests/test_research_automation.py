import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation import ResearchAutomation

class TestResearchAutomation:
    
    def test_research_initialization(self):
        """Test research automation can be initialized"""
        ra = ResearchAutomation()
        assert ra is not None
        assert ra.experiments == []
        assert ra.breakthroughs == []
    
    def test_propose_hypothesis(self):
        """Test hypothesis generation"""
        ra = ResearchAutomation()
        
        # Create historical data
        historical = [{'improvement': 0.001} for _ in range(10)]
        current = {'train_val_gap': 0.05}
        
        hypotheses = ra.propose_hypothesis(current, historical)
        
        assert isinstance(hypotheses, list)
        # Should generate at least one hypothesis for low improvement
    
    def test_design_experiment(self):
        """Test experiment design"""
        ra = ResearchAutomation()
        
        hypothesis = {
            'id': 'hyp_0',
            'type': 'learning_rate',
            'hypothesis': 'Test hypothesis',
            'proposed_action': 'Increase LR',
            'expected_impact': 'high',
            'confidence': 0.7
        }
        
        exp = ra.design_experiment(hypothesis)
        
        assert exp is not None
        assert exp['hypothesis_id'] == 'hyp_0'
        assert exp['status'] == 'planned'
    
    def test_record_experiment_result(self):
        """Test recording experiment results"""
        ra = ResearchAutomation()
        
        hypothesis = {
            'id': 'hyp_0',
            'type': 'test',
            'hypothesis': 'Test',
            'proposed_action': 'Test action',
            'expected_impact': 'medium',
            'confidence': 0.5
        }
        
        exp = ra.design_experiment(hypothesis)
        result = ra.record_experiment_result(exp['id'], 1.0, 0.9)
        
        assert result is not None
        assert result['status'] == 'completed'
        # Use approximate comparison for floating point
        assert abs(result['improvement'] - 0.1) < 1e-6, f"Expected ~0.1, got {result['improvement']}"
    
    def test_breakthrough_detection(self):
        """Test breakthrough detection"""
        ra = ResearchAutomation()
        
        hypothesis = {
            'id': 'hyp_0',
            'type': 'test',
            'hypothesis': 'Test',
            'proposed_action': 'Test action',
            'expected_impact': 'high',
            'confidence': 0.8
        }
        
        exp = ra.design_experiment(hypothesis)
        ra.record_experiment_result(exp['id'], 1.0, 0.9)  # 10% improvement
        
        # Should record breakthrough (>5% improvement)
        assert len(ra.breakthroughs) == 1
        assert ra.breakthroughs[0]['improvement_pct'] >= 5.0
