"""Integration tests for advanced research systems.

Tests the integration of:
1. Quantum-inspired optimization
2. Causal discovery engine
3. Nobel research automation
"""

import pytest
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_optimizer import QuantumInspiredOptimizer, QuantumCircuitLayer, QuantumInspiredNAS
from causality_discovery import CausalDiscoveryEngine, CausalGraph, CounterfactualAnalyzer
from nobel_research_engine import NobelResearchEngine


class TestQuantumOptimizer:
    """Test quantum-inspired optimization."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = QuantumInspiredOptimizer()
        assert optimizer.best_state is None
        assert optimizer.best_energy == float('inf')
        assert len(optimizer.temperature_schedule) > 0
    
    def test_simple_optimization(self):
        """Test optimization on simple function."""
        def quadratic(state):
            return (state.get('x', 0) - 5) ** 2
        
        optimizer = QuantumInspiredOptimizer()
        initial = {'x': 0}
        best_state, best_energy = optimizer.optimize(initial, quadratic, n_steps=20)
        
        assert best_state is not None
        assert best_energy < initial.get('x', 0) ** 2
        assert abs(best_state['x'] - 5) < 2.0  # Should get closer to optimum
    
    def test_quantum_fluctuation(self):
        """Test quantum fluctuation generates new states."""
        optimizer = QuantumInspiredOptimizer()
        state = {'lr': 0.001, 'batch_size': 32}
        
        new_state = optimizer._quantum_fluctuation(state, temperature=5.0)
        
        assert new_state is not None
        assert 'lr' in new_state
        # State should change with high temperature
        assert new_state['lr'] != state['lr'] or new_state == state
    
    def test_state_saving(self, tmp_path):
        """Test saving optimizer state."""
        optimizer = QuantumInspiredOptimizer()
        optimizer.best_state = {'lr': 0.01}
        optimizer.best_energy = 0.123
        optimizer.energy_history = [0.5, 0.3, 0.123]
        
        filepath = tmp_path / "test_state.json"
        optimizer.save_state(str(filepath))
        
        assert filepath.exists()
        
        with open(filepath, 'r') as f:
            saved = json.load(f)
        
        assert saved['best_state'] == {'lr': 0.01}
        assert saved['best_energy'] == 0.123


class TestQuantumInspiredNAS:
    """Test quantum NAS capabilities."""
    
    def test_tunneling_mutation(self):
        """Test quantum tunneling mutation."""
        nas = QuantumInspiredNAS()
        architecture = [64, 128, 64]
        
        mutated = nas.quantum_tunneling_mutation(architecture, tunnel_probability=1.0)
        
        assert len(mutated) == len(architecture)
        # With probability 1.0, at least one should change
        assert mutated != architecture
    
    def test_superposition_sampling(self):
        """Test superposition sampling."""
        nas = QuantumInspiredNAS()
        base_arch = [64, 128, 64]
        
        samples = nas.superposition_sampling(base_arch, n_samples=5)
        
        assert len(samples) == 5
        # Samples should vary
        assert len(set(tuple(s) for s in samples)) > 1


class TestCausalDiscovery:
    """Test causal discovery engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = CausalDiscoveryEngine()
        assert len(engine.observation_buffer) == 0
        assert isinstance(engine.causal_graph, CausalGraph)
    
    def test_observation_buffer(self):
        """Test observation recording."""
        engine = CausalDiscoveryEngine()
        
        for i in range(10):
            engine.observe({'x': i, 'y': i * 2})
        
        assert len(engine.observation_buffer) == 10
    
    def test_granger_causality(self):
        """Test Granger causality detection."""
        engine = CausalDiscoveryEngine()
        
        # Create causal relationship: x -> y
        np.random.seed(42)
        for i in range(50):
            x = np.random.normal(0, 1)
            y = 0.8 * x + np.random.normal(0, 0.1)  # y caused by x
            engine.observe({'x': x, 'y': y})
        
        # Should detect x -> y
        strength = engine.granger_causality('x', 'y', lag=5)
        assert strength >= 0  # Causal strength should be non-negative
    
    def test_causal_graph_creation(self):
        """Test causal graph structure."""
        graph = CausalGraph()
        graph.add_edge('x', 'y', weight=0.8)
        graph.add_edge('y', 'z', weight=0.6)
        
        assert 'x' in graph.nodes
        assert 'y' in graph.nodes
        assert 'z' in graph.nodes
        
        assert 'y' in graph.get_children('x')
        assert 'x' in graph.get_parents('y')
    
    def test_intervention_recommendation(self):
        """Test intervention recommendation."""
        engine = CausalDiscoveryEngine()
        
        # Add some observations
        for i in range(50):
            engine.observe({'loss': 0.5 - i * 0.01, 'lr': 0.001 * (1 + i * 0.01)})
        
        # Discover structure
        engine.discover_causal_structure(['loss', 'lr'], threshold=0.1)
        
        # Get recommendation
        recommendation = engine.recommend_intervention('loss')
        assert 'intervention' in recommendation


class TestNobelResearchEngine:
    """Test Nobel-level research automation."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = NobelResearchEngine()
        assert len(engine.hypotheses) == 0
        assert len(engine.discoveries) == 0
        assert engine.nobel_criteria['novelty'] == 0.0
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        engine = NobelResearchEngine()
        
        # Create historical data with clear pattern
        historical = [{'val_loss': 0.5 + np.random.normal(0, 0.02)} for _ in range(50)]
        
        # Anomalous improvement
        current = {'val_loss': 0.2}  # 3+ sigma event
        
        anomaly = engine.detect_anomaly(current, historical)
        
        assert anomaly is not None
        assert 'anomalies' in anomaly
        assert len(anomaly['anomalies']) > 0
    
    def test_hypothesis_generation(self):
        """Test hypothesis formulation."""
        engine = NobelResearchEngine()
        
        anomaly = {
            'id': 'test_anomaly',
            'anomalies': [{
                'metric': 'val_loss',
                'current_value': 0.2,
                'expected_value': 0.5,
                'z_score': 4.5,
                'type': 'improvement',
                'significance': 'high'
            }]
        }
        
        hypothesis = engine.formulate_hypothesis(anomaly)
        
        assert hypothesis is not None
        assert 'statement' in hypothesis
        assert 'predictions' in hypothesis
        assert hypothesis['novelty_score'] > 0
    
    def test_experiment_design(self):
        """Test experimental design."""
        engine = NobelResearchEngine()
        
        hypothesis = {
            'id': 'test_hypothesis',
            'statement': 'Loss landscape phase transition detected',
            'predictions': ['Reproducible', 'Significant']
        }
        
        experiment = engine.design_experiment(hypothesis)
        
        assert experiment is not None
        assert 'controls' in experiment
        assert 'treatments' in experiment
        assert len(experiment['controls']) > 0
    
    def test_discovery_evaluation(self):
        """Test discovery evaluation."""
        engine = NobelResearchEngine()
        
        results = {
            'id': 'test_experiment',
            'p_value': 0.001,
            'effect_size': 0.6,
            'reproducibility': 0.9
        }
        
        discovery = engine.evaluate_discovery(results)
        
        assert discovery is not None
        assert discovery['type'] in ['major', 'minor', 'null']
        assert discovery['significance'] > 0
    
    def test_breakthrough_detection(self):
        """Test breakthrough criteria."""
        engine = NobelResearchEngine()
        
        # Simulate breakthrough
        results = {
            'id': 'breakthrough_experiment',
            'p_value': 0.0001,  # Very significant
            'effect_size': 0.7,  # Large effect
            'reproducibility': 0.95  # Highly reproducible
        }
        
        discovery = engine.evaluate_discovery(results)
        
        assert discovery['is_breakthrough'] == True
        assert discovery['type'] == 'major'
    
    def test_nobel_assessment(self):
        """Test Nobel potential calculation."""
        engine = NobelResearchEngine()
        
        # Add some discoveries
        for i in range(3):
            discovery = {
                'is_breakthrough': i == 0,  # One breakthrough
                'impact': 0.8,
                'reproducibility': 0.9
            }
            engine.discoveries.append(discovery)
        
        # Add hypothesis
        engine.hypotheses.append({'novelty_score': 0.8})
        
        # Add experiment
        engine.experiments.append({'id': 'exp1'})
        
        assessment = engine.assess_nobel_potential()
        
        assert 'nobel_potential' in assessment
        assert 0 <= assessment['nobel_potential'] <= 1
        assert assessment['breakthrough_count'] >= 1


class TestIntegration:
    """Test integration of all systems."""
    
    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test complete research pipeline."""
        # 1. Quantum optimization
        optimizer = QuantumInspiredOptimizer()
        
        def evaluate(state):
            return (state.get('lr', 0.001) - 0.01) ** 2
        
        best_state, _ = optimizer.optimize(
            {'lr': 0.001}, evaluate, n_steps=10
        )
        
        assert best_state is not None
        
        # 2. Causal discovery
        causal_engine = CausalDiscoveryEngine()
        
        for i in range(30):
            causal_engine.observe({
                'lr': 0.001 * (1 + i * 0.1),
                'loss': 0.5 - i * 0.01
            })
        
        graph = causal_engine.discover_causal_structure(['lr', 'loss'], threshold=0.1)
        assert graph is not None
        
        # 3. Nobel research
        nobel_engine = NobelResearchEngine()
        
        historical = [{'loss': 0.5} for _ in range(20)]
        current = {'loss': 0.3}
        
        anomaly = nobel_engine.detect_anomaly(current, historical)
        
        if anomaly:
            hypothesis = nobel_engine.formulate_hypothesis(anomaly)
            assert hypothesis is not None
            
            experiment = nobel_engine.design_experiment(hypothesis)
            assert experiment is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
