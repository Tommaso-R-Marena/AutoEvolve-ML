"""Causal Discovery and Intervention Engine

Discovers causal relationships in model performance and data patterns.
Enables targeted interventions for maximum improvement.

Based on:
- Pearl's Causal Hierarchy (association, intervention, counterfactuals)
- Granger causality for time series
- Structural Causal Models (SCMs)
- Do-calculus for intervention effects
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

class CausalGraph:
    """Directed Acyclic Graph for causal relationships"""
    
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(set)  # node -> set of children
        self.edge_weights = {}  # (parent, child) -> causal strength
        
    def add_node(self, node: str):
        self.nodes.add(node)
        
    def add_edge(self, parent: str, child: str, weight: float = 1.0):
        self.nodes.add(parent)
        self.nodes.add(child)
        self.edges[parent].add(child)
        self.edge_weights[(parent, child)] = weight
        
    def get_parents(self, node: str) -> List[str]:
        return [p for p in self.nodes if node in self.edges[p]]
    
    def get_children(self, node: str) -> List[str]:
        return list(self.edges[node])
    
    def export(self) -> Dict:
        return {
            'nodes': list(self.nodes),
            'edges': [(p, c, self.edge_weights[(p, c)]) 
                     for p in self.edges for c in self.edges[p]]
        }


class CausalDiscoveryEngine:
    """Discovers causal relationships in ML training dynamics"""
    
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.observation_buffer = []
        self.intervention_history = []
        
    def observe(self, state: Dict):
        """Record an observation"""
        self.observation_buffer.append(state.copy())
        
        # Keep buffer manageable
        if len(self.observation_buffer) > 1000:
            self.observation_buffer = self.observation_buffer[-500:]
    
    def granger_causality(self, cause_var: str, effect_var: str, 
                         lag: int = 5) -> float:
        """Test Granger causality between two variables"""
        if len(self.observation_buffer) < lag + 10:
            return 0.0
        
        try:
            cause_series = [obs.get(cause_var, 0) for obs in self.observation_buffer]
            effect_series = [obs.get(effect_var, 0) for obs in self.observation_buffer]
            
            # Simple Granger test: does past of cause help predict effect?
            predictions_with_cause = []
            predictions_without_cause = []
            
            for i in range(lag, len(effect_series) - 1):
                # With cause
                x_with = cause_series[i-lag:i] + effect_series[i-lag:i]
                y = effect_series[i]
                pred_with = np.mean(x_with)
                predictions_with_cause.append((y - pred_with) ** 2)
                
                # Without cause
                x_without = effect_series[i-lag:i]
                pred_without = np.mean(x_without)
                predictions_without_cause.append((y - pred_without) ** 2)
            
            mse_with = np.mean(predictions_with_cause)
            mse_without = np.mean(predictions_without_cause)
            
            # Causal strength = improvement in prediction
            if mse_without > 0:
                causal_strength = max(0, (mse_without - mse_with) / mse_without)
                return causal_strength
            
        except Exception as e:
            warnings.warn(f"Granger causality test failed: {e}")
        
        return 0.0
    
    def discover_causal_structure(self, variables: List[str], 
                                 threshold: float = 0.3) -> CausalGraph:
        """Discover causal relationships among variables"""
        print(f"Discovering causal structure for {len(variables)} variables...")
        
        graph = CausalGraph()
        
        # Test all pairs
        for cause in variables:
            for effect in variables:
                if cause == effect:
                    continue
                
                strength = self.granger_causality(cause, effect)
                
                if strength > threshold:
                    graph.add_edge(cause, effect, strength)
                    print(f"  {cause} → {effect}: {strength:.3f}")
        
        self.causal_graph = graph
        return graph
    
    def compute_intervention_effect(self, intervention_var: str, 
                                   target_var: str, 
                                   intervention_value: float) -> float:
        """Estimate effect of do(X=x) on Y using do-calculus"""
        
        # Find causal path from intervention to target
        if target_var not in self.causal_graph.get_children(intervention_var):
            # Check for indirect path
            children = self.causal_graph.get_children(intervention_var)
            found_path = False
            
            for child in children:
                if target_var in self.causal_graph.get_children(child):
                    # Two-hop path found
                    weight1 = self.causal_graph.edge_weights.get((intervention_var, child), 0)
                    weight2 = self.causal_graph.edge_weights.get((child, target_var), 0)
                    return weight1 * weight2 * intervention_value
            
            return 0.0
        
        # Direct causal effect
        weight = self.causal_graph.edge_weights.get((intervention_var, target_var), 0)
        return weight * intervention_value
    
    def recommend_intervention(self, target_var: str = 'val_loss', 
                              target_direction: str = 'decrease') -> Dict:
        """Recommend best intervention to affect target variable"""
        
        if not self.causal_graph.nodes:
            return {'intervention': None, 'reason': 'No causal graph discovered yet'}
        
        # Find variables that causally influence target
        parents = self.causal_graph.get_parents(target_var)
        
        if not parents:
            return {'intervention': None, 'reason': f'No causal parents of {target_var} found'}
        
        # Rank by causal strength
        interventions = []
        for parent in parents:
            strength = self.causal_graph.edge_weights.get((parent, target_var), 0)
            interventions.append({
                'variable': parent,
                'causal_strength': strength,
                'expected_effect': strength * 0.1  # Assume 10% change in parent
            })
        
        # Sort by strength
        interventions.sort(key=lambda x: x['causal_strength'], reverse=True)
        
        best = interventions[0]
        return {
            'intervention': best['variable'],
            'causal_strength': best['causal_strength'],
            'expected_effect': best['expected_effect'],
            'reason': f"Strongest causal influence on {target_var}",
            'all_options': interventions
        }
    
    def save_causal_graph(self, filepath: str = 'causal_graph.json'):
        """Save discovered causal structure"""
        data = {
            'graph': self.causal_graph.export(),
            'observations': len(self.observation_buffer),
            'interventions': len(self.intervention_history)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Causal graph saved to {filepath}")


class CounterfactualAnalyzer:
    """Analyze counterfactual scenarios (what if?)"""
    
    def __init__(self, causal_engine: CausalDiscoveryEngine):
        self.causal_engine = causal_engine
        
    def counterfactual_query(self, observed_outcome: Dict, 
                            intervention: Dict) -> Dict:
        """Answer: What if we had done X instead?"""
        
        # Get current causal graph
        graph = self.causal_engine.causal_graph
        
        # Compute alternative outcome under intervention
        alt_outcome = observed_outcome.copy()
        
        for var, value in intervention.items():
            # Update variable
            alt_outcome[var] = value
            
            # Propagate causal effects
            for child in graph.get_children(var):
                weight = graph.edge_weights.get((var, child), 0)
                if child in alt_outcome:
                    # Add causal effect
                    alt_outcome[child] += weight * (value - observed_outcome.get(var, 0))
        
        return {
            'observed': observed_outcome,
            'intervention': intervention,
            'counterfactual': alt_outcome,
            'difference': {k: alt_outcome.get(k, 0) - observed_outcome.get(k, 0) 
                          for k in set(alt_outcome.keys()) | set(observed_outcome.keys())}
        }


if __name__ == "__main__":
    # Demo causal discovery
    print("Causal Discovery Demo")
    print("=" * 50)
    
    # Simulate training observations
    engine = CausalDiscoveryEngine()
    
    # Generate synthetic observations with causal structure:
    # lr → train_loss → val_loss
    # batch_size → train_loss
    np.random.seed(42)
    
    for i in range(100):
        lr = 0.001 * np.random.uniform(0.5, 2.0)
        batch_size = np.random.choice([32, 64, 128])
        
        train_loss = 1.0 - 0.5 * lr + 0.001 * batch_size + np.random.normal(0, 0.1)
        val_loss = train_loss * 1.1 + np.random.normal(0, 0.05)
        
        engine.observe({
            'lr': lr,
            'batch_size': batch_size,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    
    # Discover causal structure
    variables = ['lr', 'batch_size', 'train_loss', 'val_loss']
    graph = engine.discover_causal_structure(variables, threshold=0.2)
    
    # Get intervention recommendation
    recommendation = engine.recommend_intervention('val_loss')
    print(f"\nRecommended intervention: {recommendation['intervention']}")
    print(f"Causal strength: {recommendation['causal_strength']:.3f}")
    print(f"Expected effect: {recommendation['expected_effect']:.3f}")
    
    # Save
    engine.save_causal_graph()
    print("\nCausal discovery complete!")
