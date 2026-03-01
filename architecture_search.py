import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NeuralArchitectureSearch:
    """Automated architecture search for optimal model design"""
    
    def __init__(self):
        self.search_history = []
        self.architecture_performance = {}
        self.best_architectures = []
        
    def generate_candidate_architectures(self, base_architecture, n_candidates=5):
        """Generate candidate architectures based on current best"""
        candidates = []
        
        # Strategy 1: Wider networks
        wider = [int(x * 1.5) for x in base_architecture]
        candidates.append(('wider', wider))
        
        # Strategy 2: Deeper networks
        deeper = base_architecture + [base_architecture[-1]]
        candidates.append(('deeper', deeper))
        
        # Strategy 3: Bottleneck architecture
        if len(base_architecture) >= 3:
            bottleneck = [
                base_architecture[0] * 2,
                base_architecture[0] // 2,
                base_architecture[0] * 2
            ]
            candidates.append(('bottleneck', bottleneck))
        
        # Strategy 4: Pyramid (decreasing)
        pyramid = [int(base_architecture[0] * (0.7 ** i)) for i in range(len(base_architecture))]
        candidates.append(('pyramid', pyramid))
        
        # Strategy 5: Inverted pyramid (increasing)
        inv_pyramid = [int(base_architecture[0] * (1.3 ** i)) for i in range(len(base_architecture))]
        candidates.append(('inv_pyramid', inv_pyramid))
        
        return candidates[:n_candidates]
    
    def evaluate_architecture(self, architecture, performance_metrics):
        """Evaluate and record architecture performance"""
        arch_key = str(architecture)
        
        record = {
            'architecture': architecture,
            'timestamp': datetime.now().isoformat(),
            'loss': performance_metrics.get('loss', float('inf')),
            'val_loss': performance_metrics.get('val_loss', float('inf')),
            'params': self._count_parameters(architecture),
            'efficiency': performance_metrics.get('val_loss', float('inf')) / self._count_parameters(architecture) if self._count_parameters(architecture) > 0 else float('inf')
        }
        
        self.search_history.append(record)
        
        if arch_key not in self.architecture_performance:
            self.architecture_performance[arch_key] = []
        self.architecture_performance[arch_key].append(record)
        
        return record
    
    def _count_parameters(self, architecture, input_size=10, output_size=1):
        """Estimate parameter count for architecture"""
        params = 0
        prev = input_size
        for units in architecture:
            params += prev * units + units  # weights + bias
            prev = units
        params += prev * output_size + output_size
        return params
    
    def get_best_architecture(self):
        """Return the best performing architecture"""
        if not self.search_history:
            return [64, 128, 64]
        
        # Sort by validation loss (lower is better)
        sorted_archs = sorted(self.search_history, key=lambda x: x['val_loss'])
        return sorted_archs[0]['architecture']
    
    def get_pareto_optimal_architectures(self):
        """Find architectures on the Pareto frontier (performance vs complexity)"""
        if not self.search_history:
            return []
        
        pareto = []
        for record in self.search_history:
            dominated = False
            for other in self.search_history:
                # Check if other dominates record (better loss AND fewer params)
                if (other['val_loss'] < record['val_loss'] and 
                    other['params'] <= record['params']):
                    dominated = True
                    break
            if not dominated:
                pareto.append(record)
        
        return sorted(pareto, key=lambda x: x['val_loss'])
    
    def save_search_state(self, path='nas_state.json'):
        """Save NAS state"""
        try:
            state = {
                'search_history': self.search_history[-50:],
                'architecture_performance': {k: v[-10:] for k, v in self.architecture_performance.items()},
                'last_updated': datetime.now().isoformat()
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Could not save NAS state: {e}")
    
    def load_search_state(self, path='nas_state.json'):
        """Load NAS state"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.search_history = state.get('search_history', [])
            self.architecture_performance = state.get('architecture_performance', {})
            print(f"Loaded NAS state: {len(self.search_history)} architectures evaluated")
        except Exception as e:
            print(f"Could not load NAS state: {e}")
