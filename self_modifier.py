import json
import os
import torch
import numpy as np
from datetime import datetime
from train import SelfImprovingModel, TrainingManager

class SelfModifier:
    """Analyzes model performance and proposes code modifications"""
    
    def __init__(self):
        self.manager = TrainingManager()
        self.thresholds = {
            'min_improvement': 0.05,  # 5% minimum improvement
            'stagnation_cycles': 10,   # Cycles before considering modification
            'confidence_threshold': 0.7  # Confidence in improvement
        }
    
    def analyze_performance(self):
        """Analyze recent training history"""
        try:
            with open('metrics.json', 'r') as f:
                history = json.load(f)
        except:
            return None
        
        if len(history['loss']) < self.thresholds['stagnation_cycles']:
            return None
        
        recent_losses = history['val_loss'][-self.thresholds['stagnation_cycles']:]
        improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        
        analysis = {
            'total_cycles': len(history['loss']),
            'recent_improvement': improvement,
            'current_loss': recent_losses[-1],
            'best_loss': min(history['val_loss']),
            'stagnating': improvement < self.thresholds['min_improvement'],
            'trend': np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        }
        
        return analysis
    
    def propose_architecture_modification(self, analysis):
        """Propose specific architecture changes"""
        if not analysis['stagnating']:
            return None
        
        # Load current architecture
        checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
        current_sizes = checkpoint.get('hidden_sizes', [64, 128, 64])
        
        # Propose modification based on trend
        if analysis['trend'] > 0:  # Loss increasing
            # Add regularization
            modification = {
                'type': 'add_regularization',
                'description': 'Loss trending upward - adding L2 regularization',
                'confidence': 0.8,
                'expected_improvement': 0.15
            }
        elif len(current_sizes) < 5:  # Can add depth
            # Add layer
            new_sizes = current_sizes[:len(current_sizes)//2] + [max(current_sizes)] + current_sizes[len(current_sizes)//2:]
            modification = {
                'type': 'add_layer',
                'new_architecture': new_sizes,
                'description': f'Adding layer to increase capacity: {new_sizes}',
                'confidence': 0.75,
                'expected_improvement': 0.10
            }
        else:  # Modify training strategy
            modification = {
                'type': 'modify_training',
                'description': 'Implementing cyclic learning rate schedule',
                'confidence': 0.85,
                'expected_improvement': 0.12
            }
        
        return modification
    
    def generate_modified_code(self, modification):
        """Generate actual code modifications"""
        files = []
        
        if modification['type'] == 'add_layer':
            # Read current train.py
            with open('train.py', 'r') as f:
                code = f.read()
            
            # Modify default architecture
            old_line = "hidden_sizes=[64, 128, 64]"
            new_line = f"hidden_sizes={modification['new_architecture']}"
            modified_code = code.replace(old_line, new_line)
            
            files.append({
                'path': 'train.py',
                'content': modified_code,
                'reason': modification['description']
            })
            
            # Add test for new architecture
            test_code = f'''import pytest
import torch
from train import SelfImprovingModel

def test_modified_architecture():
    """Test that modified architecture works correctly"""
    model = SelfImprovingModel(hidden_sizes={modification['new_architecture']})
    x = torch.randn(10, 10)
    output = model(x)
    assert output.shape == (10, 1), "Output shape incorrect"
    assert not torch.isnan(output).any(), "NaN in output"
    print(f"Architecture test passed: {modification['new_architecture']}")
'''
            files.append({
                'path': 'tests/test_modified_architecture.py',
                'content': test_code,
                'reason': 'Add test for new architecture'
            })
        
        elif modification['type'] == 'add_regularization':
            with open('train.py', 'r') as f:
                code = f.read()
            
            # Add L2 regularization to optimizer
            old_opt = "optimizer = optim.Adam(self.model.parameters(), lr=lr)"
            new_opt = "optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)"
            modified_code = code.replace(old_opt, new_opt)
            
            files.append({
                'path': 'train.py',
                'content': modified_code,
                'reason': modification['description']
            })
        
        elif modification['type'] == 'modify_training':
            with open('train.py', 'r') as f:
                code = f.read()
            
            # Add cyclic LR
            old_scheduler = "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)"
            new_scheduler = "scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=20)"
            modified_code = code.replace(old_scheduler, new_scheduler)
            
            # Update scheduler call
            modified_code = modified_code.replace("scheduler.step(val_loss)", "scheduler.step()")
            
            files.append({
                'path': 'train.py',
                'content': modified_code,
                'reason': modification['description']
            })
        
        return files
    
    def create_proposal(self):
        """Main method to analyze and create modification proposal"""
        analysis = self.analyze_performance()
        
        if analysis is None:
            print("Insufficient data for self-modification")
            return False
        
        print(f"Performance analysis:")
        print(f"  Recent improvement: {analysis['recent_improvement']*100:.2f}%")
        print(f"  Stagnating: {analysis['stagnating']}")
        
        if not analysis['stagnating']:
            print("Model is improving adequately - no modification needed")
            return False
        
        modification = self.propose_architecture_modification(analysis)
        
        if modification is None:
            return False
        
        if modification['confidence'] < self.thresholds['confidence_threshold']:
            print(f"Confidence too low ({modification['confidence']}) - skipping modification")
            return False
        
        files = self.generate_modified_code(modification)
        
        proposal = {
            'title': modification['description'],
            'description': f"The model has shown {analysis['recent_improvement']*100:.2f}% improvement over the last {self.thresholds['stagnation_cycles']} cycles, which is below the {self.thresholds['min_improvement']*100}% threshold. This modification is expected to improve performance by {modification['expected_improvement']*100:.1f}%.",
            'files': files,
            'expected_improvement': modification['expected_improvement'],
            'confidence': modification['confidence'],
            'analysis': analysis
        }
        
        with open('modification_proposal.json', 'w') as f:
            json.dump(proposal, f, indent=2)
        
        print(f"\n✓ Created modification proposal: {modification['description']}")
        print(f"  Expected improvement: {modification['expected_improvement']*100:.1f}%")
        print(f"  Confidence: {modification['confidence']*100:.1f}%")
        
        return True

if __name__ == "__main__":
    modifier = SelfModifier()
    modifier.create_proposal()
