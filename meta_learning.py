import numpy as np
import torch
import torch.nn as nn
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MetaLearner:
    """Meta-learning system that learns how to learn better"""
    
    def __init__(self):
        self.learning_history = []
        self.strategy_performance = {}
        self.best_strategies = []
        
    def record_learning_episode(self, strategy, initial_loss, final_loss, epochs, lr, architecture):
        """Record a learning episode with its strategy and results"""
        episode = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'improvement': float(initial_loss - final_loss),
            'improvement_rate': float((initial_loss - final_loss) / epochs) if epochs > 0 else 0,
            'epochs': epochs,
            'learning_rate': lr,
            'architecture': architecture,
            'convergence_speed': self._calculate_convergence_speed(initial_loss, final_loss, epochs)
        }
        self.learning_history.append(episode)
        self._update_strategy_performance(strategy, episode)
        return episode
    
    def _calculate_convergence_speed(self, initial, final, epochs):
        """Calculate how quickly the model converged"""
        if initial == final or epochs == 0:
            return 0.0
        return float((initial - final) / (epochs * initial)) if initial != 0 else 0.0
    
    def _update_strategy_performance(self, strategy, episode):
        """Update performance statistics for a learning strategy"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_episodes': 0,
                'avg_improvement': 0.0,
                'avg_convergence_speed': 0.0,
                'success_rate': 0.0,
                'best_improvement': 0.0
            }
        
        stats = self.strategy_performance[strategy]
        n = stats['total_episodes']
        
        # Update running averages
        stats['avg_improvement'] = (stats['avg_improvement'] * n + episode['improvement']) / (n + 1)
        stats['avg_convergence_speed'] = (stats['avg_convergence_speed'] * n + episode['convergence_speed']) / (n + 1)
        stats['success_rate'] = (stats['success_rate'] * n + (1 if episode['improvement'] > 0 else 0)) / (n + 1)
        stats['best_improvement'] = max(stats['best_improvement'], episode['improvement'])
        stats['total_episodes'] = n + 1
    
    def recommend_strategy(self):
        """Recommend the best learning strategy based on historical performance"""
        if not self.strategy_performance:
            return 'adamw_cosine'  # Default
        
        # Score strategies
        scores = {}
        for strategy, stats in self.strategy_performance.items():
            # Weighted score: improvement, convergence speed, success rate
            score = (
                stats['avg_improvement'] * 0.4 +
                stats['avg_convergence_speed'] * 100 * 0.3 +
                stats['success_rate'] * 0.3
            )
            scores[strategy] = score
        
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    def recommend_hyperparameters(self):
        """Recommend optimal hyperparameters based on learning history"""
        if len(self.learning_history) < 5:
            return {'lr': 0.001, 'weight_decay': 0.01, 'dropout': 0.2}
        
        # Get top 20% performing episodes
        sorted_episodes = sorted(self.learning_history, key=lambda x: x['improvement'], reverse=True)
        top_episodes = sorted_episodes[:max(1, len(sorted_episodes) // 5)]
        
        # Average their hyperparameters
        avg_lr = np.mean([ep['learning_rate'] for ep in top_episodes])
        
        return {
            'lr': float(avg_lr),
            'weight_decay': 0.01,
            'dropout': 0.2
        }
    
    def suggest_architecture_modification(self, current_architecture):
        """Suggest architecture improvements based on meta-learning"""
        if len(self.learning_history) < 3:
            return None
        
        # Analyze what architectures worked best
        recent_episodes = self.learning_history[-10:]
        improvements = [ep['improvement'] for ep in recent_episodes]
        avg_improvement = np.mean(improvements)
        
        # If recent improvements are low, suggest change
        if avg_improvement < 0.01:
            return {
                'action': 'expand',
                'reason': 'Low recent improvement suggests need for more capacity',
                'suggestion': [int(x * 1.3) for x in current_architecture]
            }
        elif avg_improvement > 0.1:
            return {
                'action': 'regularize',
                'reason': 'High improvements but check for overfitting',
                'suggestion': 'increase_dropout'
            }
        
        return None
    
    def save_meta_knowledge(self, path='meta_learning_state.json'):
        """Save meta-learning knowledge"""
        try:
            state = {
                'learning_history': self.learning_history[-100:],  # Keep last 100
                'strategy_performance': self.strategy_performance,
                'last_updated': datetime.now().isoformat()
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Could not save meta-learning state: {e}")
    
    def load_meta_knowledge(self, path='meta_learning_state.json'):
        """Load meta-learning knowledge"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.learning_history = state.get('learning_history', [])
            self.strategy_performance = state.get('strategy_performance', {})
            print(f"Loaded meta-learning state: {len(self.learning_history)} episodes")
        except Exception as e:
            print(f"Could not load meta-learning state: {e}")
