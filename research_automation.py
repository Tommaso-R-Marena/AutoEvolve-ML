import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ResearchAutomation:
    """Automates research experiments and tracks breakthroughs"""
    
    def __init__(self):
        self.experiments = []
        self.breakthroughs = []
        self.hypotheses = []
        
    def propose_hypothesis(self, current_performance, historical_data):
        """Generate research hypotheses based on current state"""
        hypotheses = []
        
        # Hypothesis 1: Learning rate adaptation
        if len(historical_data) >= 5:
            recent_improvements = [historical_data[i]['improvement'] for i in range(-5, 0)]
            if np.mean(recent_improvements) < 0.01:
                hypotheses.append({
                    'id': f'hyp_{len(self.hypotheses)}',
                    'type': 'learning_rate',
                    'hypothesis': 'Current learning rate may be too low for escaping local minimum',
                    'proposed_action': 'Increase learning rate by 2x and add cyclical schedule',
                    'expected_impact': 'medium',
                    'confidence': 0.6
                })
        
        # Hypothesis 2: Architecture capacity
        if current_performance.get('train_val_gap', 0) < 0.1:
            hypotheses.append({
                'id': f'hyp_{len(self.hypotheses)}',
                'type': 'architecture',
                'hypothesis': 'Model may be underfitting - small train/val gap suggests need for more capacity',
                'proposed_action': 'Increase hidden layer sizes by 50%',
                'expected_impact': 'high',
                'confidence': 0.7
            })
        
        # Hypothesis 3: Regularization
        if current_performance.get('train_val_gap', 0) > 0.3:
            hypotheses.append({
                'id': f'hyp_{len(self.hypotheses)}',
                'type': 'regularization',
                'hypothesis': 'Large train/val gap indicates overfitting',
                'proposed_action': 'Increase dropout to 0.4 and add stronger L2 regularization',
                'expected_impact': 'high',
                'confidence': 0.8
            })
        
        # Hypothesis 4: Data complexity
        if len(historical_data) >= 10:
            recent_plateaus = sum(1 for i in range(-10, -1) if abs(historical_data[i]['improvement']) < 0.005)
            if recent_plateaus >= 7:
                hypotheses.append({
                    'id': f'hyp_{len(self.hypotheses)}',
                    'type': 'data_complexity',
                    'hypothesis': 'Model has learned current data distribution - increase complexity',
                    'proposed_action': 'Increase data complexity level and add new feature interactions',
                    'expected_impact': 'high',
                    'confidence': 0.75
                })
        
        self.hypotheses.extend(hypotheses)
        return hypotheses
    
    def design_experiment(self, hypothesis):
        """Design an experiment to test a hypothesis"""
        experiment = {
            'id': f'exp_{len(self.experiments)}',
            'hypothesis_id': hypothesis['id'],
            'hypothesis': hypothesis['hypothesis'],
            'method': hypothesis['proposed_action'],
            'start_time': datetime.now().isoformat(),
            'status': 'planned',
            'expected_impact': hypothesis['expected_impact'],
            'control_metrics': {},
            'treatment_metrics': {}
        }
        self.experiments.append(experiment)
        return experiment
    
    def record_experiment_result(self, experiment_id, control_loss, treatment_loss, additional_metrics=None):
        """Record results of an experiment"""
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                exp['control_metrics'] = {'loss': float(control_loss)}
                exp['treatment_metrics'] = {'loss': float(treatment_loss)}
                exp['improvement'] = float(control_loss - treatment_loss)
                exp['improvement_pct'] = float((control_loss - treatment_loss) / control_loss * 100) if control_loss > 0 else 0
                exp['status'] = 'completed'
                exp['end_time'] = datetime.now().isoformat()
                
                if additional_metrics:
                    exp['control_metrics'].update(additional_metrics.get('control', {}))
                    exp['treatment_metrics'].update(additional_metrics.get('treatment', {}))
                
                # Check if this is a breakthrough
                if exp['improvement_pct'] >= 5.0:
                    self._record_breakthrough(exp)
                
                return exp
        return None
    
    def _record_breakthrough(self, experiment):
        """Record a breakthrough discovery"""
        breakthrough = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': experiment['id'],
            'hypothesis': experiment['hypothesis'],
            'method': experiment['method'],
            'improvement_pct': experiment['improvement_pct'],
            'impact': 'BREAKTHROUGH - Significant improvement achieved'
        }
        self.breakthroughs.append(breakthrough)
        print(f"\n{'='*60}")
        print(f"🎉 BREAKTHROUGH DISCOVERED!")
        print(f"Improvement: {experiment['improvement_pct']:.2f}%")
        print(f"Method: {experiment['method']}")
        print(f"{'='*60}\n")
    
    def get_best_experiments(self, n=5):
        """Get top N experiments by improvement"""
        completed = [e for e in self.experiments if e['status'] == 'completed']
        sorted_exps = sorted(completed, key=lambda x: x.get('improvement_pct', 0), reverse=True)
        return sorted_exps[:n]
    
    def generate_research_report(self):
        """Generate a research summary report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiments),
            'completed_experiments': len([e for e in self.experiments if e['status'] == 'completed']),
            'breakthroughs': len(self.breakthroughs),
            'best_improvements': self.get_best_experiments(3),
            'active_hypotheses': len([h for h in self.hypotheses if h['id'] not in [e['hypothesis_id'] for e in self.experiments]]),
            'avg_improvement': np.mean([e.get('improvement_pct', 0) for e in self.experiments if e['status'] == 'completed']) if self.experiments else 0
        }
        return report
    
    def save_research_state(self, path='research_state.json'):
        """Save research automation state"""
        try:
            state = {
                'experiments': self.experiments[-100:],
                'breakthroughs': self.breakthroughs,
                'hypotheses': self.hypotheses[-50:],
                'last_updated': datetime.now().isoformat()
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Could not save research state: {e}")
    
    def load_research_state(self, path='research_state.json'):
        """Load research automation state"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.experiments = state.get('experiments', [])
            self.breakthroughs = state.get('breakthroughs', [])
            self.hypotheses = state.get('hypotheses', [])
            print(f"Loaded research state: {len(self.experiments)} experiments, {len(self.breakthroughs)} breakthroughs")
        except Exception as e:
            print(f"Could not load research state: {e}")
