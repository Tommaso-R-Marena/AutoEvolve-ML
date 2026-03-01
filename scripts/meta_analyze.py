import sys
import json
import os
sys.path.insert(0, '.')

try:
    from meta_learning import MetaLearner
    
    ml = MetaLearner()
    ml.load_meta_knowledge()
    
    if ml.learning_history:
        print(f'Loaded {len(ml.learning_history)} learning episodes')
        
        best_strategy = ml.recommend_strategy()
        print(f'Best strategy: {best_strategy}')
        
        params = ml.recommend_hyperparameters()
        print(f'Recommended hyperparameters: {params}')
        
        report = {
            'total_episodes': len(ml.learning_history),
            'best_strategy': best_strategy,
            'recommended_params': params,
            'strategy_performance': ml.strategy_performance
        }
        
        with open('meta_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print('Meta-learning report generated')
    else:
        print('No meta-learning history yet')
except Exception as e:
    print(f'Meta-learning analysis skipped: {e}')
