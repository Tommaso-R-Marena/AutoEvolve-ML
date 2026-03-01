import sys
import os
sys.path.insert(0, '.')

from architecture_search import NeuralArchitectureSearch
from train import SelfImprovingModel
import torch
import numpy as np

nas = NeuralArchitectureSearch()
nas.load_search_state()

base_arch = [64, 128, 64]
candidates = nas.generate_candidate_architectures(base_arch, n_candidates=3)

arch_type = os.environ.get('ARCH_TYPE', 'wider')

for name, arch in candidates:
    if name == arch_type:
        print(f'Testing {name}: {arch}')
        model = SelfImprovingModel(hidden_sizes=arch)
        
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            pred = model(X)
            loss = criterion(pred, y).item()
        
        nas.evaluate_architecture(arch, {'val_loss': loss})
        print(f'Score: {loss:.4f}')

nas.save_search_state()
print(f'Best architecture so far: {nas.get_best_architecture()}')
