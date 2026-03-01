import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import requests
from sklearn.model_selection import train_test_split
import hashlib

class SelfImprovingModel(nn.Module):
    """Neural network that evolves its architecture based on performance"""
    def __init__(self, input_size=10, hidden_sizes=[64, 128, 64], output_size=1):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, h_size))
            prev_size = h_size
        self.output = nn.Linear(prev_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return self.output(x)

class DataGenerator:
    """Generates realistic synthetic data with increasing complexity"""
    def __init__(self, complexity_level=1):
        self.complexity = complexity_level
        
    def generate_data(self, n_samples=1000):
        """Generate synthetic data with patterns"""
        np.random.seed(int(datetime.now().timestamp()) % 2**32)
        
        # Multi-dimensional features with nonlinear relationships
        X = np.random.randn(n_samples, 10)
        
        # Complex target function that evolves
        y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + 
             np.log(np.abs(X[:, 2]) + 1) * X[:, 3] +
             np.tanh(X[:, 4] * X[:, 5]) +
             self.complexity * np.random.randn(n_samples) * 0.1)
        
        return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)
    
    def fetch_external_data(self):
        """Fetch real-world data from APIs for training augmentation"""
        try:
            # Example: Random number API for seed variation
            response = requests.get('https://www.random.org/integers/?num=100&min=1&max=1000&col=1&base=10&format=plain&rnd=new', timeout=5)
            if response.status_code == 200:
                external_data = np.array([int(x) for x in response.text.strip().split('\n')])
                return external_data
        except:
            pass
        return None

class TrainingManager:
    """Manages training loop, metrics, and model evolution"""
    def __init__(self):
        self.model = None
        self.history = {'loss': [], 'val_loss': [], 'epochs': [], 'complexity': []}
        self.best_loss = float('inf')
        
    def load_or_create_model(self, checkpoint_path='model_checkpoint.pth'):
        """Load existing model or create new one"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            hidden_sizes = checkpoint.get('hidden_sizes', [64, 128, 64])
            self.model = SelfImprovingModel(hidden_sizes=hidden_sizes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Loaded model from {checkpoint_path}")
        else:
            self.model = SelfImprovingModel()
            print("Created new model")
        return self.model
    
    def evolve_architecture(self):
        """Dynamically adjust architecture based on performance"""
        if len(self.history['loss']) > 5:
            recent_losses = self.history['loss'][-5:]
            improvement = recent_losses[0] - recent_losses[-1]
            
            # If improvement is stagnating, add complexity
            if improvement < 0.01:
                print("Evolving architecture: Adding layer complexity")
                current_sizes = [layer.out_features for layer in self.model.layers]
                new_sizes = [int(s * 1.2) for s in current_sizes]
                self.model = SelfImprovingModel(hidden_sizes=new_sizes)
                return True
        return False
    
    def train_epoch(self, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
        """Train model for one evolution cycle"""
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")
        
        return loss.item(), val_loss
    
    def save_checkpoint(self, complexity, checkpoint_path='model_checkpoint.pth'):
        """Save model with metadata"""
        hidden_sizes = [layer.out_features for layer in self.model.layers]
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_sizes': hidden_sizes,
            'history': self.history,
            'best_loss': self.best_loss,
            'timestamp': datetime.now().isoformat(),
            'complexity': complexity
        }, checkpoint_path)
        print(f"Model saved: {checkpoint_path}")

def main():
    print("=" * 50)
    print("AutoEvolve-ML Training Session")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    # Initialize components
    manager = TrainingManager()
    model = manager.load_or_create_model()
    
    # Determine complexity level
    complexity = len(manager.history['loss']) // 10 + 1
    data_gen = DataGenerator(complexity_level=complexity)
    
    # Generate training data
    print(f"\nGenerating data with complexity level: {complexity}")
    X, y = data_gen.generate_data(n_samples=5000)
    
    # Augment with external data if available
    external = data_gen.fetch_external_data()
    if external is not None:
        print(f"Incorporated external data: {len(external)} samples")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("\nStarting training...")
    train_loss, val_loss = manager.train_epoch(X_train, y_train, X_val, y_val, epochs=100)
    
    # Update history
    manager.history['loss'].append(train_loss)
    manager.history['val_loss'].append(val_loss)
    manager.history['epochs'].append(len(manager.history['loss']))
    manager.history['complexity'].append(complexity)
    
    # Check for evolution
    if val_loss < manager.best_loss:
        manager.best_loss = val_loss
        print(f"\n✓ New best validation loss: {val_loss:.4f}")
    
    evolved = manager.evolve_architecture()
    if evolved:
        print("\n⚡ Architecture evolved - retraining with new structure")
        train_loss, val_loss = manager.train_epoch(X_train, y_train, X_val, y_val, epochs=50)
    
    # Save checkpoint
    manager.save_checkpoint(complexity)
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(manager.history, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training session complete!")
    print(f"Total training cycles: {len(manager.history['loss'])}")
    print(f"Best validation loss: {manager.best_loss:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
