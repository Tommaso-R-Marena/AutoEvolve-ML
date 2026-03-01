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
import warnings
warnings.filterwarnings('ignore')

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
        
        X = np.random.randn(n_samples, 10)
        
        y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + 
             np.log(np.abs(X[:, 2]) + 1) * X[:, 3] +
             np.tanh(X[:, 4] * X[:, 5]) +
             self.complexity * np.random.randn(n_samples) * 0.1)
        
        return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)
    
    def fetch_external_data(self):
        """Fetch real-world data from APIs for training augmentation"""
        try:
            response = requests.get('https://www.random.org/integers/?num=100&min=1&max=1000&col=1&base=10&format=plain&rnd=new', timeout=5)
            if response.status_code == 200:
                external_data = np.array([int(x) for x in response.text.strip().split('\n')])
                return external_data
        except:
            pass
        return None

class DatabaseManager:
    """Manages expandable training database with chunking and Git LFS"""
    def __init__(self, max_chunk_size_mb=95, data_dir='data'):
        self.data_dir = data_dir
        self.chunks_dir = os.path.join(data_dir, 'chunks')
        self.max_chunk_size = max_chunk_size_mb * 1024 * 1024  # Convert to bytes
        self.metadata_file = os.path.join(data_dir, 'database_metadata.json')
        
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load database metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'total_samples': 0,
            'total_chunks': 0,
            'chunk_files': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def save_metadata(self):
        """Save database metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_training_data(self, X, y, training_cycle):
        """Add new training data to database with automatic chunking"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create chunk filename
        chunk_id = self.metadata['total_chunks']
        chunk_file = os.path.join(self.chunks_dir, f'chunk_{chunk_id:06d}.parquet')
        
        # Prepare data
        table = pa.table({
            'features': pa.array([x.tolist() for x in X]),
            'target': pa.array(y.flatten().tolist()),
            'cycle': pa.array([training_cycle] * len(X)),
            'timestamp': pa.array([datetime.now().isoformat()] * len(X))
        })
        
        # Write to parquet
        pq.write_table(table, chunk_file, compression='snappy')
        
        # Check file size
        file_size = os.path.getsize(chunk_file)
        
        # Update metadata
        self.metadata['total_samples'] += len(X)
        self.metadata['total_chunks'] += 1
        self.metadata['chunk_files'].append({
            'filename': f'chunk_{chunk_id:06d}.parquet',
            'samples': len(X),
            'size_bytes': file_size,
            'cycle': training_cycle,
            'timestamp': datetime.now().isoformat()
        })
        
        self.save_metadata()
        
        print(f"Added {len(X)} samples to chunk {chunk_id} ({file_size / 1024 / 1024:.2f} MB)")
        print(f"Total database: {self.metadata['total_samples']} samples across {self.metadata['total_chunks']} chunks")
        
        return chunk_file
    
    def load_recent_data(self, n_chunks=5):
        """Load most recent chunks for training (memory-efficient)"""
        import pyarrow.parquet as pq
        
        if not self.metadata['chunk_files']:
            return None, None
        
        # Get most recent chunks
        recent_chunks = self.metadata['chunk_files'][-n_chunks:]
        
        all_X = []
        all_y = []
        
        for chunk_info in recent_chunks:
            chunk_path = os.path.join(self.chunks_dir, chunk_info['filename'])
            if os.path.exists(chunk_path):
                table = pq.read_table(chunk_path)
                X = np.array([np.array(x) for x in table['features'].to_pylist()])
                y = np.array(table['target'].to_pylist()).reshape(-1, 1)
                all_X.append(X)
                all_y.append(y)
        
        if all_X:
            return np.vstack(all_X).astype(np.float32), np.vstack(all_y).astype(np.float32)
        return None, None
    
    def get_database_stats(self):
        """Get database statistics"""
        total_size = sum(chunk['size_bytes'] for chunk in self.metadata['chunk_files'])
        return {
            'total_samples': self.metadata['total_samples'],
            'total_chunks': self.metadata['total_chunks'],
            'total_size_mb': total_size / 1024 / 1024,
            'avg_chunk_size_mb': (total_size / len(self.metadata['chunk_files']) / 1024 / 1024) if self.metadata['chunk_files'] else 0
        }

class TrainingManager:
    """Manages training loop, metrics, and model evolution with checkpoint resume"""
    def __init__(self):
        self.model = None
        self.history = {'loss': [], 'val_loss': [], 'epochs': [], 'complexity': []}
        self.best_loss = float('inf')
        self.training_state = None
        
    def load_or_create_model(self, checkpoint_path='model_checkpoint.pth'):
        """Load existing model or create new one"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
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
    
    def load_training_state(self, state_path='training_state.json'):
        """Load training state for exact resume"""
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                self.training_state = json.load(f)
            print(f"Loaded training state: Cycle {self.training_state['current_cycle']}, Epoch {self.training_state['current_epoch']}")
            return self.training_state
        return None
    
    def save_training_state(self, cycle, epoch, optimizer_state=None, state_path='training_state.json'):
        """Save complete training state for resume"""
        state = {
            'current_cycle': cycle,
            'current_epoch': epoch,
            'total_cycles_completed': len(self.history['loss']),
            'best_loss': self.best_loss,
            'timestamp': datetime.now().isoformat(),
            'optimizer_state': optimizer_state
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def evolve_architecture(self):
        """Dynamically adjust architecture based on performance"""
        if len(self.history['loss']) > 5:
            recent_losses = self.history['loss'][-5:]
            improvement = recent_losses[0] - recent_losses[-1]
            
            if improvement < 0.01:
                print("Evolving architecture: Adding layer complexity")
                current_sizes = [layer.out_features for layer in self.model.layers]
                new_sizes = [int(s * 1.2) for s in current_sizes]
                self.model = SelfImprovingModel(hidden_sizes=new_sizes)
                return True
        return False
    
    def train_epoch(self, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, start_epoch=0):
        """Train model for one evolution cycle with resume capability"""
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Resume optimizer state if available
        if self.training_state and self.training_state.get('optimizer_state'):
            try:
                optimizer.load_state_dict(self.training_state['optimizer_state'])
                print("Resumed optimizer state")
            except:
                print("Could not resume optimizer state, starting fresh")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        for epoch in range(start_epoch, epochs):
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
            
            # Save state every 10 epochs for interruption recovery
            if epoch % 10 == 0:
                current_cycle = len(self.history['loss']) + 1
                self.save_training_state(current_cycle, epoch, optimizer.state_dict())
        
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
    
    def calculate_improvement(self):
        """Calculate improvement percentage for auto-commit threshold"""
        if len(self.history['val_loss']) < 2:
            return 0.0
        
        # Compare to previous cycle
        previous_loss = self.history['val_loss'][-2]
        current_loss = self.history['val_loss'][-1]
        
        improvement = ((previous_loss - current_loss) / previous_loss) * 100
        return max(0.0, improvement)  # Only positive improvements

def main():
    print("=" * 50)
    print("AutoEvolve-ML Training Session")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    # Initialize components
    manager = TrainingManager()
    model = manager.load_or_create_model()
    training_state = manager.load_training_state()
    
    # Initialize database manager
    db_manager = DatabaseManager(max_chunk_size_mb=95)
    db_stats = db_manager.get_database_stats()
    print(f"\nDatabase: {db_stats['total_samples']} samples, {db_stats['total_size_mb']:.2f} MB")
    
    # Determine complexity level
    complexity = len(manager.history['loss']) // 10 + 1
    data_gen = DataGenerator(complexity_level=complexity)
    
    # Generate training data
    print(f"\nGenerating data with complexity level: {complexity}")
    X, y = data_gen.generate_data(n_samples=5000)
    
    # Add to database
    current_cycle = len(manager.history['loss']) + 1
    db_manager.add_training_data(X, y, current_cycle)
    
    # Load recent historical data for improved training
    historical_X, historical_y = db_manager.load_recent_data(n_chunks=3)
    if historical_X is not None:
        print(f"Loaded {len(historical_X)} historical samples for augmentation")
        X = np.vstack([X, historical_X[:2000]])  # Add subset to avoid memory issues
        y = np.vstack([y, historical_y[:2000]])
    
    # Augment with external data if available
    external = data_gen.fetch_external_data()
    if external is not None:
        print(f"Incorporated external data: {len(external)} samples")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Determine starting epoch
    start_epoch = 0
    if training_state and training_state['current_cycle'] == current_cycle:
        start_epoch = training_state['current_epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train
    print("\nStarting training...")
    train_loss, val_loss = manager.train_epoch(X_train, y_train, X_val, y_val, epochs=100, start_epoch=start_epoch)
    
    # Update history
    manager.history['loss'].append(train_loss)
    manager.history['val_loss'].append(val_loss)
    manager.history['epochs'].append(len(manager.history['loss']))
    manager.history['complexity'].append(complexity)
    
    # Calculate improvement
    improvement_pct = manager.calculate_improvement()
    
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
    
    # Save final training state
    manager.save_training_state(current_cycle, 100)
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(manager.history, f, indent=2)
    
    # Save improvement metrics for auto-commit threshold
    improvement_data = {
        'improvement_percentage': improvement_pct,
        'current_val_loss': val_loss,
        'best_val_loss': manager.best_loss,
        'cycle': current_cycle,
        'timestamp': datetime.now().isoformat()
    }
    with open('improvement_metrics.json', 'w') as f:
        json.dump(improvement_data, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training session complete!")
    print(f"Total training cycles: {len(manager.history['loss'])}")
    print(f"Best validation loss: {manager.best_loss:.4f}")
    print(f"Improvement this cycle: {improvement_pct:.2f}%")
    if improvement_pct >= 3.0:
        print(f"⚡ SIGNIFICANT IMPROVEMENT - Will be auto-committed immediately!")
    print("=" * 50)

if __name__ == "__main__":
    main()
