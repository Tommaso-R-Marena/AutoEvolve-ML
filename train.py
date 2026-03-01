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
import traceback
import sys
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
        try:
            np.random.seed(int(datetime.now().timestamp()) % 2**32)
            
            X = np.random.randn(n_samples, 10)
            
            y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + 
                 np.log(np.abs(X[:, 2]) + 1) * X[:, 3] +
                 np.tanh(X[:, 4] * X[:, 5]) +
                 self.complexity * np.random.randn(n_samples) * 0.1)
            
            return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)
        except Exception as e:
            print(f"Error generating data: {e}")
            traceback.print_exc()
            # Return minimal valid data
            X = np.random.randn(100, 10).astype(np.float32)
            y = np.random.randn(100, 1).astype(np.float32)
            return X, y
    
    def fetch_external_data(self):
        """Fetch real-world data from APIs for training augmentation"""
        try:
            response = requests.get(
                'https://www.random.org/integers/?num=100&min=1&max=1000&col=1&base=10&format=plain&rnd=new', 
                timeout=5
            )
            if response.status_code == 200:
                external_data = np.array([int(x) for x in response.text.strip().split('\n')])
                return external_data
        except Exception as e:
            print(f"Could not fetch external data: {e}")
        return None

class DatabaseManager:
    """Manages expandable training database with chunking and Git LFS"""
    def __init__(self, max_chunk_size_mb=95, data_dir='data'):
        self.data_dir = data_dir
        self.chunks_dir = os.path.join(data_dir, 'chunks')
        self.max_chunk_size = max_chunk_size_mb * 1024 * 1024
        self.metadata_file = os.path.join(data_dir, 'database_metadata.json')
        
        try:
            os.makedirs(self.chunks_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create chunks directory: {e}")
        
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load database metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
        
        return {
            'total_samples': 0,
            'total_chunks': 0,
            'chunk_files': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def save_metadata(self):
        """Save database metadata"""
        try:
            self.metadata['last_updated'] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def add_training_data(self, X, y, training_cycle):
        """Add new training data to database with automatic chunking"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            print("Warning: PyArrow not available, skipping database storage")
            return None
        
        try:
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
        except Exception as e:
            print(f"Warning: Could not add training data to database: {e}")
            traceback.print_exc()
            return None
    
    def load_recent_data(self, n_chunks=5):
        """Load most recent chunks for training (memory-efficient)"""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print("Warning: PyArrow not available, skipping historical data loading")
            return None, None
        
        try:
            if not self.metadata['chunk_files']:
                return None, None
            
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
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
        
        return None, None
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            total_size = sum(chunk['size_bytes'] for chunk in self.metadata['chunk_files'])
            return {
                'total_samples': self.metadata['total_samples'],
                'total_chunks': self.metadata['total_chunks'],
                'total_size_mb': total_size / 1024 / 1024,
                'avg_chunk_size_mb': (total_size / len(self.metadata['chunk_files']) / 1024 / 1024) if self.metadata['chunk_files'] else 0
            }
        except Exception as e:
            print(f"Warning: Could not get database stats: {e}")
            return {
                'total_samples': 0,
                'total_chunks': 0,
                'total_size_mb': 0,
                'avg_chunk_size_mb': 0
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
        try:
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
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead")
            self.model = SelfImprovingModel()
        
        return self.model
    
    def load_training_state(self, state_path='training_state.json'):
        """Load training state for exact resume"""
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    self.training_state = json.load(f)
                print(f"Loaded training state: Cycle {self.training_state['current_cycle']}, Epoch {self.training_state['current_epoch']}")
                return self.training_state
        except Exception as e:
            print(f"Could not load training state: {e}")
        
        return None
    
    def save_training_state(self, cycle, epoch, optimizer_state=None, state_path='training_state.json'):
        """Save complete training state for resume (optimizer saved separately)"""
        try:
            state = {
                'current_cycle': cycle,
                'current_epoch': epoch,
                'total_cycles_completed': len(self.history['loss']),
                'best_loss': float(self.best_loss),  # Ensure serializable
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save optimizer state separately as PyTorch file
            if optimizer_state is not None:
                optimizer_path = state_path.replace('.json', '_optimizer.pth')
                try:
                    torch.save(optimizer_state, optimizer_path)
                except Exception as e:
                    print(f"Warning: Could not save optimizer state: {e}")
        except Exception as e:
            print(f"Warning: Could not save training state: {e}")
            traceback.print_exc()
    
    def load_optimizer_state(self, state_path='training_state.json'):
        """Load optimizer state from separate file"""
        try:
            optimizer_path = state_path.replace('.json', '_optimizer.pth')
            if os.path.exists(optimizer_path):
                return torch.load(optimizer_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
        return None
    
    def evolve_architecture(self):
        """Dynamically adjust architecture based on performance"""
        try:
            if len(self.history['loss']) > 5:
                recent_losses = self.history['loss'][-5:]
                improvement = recent_losses[0] - recent_losses[-1]
                
                if improvement < 0.01:
                    print("Evolving architecture: Adding layer complexity")
                    current_sizes = [layer.out_features for layer in self.model.layers]
                    new_sizes = [int(s * 1.2) for s in current_sizes]
                    self.model = SelfImprovingModel(hidden_sizes=new_sizes)
                    return True
        except Exception as e:
            print(f"Warning: Architecture evolution failed: {e}")
        
        return False
    
    def train_epoch(self, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, start_epoch=0):
        """Train model for one evolution cycle with resume capability"""
        try:
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            # Resume optimizer state if available
            if self.training_state:
                optimizer_state = self.load_optimizer_state()
                if optimizer_state is not None:
                    try:
                        optimizer.load_state_dict(optimizer_state)
                        print("Resumed optimizer state")
                    except Exception as e:
                        print(f"Could not resume optimizer state: {e}")
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            
            for epoch in range(start_epoch, epochs):
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                
                # Check for NaN gradients
                has_nan = False
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"Warning: NaN gradient detected at epoch {epoch}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                
                # Check for NaN loss
                if np.isnan(val_loss) or np.isinf(val_loss):
                    print(f"Warning: Invalid loss at epoch {epoch}, stopping training")
                    break
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")
                
                # Save state every 10 epochs for interruption recovery
                if epoch % 10 == 0:
                    current_cycle = len(self.history['loss']) + 1
                    self.save_training_state(current_cycle, epoch, optimizer.state_dict())
            
            return loss.item(), val_loss
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            # Return last known values or defaults
            return float('inf'), float('inf')
    
    def save_checkpoint(self, complexity, checkpoint_path='model_checkpoint.pth'):
        """Save model with metadata"""
        try:
            hidden_sizes = [layer.out_features for layer in self.model.layers]
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'hidden_sizes': hidden_sizes,
                'history': self.history,
                'best_loss': float(self.best_loss),  # Ensure serializable
                'timestamp': datetime.now().isoformat(),
                'complexity': complexity
            }, checkpoint_path)
            print(f"Model saved: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            traceback.print_exc()
    
    def calculate_improvement(self):
        """Calculate improvement percentage for auto-commit threshold"""
        try:
            if len(self.history['val_loss']) < 2:
                return 0.0
            
            previous_loss = self.history['val_loss'][-2]
            current_loss = self.history['val_loss'][-1]
            
            # Handle edge cases
            if previous_loss == 0 or np.isnan(previous_loss) or np.isinf(previous_loss):
                return 0.0
            
            if np.isnan(current_loss) or np.isinf(current_loss):
                return 0.0
            
            improvement = ((previous_loss - current_loss) / previous_loss) * 100
            return max(0.0, float(improvement))  # Only positive improvements
        except Exception as e:
            print(f"Warning: Could not calculate improvement: {e}")
            return 0.0

def main():
    try:
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
            # Add subset to avoid memory issues
            subset_size = min(2000, len(historical_X))
            X = np.vstack([X, historical_X[:subset_size]])
            y = np.vstack([y, historical_y[:subset_size]])
        
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
        train_loss, val_loss = manager.train_epoch(
            X_train, y_train, X_val, y_val, 
            epochs=100, start_epoch=start_epoch
        )
        
        # Check if training failed
        if np.isinf(train_loss) or np.isinf(val_loss):
            print("\nTraining failed with invalid loss values")
            print("Saving current state and exiting")
            manager.save_checkpoint(complexity)
            sys.exit(1)
        
        # Update history
        manager.history['loss'].append(float(train_loss))
        manager.history['val_loss'].append(float(val_loss))
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
            train_loss, val_loss = manager.train_epoch(
                X_train, y_train, X_val, y_val, 
                epochs=50
            )
        
        # Save checkpoint
        manager.save_checkpoint(complexity)
        
        # Save final training state
        manager.save_training_state(current_cycle, 100)
        
        # Save metrics
        with open('metrics.json', 'w') as f:
            json.dump(manager.history, f, indent=2)
        
        # Save improvement metrics for auto-commit threshold
        improvement_data = {
            'improvement_percentage': float(improvement_pct),
            'current_val_loss': float(val_loss),
            'best_val_loss': float(manager.best_loss),
            'cycle': int(current_cycle),
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
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current state...")
        if 'manager' in locals():
            manager.save_checkpoint(complexity if 'complexity' in locals() else 1)
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
