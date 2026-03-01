import numpy as np
import requests
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealDataIntegration:
    """Integrates real-world datasets to inform and improve synthetic data generation"""
    
    def __init__(self):
        self.cache = {}
        self.data_quality_metrics = []
    
    def fetch_sklearn_dataset(self, dataset_name='diabetes'):
        """Load real datasets from scikit-learn for reference"""
        try:
            from sklearn import datasets
            
            if dataset_name == 'diabetes':
                data = datasets.load_diabetes()
                return data.data, data.target.reshape(-1, 1)
            elif dataset_name == 'california_housing':
                data = datasets.fetch_california_housing()
                return data.data[:1000], data.target[:1000].reshape(-1, 1)
            elif dataset_name == 'wine':
                data = datasets.load_wine()
                return data.data, data.target.reshape(-1, 1)
        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
        return None, None
    
    def fetch_uci_data(self):
        """Fetch data from UCI Machine Learning Repository"""
        try:
            # Iris dataset (classic, always available)
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                data = []
                for line in lines:
                    if line:
                        parts = line.split(',')[:-1]  # Exclude class label
                        try:
                            data.append([float(x) for x in parts])
                        except:
                            continue
                if data:
                    X = np.array(data)
                    # Simple regression target
                    y = np.sum(X, axis=1).reshape(-1, 1)
                    return X, y
        except Exception as e:
            print(f"Could not fetch UCI data: {e}")
        return None, None
    
    def fetch_openml_sample(self):
        """Fetch sample from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            # Fetch a small regression dataset
            data = fetch_openml(name='boston', version=1, parser='auto')
            X = data.data.to_numpy() if hasattr(data.data, 'to_numpy') else data.data
            y = data.target.to_numpy() if hasattr(data.target, 'to_numpy') else data.target
            return X[:500], y[:500].reshape(-1, 1)
        except Exception as e:
            print(f"Could not fetch OpenML data: {e}")
        return None, None
    
    def analyze_real_data_characteristics(self, X, y):
        """Analyze statistical properties of real data"""
        try:
            characteristics = {
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'feature_means': np.mean(X, axis=0).tolist(),
                'feature_stds': np.std(X, axis=0).tolist(),
                'feature_correlations': np.corrcoef(X.T).tolist(),
                'target_mean': float(np.mean(y)),
                'target_std': float(np.std(y)),
                'target_range': [float(np.min(y)), float(np.max(y))],
                'feature_target_corr': [float(np.corrcoef(X[:, i], y.flatten())[0, 1]) for i in range(X.shape[1])],
                'skewness': [float(self._skewness(X[:, i])) for i in range(X.shape[1])],
                'kurtosis': [float(self._kurtosis(X[:, i])) for i in range(X.shape[1])]
            }
            return characteristics
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return None
    
    def _skewness(self, x):
        """Calculate skewness"""
        try:
            mean = np.mean(x)
            std = np.std(x)
            return np.mean(((x - mean) / std) ** 3) if std > 0 else 0
        except:
            return 0
    
    def _kurtosis(self, x):
        """Calculate kurtosis"""
        try:
            mean = np.mean(x)
            std = np.std(x)
            return np.mean(((x - mean) / std) ** 4) - 3 if std > 0 else 0
        except:
            return 0
    
    def get_best_real_dataset(self):
        """Try multiple sources and return the best available dataset"""
        datasets = [
            ('sklearn_diabetes', self.fetch_sklearn_dataset, {'dataset_name': 'diabetes'}),
            ('sklearn_california', self.fetch_sklearn_dataset, {'dataset_name': 'california_housing'}),
            ('uci_iris', self.fetch_uci_data, {}),
            ('openml_boston', self.fetch_openml_sample, {})
        ]
        
        for name, fetcher, kwargs in datasets:
            try:
                X, y = fetcher(**kwargs)
                if X is not None and y is not None:
                    print(f"Successfully loaded real dataset: {name} ({X.shape[0]} samples, {X.shape[1]} features)")
                    return X, y, name
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                continue
        
        return None, None, None
