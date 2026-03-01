# AutoEvolve-ML: Self-Improving Machine Learning System

[![Auto-Train](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/auto-train.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/auto-train.yml)
[![Model Evaluation](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/evaluate.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/evaluate.yml)

A novel self-improving machine learning system that continuously evolves through automated training, synthetic data generation, and architectural adaptation. The model trains automatically via GitHub Actions and Google Colab, with weights persistently saved to the repository.

## 🚀 Key Features

- **Self-Improving Architecture**: Neural network that dynamically evolves its structure based on performance metrics
- **Automated Training**: GitHub Actions workflows train the model every 6 hours without manual intervention
- **Progressive Complexity**: Synthetic data generation with increasing difficulty levels
- **External Data Integration**: API calls to fetch real-world data for training augmentation
- **Colab Integration**: Jupyter notebook for extended training sessions with automatic GitHub synchronization
- **Persistent Evolution**: Every training session builds on previous weights, creating continuous improvement
- **Weekly Evaluation**: Automated performance reports generated and posted as GitHub Issues

## 🏗️ Architecture

### Core Components

1. **SelfImprovingModel**: PyTorch neural network with dynamic layer evolution
2. **DataGenerator**: Creates increasingly complex synthetic datasets and fetches external data
3. **TrainingManager**: Orchestrates training loops, checkpointing, and architectural evolution
4. **CI/CD Pipeline**: GitHub Actions for automated training and evaluation

### Training Evolution Process

```
Initial Model → Train → Evaluate → Evolve Architecture → Generate Complex Data → Train → ...
      ↓                                                                              ↓
  Save to GitHub ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← Save to GitHub
```

## 📦 Installation

### Local Setup

```bash
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ML.git
cd AutoEvolve-ML
pip install -r requirements.txt
python train.py
```

### Google Colab Setup

1. Open `colab_training.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Add your GitHub Personal Access Token to Colab Secrets:
   - Click the key icon (🔑) in the left sidebar
   - Add a new secret named `GITHUB_TOKEN`
   - Paste your token (requires `repo` scope)
3. Run all cells

**Generate GitHub Token**: [Settings → Developer settings → Personal access tokens → Tokens (classic)](https://github.com/settings/tokens) with `repo` permissions

## 🤖 Automated Training

### GitHub Actions Workflows

#### Auto-Train (`auto-train.yml`)
- **Schedule**: Every 6 hours
- **Triggers**: Push to `main`, manual dispatch
- **Actions**: 
  - Pulls latest code and weights
  - Runs training session
  - Commits updated model checkpoint
  - Uploads artifacts

#### Evaluation (`evaluate.yml`)
- **Schedule**: Weekly on Sundays
- **Actions**:
  - Generates comprehensive performance report
  - Creates GitHub Issue with evaluation metrics
  - Tracks architecture evolution

### Continuous Evolution Strategy

1. **Training Cycle**: 100 epochs per session
2. **Data Complexity**: Increases every 10 training cycles
3. **Architecture Evolution**: Triggered when improvement < 1% over 5 cycles
4. **External Data**: Random seed variation from external APIs
5. **Checkpointing**: Full state saved after every cycle

## 📊 Model Features

### Dynamic Architecture
- **Initial**: 3 hidden layers [64, 128, 64]
- **Evolution**: Layers expand by 20% when performance plateaus
- **Dropout**: 0.2 for regularization
- **Activation**: ReLU with adaptive learning rate

### Data Generation
- **Synthetic Function**: `y = sin(x₁)·cos(x₂) + log(|x₃|+1)·x₄ + tanh(x₅·x₆) + complexity·noise`
- **Samples**: 5,000 per training session
- **Features**: 10-dimensional input space
- **Validation Split**: 80/20 train/test

### Training Metrics
- Training loss (MSE)
- Validation loss
- Best validation loss (historical)
- Training cycles completed
- Current complexity level
- Architecture parameters

## 📈 Monitoring Progress

### View Training History

Metrics are stored in `metrics.json`:

```json
{
  "loss": [0.523, 0.412, 0.389, ...],
  "val_loss": [0.531, 0.425, 0.401, ...],
  "epochs": [1, 2, 3, ...],
  "complexity": [1, 1, 1, 2, 2, ...]
}
```

### GitHub Actions Logs

[View workflow runs](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions) to see training progress, loss curves, and evolution events.

### Evaluation Reports

Weekly evaluation reports are automatically posted as [GitHub Issues](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/issues) with:
- Total training cycles
- Performance improvements
- Architecture details
- Parameter counts

## 🔧 Customization

### Modify Training Schedule

Edit `.github/workflows/auto-train.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Change to desired frequency
```

### Adjust Model Architecture

Edit `train.py`:

```python
self.model = SelfImprovingModel(
    input_size=10,
    hidden_sizes=[128, 256, 128],  # Customize layer sizes
    output_size=1
)
```

### Change Data Complexity

Modify `DataGenerator.generate_data()` in `train.py` to adjust the target function complexity.

### External API Integration

Add custom API calls in `DataGenerator.fetch_external_data()` for domain-specific data augmentation.

## 🛠️ Advanced Usage

### Manual Training Trigger

```bash
# Via GitHub Actions UI
Actions → Auto-Train Self-Improving Model → Run workflow

# Or locally
python train.py
git add model_checkpoint.pth metrics.json
git commit -m "Manual training update"
git push
```

### Extended Colab Sessions

For intensive training (recommended with Colab Pro):

```python
ITERATIONS = 50  # Increase in notebook
PUSH_FREQUENCY = 5  # Push every 5 iterations
```

### Model Inference

```python
import torch
from train import SelfImprovingModel

# Load model
checkpoint = torch.load('model_checkpoint.pth')
model = SelfImprovingModel(hidden_sizes=checkpoint['hidden_sizes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
import numpy as np
X_new = np.random.randn(10, 10).astype(np.float32)
with torch.no_grad():
    predictions = model(torch.FloatTensor(X_new))
print(predictions)
```

## 🎯 Use Cases

- **Research**: Study self-improving ML systems and architectural evolution
- **Benchmarking**: Continuous baseline for testing new algorithms
- **Education**: Learn about automated ML pipelines and CI/CD for models
- **Experimentation**: Framework for testing self-evolution strategies

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Additional evolution strategies (genetic algorithms, NAS)
- Multi-objective optimization (accuracy + efficiency)
- Transfer learning from external datasets
- Reinforcement learning for architecture search
- Distributed training across multiple runners

## 📝 License

MIT License - feel free to use and modify for your projects.

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Google Colab](https://colab.research.google.com/)

---

**Built with ❤️ for autonomous ML systems**

*Last auto-trained: Check [Actions](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions) for latest run*
