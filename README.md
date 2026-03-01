# AutoEvolve-ML: Self-Improving Machine Learning System

[![Auto-Train](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/auto-train.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/auto-train.yml)
[![Model Evaluation](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/evaluate.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/evaluate.yml)
[![Quality Gate](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/quality-gate.yml)

A novel self-improving machine learning system that continuously evolves through automated training, synthetic data generation, and **self-modification with quality gates**. The model trains automatically via GitHub Actions and Google Colab, with weights persistently saved to the repository. When performance stagnates, the system proposes modifications to its own code, but must pass rigorous quality thresholds before changes are approved.

## 🚀 Key Features

- **Self-Improving Architecture**: Neural network that dynamically evolves its structure based on performance metrics
- **Self-Modifying Code**: System analyzes its own performance and proposes code changes to improve training
- **Quality Gates**: All self-modifications must pass tests, coverage requirements, and performance thresholds
- **Automated Training**: GitHub Actions workflows train the model every 6 hours without manual intervention
- **Progressive Complexity**: Synthetic data generation with increasing difficulty levels
- **External Data Integration**: API calls to fetch real-world data for training augmentation
- **Colab Integration**: Jupyter notebook for extended training sessions with automatic GitHub synchronization
- **Persistent Evolution**: Every training session builds on previous weights, creating continuous improvement
- **CI/CD Testing**: Comprehensive test suite ensures modifications don't degrade performance
- **Weekly Evaluation**: Automated performance reports generated and posted as GitHub Issues

## 🏗️ Architecture

### Core Components

1. **SelfImprovingModel**: PyTorch neural network with dynamic layer evolution
2. **DataGenerator**: Creates increasingly complex synthetic datasets and fetches external data
3. **TrainingManager**: Orchestrates training loops, checkpointing, and architectural evolution
4. **SelfModifier**: Analyzes performance and proposes code modifications with confidence scores
5. **Quality Gate Pipeline**: CI/CD system that validates all changes before approval
6. **Threshold Checker**: Ensures performance, coverage, and code quality standards are met

### Self-Modification Process

```
Train → Analyze Performance → Detect Stagnation → Propose Modification
                                                           ↓
                                                    Create PR with Changes
                                                           ↓
                                            Run Quality Gate Checks:
                                            - Unit tests (must pass)
                                            - Coverage ≥80%
                                            - Performance validation
                                            - Code quality checks
                                                           ↓
                                            ┌──────────────┴──────────────┐
                                        PASS                            FAIL
                                            ↓                               ↓
                                    Auto-merge PR                    Block merge
                                            ↓                         Post reasons
                                    Continue training              Human review
```

## 📦 Installation

### Local Setup

```bash
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ML.git
cd AutoEvolve-ML
pip install -r requirements.txt
pip install pytest pytest-cov black flake8  # For testing
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

## 🤖 Automated Training & Self-Modification

### GitHub Actions Workflows

#### Auto-Train (`auto-train.yml`)
- **Schedule**: Every 6 hours
- **Triggers**: Push to `main`, manual dispatch
- **Actions**: 
  - Runs pre-training tests
  - Executes training session
  - Analyzes if self-modification is needed
  - Creates PR with proposed changes if stagnating
  - Commits weights directly if performing well

#### Quality Gate (`quality-gate.yml`)
- **Triggers**: Pull requests to `main`
- **Required Checks**:
  - ✅ All unit tests pass
  - ✅ Test coverage ≥80%
  - ✅ Model performance within 2x of historical best
  - ✅ No critical linting errors
  - ✅ No NaN/Inf values in outputs
- **Actions**:
  - Runs complete test suite with coverage
  - Validates model performance on fresh data
  - Posts detailed report to PR
  - Blocks merge if any check fails

#### Evaluation (`evaluate.yml`)
- **Schedule**: Weekly on Sundays
- **Actions**:
  - Generates comprehensive performance report
  - Creates GitHub Issue with evaluation metrics
  - Tracks architecture evolution

### Self-Modification Thresholds

**When does the system propose changes?**
- Recent improvement <5% over last 10 cycles
- Confidence in proposed change ≥70%
- No active modification PR pending

**What can it modify?**
- **Architecture**: Add layers, adjust sizes
- **Regularization**: Add L2, dropout adjustments
- **Training**: Learning rate schedules, optimization strategies
- **Tests**: Automatically generates tests for new architectures

**Quality Requirements**
- All existing tests must pass
- Test coverage maintained at ≥80%
- Model performance cannot degrade >2x from best
- No critical code quality issues (syntax errors, undefined variables)
- Changes must improve expected performance by ≥10%

## 📊 Model Features

### Dynamic Architecture
- **Initial**: 3 hidden layers [64, 128, 64]
- **Evolution**: Layers expand by 20% when performance plateaus
- **Self-Modification**: Adds layers or changes training strategy when stagnation detected
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
- Self-modification attempts and success rate

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

[View workflow runs](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions) to see:
- Training progress and loss curves
- Self-modification proposals and decisions
- Quality gate results
- Test coverage reports

### Self-Modification PRs

When the system proposes changes, it creates a PR with:
- Description of performance issue
- Proposed modifications with confidence scores
- Expected improvement percentage
- Auto-generated tests for changes
- Quality gate status

### Running Tests Locally

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term --cov-report=html

# Run quality checks
python check_thresholds.py

# Validate model performance
python validate_model.py
```

## 🔧 Customization

### Modify Training Schedule

Edit `.github/workflows/auto-train.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Change to desired frequency
```

### Adjust Quality Thresholds

Edit `self_modifier.py`:

```python
self.thresholds = {
    'min_improvement': 0.05,      # 5% minimum improvement
    'stagnation_cycles': 10,       # Cycles before modification
    'confidence_threshold': 0.7    # Confidence in improvement
}
```

Edit `check_thresholds.py` for quality gate thresholds:

```python
# Coverage requirement
if coverage >= 80:  # Adjust this value

# Performance degradation limit
if current_loss < best_loss * 2.0:  # Adjust multiplier
```

### Add Custom Self-Modifications

Extend `SelfModifier.propose_architecture_modification()` in `self_modifier.py`:

```python
elif condition:
    modification = {
        'type': 'custom_modification',
        'description': 'Your modification description',
        'confidence': 0.8,
        'expected_improvement': 0.15
    }
```

Then implement in `generate_modified_code()`.

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

### Test Self-Modification Locally

```bash
# Analyze performance and create proposal
python self_modifier.py

# Check if modification_proposal.json was created
cat modification_proposal.json

# Validate quality thresholds
python check_thresholds.py
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

- **Research**: Study self-improving ML systems and code evolution
- **AutoML**: Automated architecture search with safety guarantees
- **CI/CD for ML**: Learn best practices for testing ML systems
- **Education**: Understand quality gates and testing requirements for production ML
- **Experimentation**: Framework for testing self-evolution strategies with constraints

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Additional self-modification strategies (genetic algorithms, NAS)
- Multi-objective optimization (accuracy + efficiency + interpretability)
- More sophisticated quality metrics (fairness, robustness)
- Integration with external datasets and APIs
- Advanced testing strategies (property-based testing, adversarial validation)
- Distributed training with quality gates

## 📝 License

MIT License - feel free to use and modify for your projects.

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)
- [Google Colab](https://colab.research.google.com/)

---

**Built with ❤️ for safe, autonomous ML systems**

*Last auto-trained: Check [Actions](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions) for latest run*
*Self-modifications proposed: Check [Pull Requests](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/pulls) for pending changes*
