# Contributing to AutoEvolve-ML

🎉 Thank you for your interest in contributing to this Nobel Prize-level research system!

## 🎯 Vision

AutoEvolve-ML aims to push the boundaries of autonomous machine learning research through:
- Quantum-inspired optimization
- Causal discovery and intervention
- Nobel-level research automation
- Meta-learning and architecture search

## 🚀 Areas for Contribution

### 1. Quantum Algorithms
- Improve quantum annealing schedules
- Implement additional quantum-inspired operators
- Add support for actual quantum hardware (IBM Quantum, IonQ)
- Develop quantum circuit optimization

### 2. Causal Discovery
- Implement advanced causal discovery methods (PC algorithm, FCI)
- Add support for latent confounders
- Improve intervention effect estimation
- Develop causal reinforcement learning

### 3. Research Automation
- Enhance hypothesis generation (LLM integration)
- Improve experimental design (Bayesian optimization)
- Add automatic theorem proving
- Develop research collaboration protocols

### 4. Architecture & Optimization
- Implement differentiable NAS
- Add multi-objective optimization (Pareto frontiers)
- Develop neural tangent kernel analysis
- Lottery ticket hypothesis testing

### 5. Visualization & UI
- Create interactive dashboards (Streamlit, Dash)
- Develop real-time monitoring
- Add 3D loss landscape visualization
- Build research progress tracker

### 6. Infrastructure
- Optimize GitHub Actions workflows
- Add cloud platform support (AWS, GCP, Azure)
- Implement distributed training
- Develop self-hosted runner support

## 📝 Getting Started

### Prerequisites

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ML.git
cd AutoEvolve-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test
pytest tests/test_quantum_optimizer.py -v
```

### Code Style

- **PEP 8** compliance (use `black` for formatting)
- **Type hints** for function signatures
- **Docstrings** for all public functions/classes
- **Comments** for complex algorithms

```python
# Good example
def quantum_tunneling(state: Dict[str, float], 
                     temperature: float) -> Dict[str, float]:
    """Apply quantum tunneling to escape local minima.
    
    Args:
        state: Current optimization state
        temperature: Annealing temperature
        
    Returns:
        Modified state after quantum fluctuations
    """
    # Implementation
    pass
```

## 🔥 Pull Request Process

### 1. Fork & Branch

```bash
# Fork repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/AutoEvolve-ML.git
cd AutoEvolve-ML

# Create feature branch
git checkout -b feature/quantum-circuit-optimization
```

### 2. Make Changes

- Keep commits atomic and focused
- Write descriptive commit messages
- Add tests for new features
- Update documentation

```bash
git add quantum_optimizer.py tests/test_quantum_optimizer.py
git commit -m "feat: Add quantum circuit optimization with variational layers

- Implement parameterized quantum circuits
- Add gradient computation for quantum parameters
- Include tests for circuit optimization
- Update documentation with usage examples"
```

### 3. Test Locally

```bash
# Run tests
pytest tests/ -v

# Check code style
black --check .
flake8 .

# Run type checking
mypy .
```

### 4. Push & Create PR

```bash
git push origin feature/quantum-circuit-optimization
```

Then create pull request on GitHub with:
- **Clear title** describing the change
- **Detailed description** of what was added/changed
- **Testing results** showing it works
- **Screenshots** if UI changes
- **Breaking changes** if any

### 5. Code Review

- Address reviewer feedback promptly
- Keep discussions respectful and constructive
- Be open to suggestions and alternative approaches

## 📚 Documentation

### Adding Documentation

1. **Code documentation**: Docstrings in code
2. **README updates**: For major features
3. **Examples**: Add to `examples/` directory
4. **Tutorials**: Add to `docs/tutorials/`

### Documentation Style

```python
class QuantumOptimizer:
    """Quantum-inspired optimization engine.
    
    Implements quantum annealing for hyperparameter search,
    allowing quantum tunneling through local minima.
    
    Attributes:
        temperature_schedule: Cooling schedule for annealing
        best_state: Best found configuration
        best_energy: Lowest energy (loss) achieved
        
    Example:
        >>> optimizer = QuantumOptimizer()
        >>> best_state, best_loss = optimizer.optimize(
        ...     initial_state={'lr': 0.001},
        ...     evaluate_fn=train_model,
        ...     n_steps=100
        ... )
    """
```

## ✅ Testing Guidelines

### Test Structure

```python
import pytest
from quantum_optimizer import QuantumInspiredOptimizer

class TestQuantumOptimizer:
    """Tests for quantum optimization."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = QuantumInspiredOptimizer()
        assert optimizer.best_state is None
        assert optimizer.best_energy == float('inf')
    
    def test_optimization_convergence(self):
        """Test that optimization converges."""
        def simple_fn(state):
            return (state['x'] - 5) ** 2
        
        optimizer = QuantumInspiredOptimizer()
        initial = {'x': 0}
        best_state, best_loss = optimizer.optimize(
            initial, simple_fn, n_steps=50
        )
        
        assert abs(best_state['x'] - 5) < 0.5
        assert best_loss < 1.0
    
    @pytest.mark.slow
    def test_quantum_tunneling(self):
        """Test quantum tunneling escapes local minima."""
        # Implementation
        pass
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests
- Add performance benchmarks

## 🌟 Research Contributions

### Novel Algorithms

If proposing a novel algorithm:

1. **Literature review**: Cite related work
2. **Theoretical justification**: Explain why it should work
3. **Experimental validation**: Show it improves results
4. **Reproducibility**: Include seeds, configs, results

### Research Paper Integration

If implementing from a paper:

```python
"""Implementation of Quantum Circuit Learning.

Based on:
    Mitarai et al. (2018). "Quantum Circuit Learning"
    Physical Review A, 98(3), 032309.
    https://doi.org/10.1103/PhysRevA.98.032309
    
Key differences from paper:
    - Classical simulation instead of quantum hardware
    - Gradient estimation uses parameter shift rule
    - Adapted for supervised learning tasks
"""
```

## 🐛 Bug Reports

### Reporting Bugs

Use GitHub Issues with:

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
1. Run command...
2. With config...
3. See error...

**Expected Behavior**
What should happen

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10
- PyTorch: 2.0.0
- Commit: abc123

**Logs**
```
Error traceback here
```

**Additional Context**
Any other relevant information
```

### Security Issues

For security vulnerabilities, email directly instead of public issue.

## ✨ Feature Requests

### Proposing Features

```markdown
**Feature Description**
Clear description of proposed feature

**Motivation**
Why is this needed? What problem does it solve?

**Proposed Solution**
How would this work? Include examples.

**Alternatives Considered**
What other approaches were considered?

**Additional Context**
References, papers, examples from other projects
```

## 🏅 Recognition

Significant contributors will be:
- Listed in AUTHORS.md
- Acknowledged in research papers
- Invited to collaborate on publications
- Given co-authorship for major contributions

### Contribution Tiers

- **🥇 Gold**: Major algorithm/feature (>500 lines, novel approach)
- **🥈 Silver**: Significant improvement (>200 lines, clear impact)
- **🥉 Bronze**: Bug fixes, documentation, minor features

## 💬 Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Research ideas, questions
- **Pull Requests**: Code contributions
- **Email**: Sensitive topics, collaborations

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🚀 Quick Contribution Checklist

- [ ] Fork repository
- [ ] Create feature branch
- [ ] Write code with tests
- [ ] Add documentation
- [ ] Run test suite locally
- [ ] Commit with clear message
- [ ] Push to your fork
- [ ] Create pull request
- [ ] Address review feedback

---

**Thank you for helping make AutoEvolve-ML a breakthrough research system!**

*Questions? Open a GitHub Discussion or contact the maintainers.*
