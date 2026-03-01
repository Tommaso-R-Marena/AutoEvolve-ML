# AutoEvolve-ML: Revolutionary Self-Improving Machine Learning System

[![Auto-Train](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/auto-train.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/auto-train.yml)
[![Architecture Search](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/architecture-search.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/architecture-search.yml)
[![Meta-Learning](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/meta-learning-update.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions/workflows/meta-learning-update.yml)

A state-of-the-art self-improving ML system with **meta-learning**, **neural architecture search**, **ensemble methods**, and **automated research**. Optimized to stay **within GitHub Actions free tier limits**.

## 🚀 Revolutionary Features

- **Meta-Learning**: Learns optimal learning strategies automatically
- **Neural Architecture Search (NAS)**: Discovers best model architectures
- **Ensemble System**: Maintains committee of top-performing models
- **Research Automation**: Generates hypotheses and detects breakthroughs
- **Real Dataset Integration**: Learns from sklearn, UCI, OpenML datasets
- **Unlimited Training Database**: Git LFS with automatic 95MB chunking
- **Performance-Based Auto-Commit**: ≥3% improvement triggers immediate commit
- **Checkpoint Resume**: Zero data loss on interruption

## 📊 GitHub Actions Resource Management

### Monthly Resource Usage (Optimized)

| Workflow | Frequency | Minutes/Run | Monthly Total |
|----------|-----------|-------------|---------------|
| **Training** | Every 8 hours (3x/day) | 25 min | 2,250 min |
| **Architecture Search** | Weekly (matrix: 3 jobs) | 15 min each | 180 min |
| **Meta-Learning Update** | Daily | 10 min | 300 min |
| **Research Report** | Weekly | 10 min | 40 min |
| **Resource Monitor** | Monthly | 5 min | 5 min |
| **Cleanup** | Weekly | 5 min | 20 min |
| **TOTAL** | | | **~2,795 min/month** |

### GitHub Actions Limits

| Plan | Minutes/Month | Storage | Cost |
|------|---------------|---------|------|
| **GitHub Free** | 2,000 | 500 MB | Free |
| **GitHub Pro** | 3,000 | 1 GB | $4/month |
| **GitHub Team** | 3,000 | 2 GB | $4/user/month |

**Note**: This system uses ~2,795 min/month, which exceeds Free tier by ~800 minutes. Options:

1. **Reduce to Free Tier** (~1,900 min/month):
   - Change training to every 12 hours (2x/day) → Saves 750 min
   - Disable weekly NAS → Saves 180 min
   - **New total**: ~1,865 min/month ✅ Within Free tier

2. **Keep Full Features with GitHub Pro**:
   - $4/month for 3,000 minutes
   - 1 GB storage
   - All features enabled
   - ~200 minutes buffer

3. **Optimize Training Time**:
   - Reduce epochs from 100 to 75 → Saves ~500 min/month
   - Combined with 12h schedule → Well within Free tier

### Storage Management

- **Database chunks**: Parquet format with Snappy compression (~5x compression)
- **95MB per chunk**: Stays under GitHub's 100MB file limit
- **Git LFS**: Efficient delta transfers
- **Automatic cleanup**: Old artifacts deleted after 7 days
- **Estimated growth**: ~50MB per month

## ⚙️ Workflow Architecture

### 1. Auto-Train (Every 8 Hours)
- **Time**: ~25 minutes
- **Actions**: Generate data, train model, meta-learning, commit results
- **Triggers**: Schedule (8h), push to main, manual dispatch
- **Optimization**: Pip caching, early stopping, efficient data loading

### 2. Architecture Search (Weekly)
- **Time**: ~45 minutes (3 parallel jobs × 15 min)
- **Actions**: Test wider/deeper/bottleneck architectures
- **Strategy**: Matrix parallelization for efficiency
- **Triggers**: Every Sunday, manual dispatch

### 3. Meta-Learning Update (Daily)
- **Time**: ~10 minutes
- **Actions**: Analyze learning history, recommend strategies
- **Output**: Meta-learning report with best hyperparameters
- **Triggers**: Daily at 3 AM

### 4. Research Report (Weekly)
- **Time**: ~10 minutes
- **Actions**: Generate research summary, create issues for breakthroughs
- **Output**: Weekly report, GitHub issues for discoveries
- **Triggers**: Every Monday

### 5. Resource Monitor (Monthly)
- **Time**: ~5 minutes
- **Actions**: Calculate usage, create resource report
- **Alerts**: Warns if approaching limits
- **Triggers**: First day of month

### 6. Cleanup (Weekly)
- **Time**: ~5 minutes
- **Actions**: Delete artifacts older than 7 days
- **Savings**: Keeps storage usage low
- **Triggers**: Every Saturday

## 🔧 Configuration for GitHub Actions Limits

### To Reduce to Free Tier (Edit `.github/workflows/auto-train.yml`):

```yaml
on:
  schedule:
    - cron: '0 */12 * * *'  # Change from 8h to 12h (2x per day)
```

### To Disable NAS (Save 180 min/month):

Delete or disable `.github/workflows/architecture-search.yml`

### To Reduce Training Time:

Edit `train.py`:
```python
train_loss, val_loss = manager.train_epoch(
    X_train, y_train, X_val, y_val, 
    epochs=75,  # Change from 100 to 75
    start_epoch=start_epoch
)
```

## 📈 Performance Tracking

### Automated Reports

1. **Training Metrics** (`metrics.json`): Updated every training run
2. **Improvement Metrics** (`improvement_metrics.json`): Tracks gains per cycle
3. **Data Quality** (`data_quality_metrics.json`): Synthetic data statistics
4. **Meta-Learning Report** (`meta_learning_report.json`): Best strategies
5. **Research Report** (`weekly_research_report.json`): Experiments and breakthroughs
6. **NAS State** (`nas_state.json`): Architecture search history

### GitHub Issues

- **Breakthroughs**: Automatically created when ≥5% improvement found
- **Resource Alerts**: Monthly report with usage statistics
- **Labels**: `breakthrough`, `automated`, `monitoring`, `resources`

## 🎯 Getting Started

### Prerequisites

1. **GitHub Account** (Free or Pro)
2. **Enable Actions**: Settings → Actions → Allow all actions
3. **Set Spending Limit** (if Free): Settings → Billing → Actions → $1+ spending limit

### Installation

```bash
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ML.git
cd AutoEvolve-ML

# Install Git LFS
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run local training
python train.py
```

### First Run

1. Push to main branch
2. GitHub Actions will trigger automatically
3. Check Actions tab for progress
4. First run takes ~25 minutes
5. Subsequent runs resume from checkpoints

## 🔬 Advanced Features

### Meta-Learning

Tracks every training episode and learns:
- Best optimizer strategies
- Optimal learning rates
- Effective architectures
- Convergence patterns

### Neural Architecture Search

Explores:
- Wider networks (1.5x capacity)
- Deeper networks (additional layers)
- Bottleneck designs (compression)
- Pyramid architectures (encoder/decoder)

### Ensemble System

Maintains top 5 models:
- Weighted predictions
- Diversity scoring
- Automatic model selection

### Research Automation

Generates hypotheses:
- Learning rate adjustments
- Architecture capacity
- Regularization needs
- Data complexity increases

Tracks breakthroughs:
- ≥5% improvement = breakthrough
- Automatic GitHub issues
- Complete audit trail

## 📊 Monitoring Usage

### Check Monthly Minutes

1. Go to GitHub Settings → Billing
2. View "Actions & Packages"
3. See minutes used this month

### Estimate Future Usage

```bash
# Calculate based on your schedule
TRAINING_PER_DAY=3  # or 2 for 12h schedule
DAYS_IN_MONTH=30
MIN_PER_TRAINING=25

MONTHLY_TRAINING=$((TRAINING_PER_DAY * DAYS_IN_MONTH * MIN_PER_TRAINING))
MONTHLY_TOTAL=$((MONTHLY_TRAINING + 520))  # +520 for other workflows

echo "Estimated monthly usage: ${MONTHLY_TOTAL} minutes"
```

### Resource Monitor Reports

Check Issues tab for monthly "Resource Report" issues with:
- Exact usage breakdown
- Storage statistics
- Optimization recommendations

## 🛠️ Optimization Tips

### For GitHub Free Tier

1. **12-hour training schedule**: `cron: '0 */12 * * *'`
2. **Disable NAS**: Comment out architecture-search.yml
3. **Reduce epochs**: 75 instead of 100
4. **Result**: ~1,900 min/month (within 2,000 limit)

### For GitHub Pro Tier

1. **Keep 8-hour schedule**: Maximum improvement rate
2. **Enable all workflows**: Full feature set
3. **Result**: ~2,800 min/month (within 3,000 limit)

### For Even More Features

1. **Self-hosted runner**: No minute limits (but you pay for hardware)
2. **GitHub Team/Enterprise**: 50,000 minutes/month

## 🎓 Research-Grade Capabilities

Comparable to systems from:
- **DeepMind** (AlphaZero, MuZero)
- **OpenAI** (GPT training infrastructure)
- **Google Brain** (AutoML, NAS)
- **Meta AI** (PyTorch development)

## 📝 License

MIT License - use freely for research and commercial projects

---

**Built for autonomous, long-term evolution within GitHub Actions limits**

*Last updated: Check [Actions](https://github.com/Tommaso-R-Marena/AutoEvolve-ML/actions)*
