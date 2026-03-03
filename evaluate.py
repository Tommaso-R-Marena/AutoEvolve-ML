import json
import torch
import numpy as np
from datetime import datetime
from train import SelfImprovingModel, AdvancedDataGenerator

def generate_report():
    """Generate comprehensive evaluation report"""
    report = []
    report.append("# Model Evaluation Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
    
    # Load metrics
    try:
        with open('metrics.json', 'r') as f:
            history = json.load(f)
        
        if not history.get('loss') or len(history['loss']) == 0:
            report.append("⚠️ No training data available yet. Run training first.\n")
            report_text = ''.join(report)
            with open('evaluation_report.md', 'w') as f:
                f.write(report_text)
            print(report_text)
            return report_text
        
        report.append("## Training Progress\n")
        report.append(f"- **Total training cycles:** {len(history['loss'])}\n")
        report.append(f"- **Current complexity level:** {history['complexity'][-1] if history.get('complexity') and len(history['complexity']) > 0 else 1}\n")
        report.append(f"- **Latest training loss:** {history['loss'][-1]:.4f}\n")
        report.append(f"- **Latest validation loss:** {history['val_loss'][-1]:.4f}\n")
        report.append(f"- **Best validation loss:** {min(history['val_loss']):.4f}\n\n")
        
        # Calculate improvement
        if len(history['loss']) > 10:
            initial_loss = np.mean(history['loss'][:5])
            recent_loss = np.mean(history['loss'][-5:])
            improvement = ((initial_loss - recent_loss) / initial_loss) * 100
            report.append(f"## Performance Improvement\n")
            report.append(f"- **Overall improvement:** {improvement:.2f}%\n")
            report.append(f"- **Initial avg loss:** {initial_loss:.4f}\n")
            report.append(f"- **Recent avg loss:** {recent_loss:.4f}\n\n")
        
        # Load model architecture
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load('model_checkpoint.pth')
            else:
                checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
            
            hidden_sizes = checkpoint.get('hidden_sizes', [64, 128, 64])
            report.append("## Model Architecture\n")
            report.append(f"- **Hidden layers:** {len(hidden_sizes)}\n")
            report.append(f"- **Layer sizes:** {hidden_sizes}\n")
            
            # Calculate total parameters
            model = SelfImprovingModel(hidden_sizes=hidden_sizes)
            model.load_state_dict(checkpoint['model_state_dict'])
            total_params = sum(p.numel() for p in model.parameters())
            report.append(f"- **Total parameters:** {total_params:,}\n\n")
        except FileNotFoundError:
            report.append("## Model Architecture\n")
            report.append("⚠️ Model checkpoint not found yet.\n\n")
        
        # Database statistics
        try:
            with open('data/database_metadata.json', 'r') as f:
                db_meta = json.load(f)
            report.append("## Training Database\n")
            report.append(f"- **Total samples:** {db_meta.get('total_samples', 0):,}\n")
            report.append(f"- **Total chunks:** {db_meta.get('total_chunks', 0)}\n")
            report.append(f"- **Database size:** {db_meta.get('total_chunks', 0) * 95:.0f} MB (est.)\n\n")
        except:
            pass
        
        # Meta-learning statistics
        try:
            with open('meta_learning_state.json', 'r') as f:
                meta_state = json.load(f)
            if meta_state.get('learning_history'):
                report.append("## Meta-Learning\n")
                report.append(f"- **Learning episodes recorded:** {len(meta_state['learning_history'])}\n")
                report.append(f"- **Strategies evaluated:** {len(meta_state.get('strategy_performance', {}))}\n\n")
        except:
            pass
        
        # Architecture search statistics
        try:
            with open('nas_state.json', 'r') as f:
                nas_state = json.load(f)
            if nas_state.get('search_history'):
                report.append("## Neural Architecture Search\n")
                report.append(f"- **Architectures evaluated:** {len(nas_state['search_history'])}\n\n")
        except:
            pass
        
        # Research automation statistics
        try:
            with open('research_state.json', 'r') as f:
                research_state = json.load(f)
            if research_state.get('experiments'):
                completed = len([e for e in research_state['experiments'] if e['status'] == 'completed'])
                breakthroughs = len(research_state.get('breakthroughs', []))
                report.append("## Research Automation\n")
                report.append(f"- **Total experiments:** {len(research_state['experiments'])}\n")
                report.append(f"- **Completed experiments:** {completed}\n")
                report.append(f"- **Breakthroughs discovered:** {breakthroughs}\n\n")
        except:
            pass
        
        report.append("## System Status\n")
        report.append("✅ **Active:** Model continues to train automatically every 8 hours\n")
        report.append("✅ **Meta-Learning:** Learning optimal strategies from training history\n")
        report.append("✅ **Architecture Search:** Weekly evaluation of new architectures\n")
        report.append("✅ **Research Automation:** Generating and testing hypotheses\n\n")
        
        report.append("## Next Steps\n")
        report.append("- Architecture evolves when performance plateaus\n")
        report.append("- Data complexity increases every 10 cycles\n")
        report.append("- Best models added to ensemble\n")
        report.append("- Breakthroughs trigger immediate commits\n")
        
    except Exception as e:
        report.append(f"\n⚠️ Error generating report: {str(e)}\n")
        import traceback
        report.append(f"\n```\n{traceback.format_exc()}\n```\n")
    
    report_text = ''.join(report)
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

if __name__ == "__main__":
    generate_report()
