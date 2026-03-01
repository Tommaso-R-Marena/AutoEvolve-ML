import json
import torch
import numpy as np
from datetime import datetime
from train import SelfImprovingModel, DataGenerator

def generate_report():
    """Generate comprehensive evaluation report"""
    report = []
    report.append("# Model Evaluation Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
    
    # Load metrics
    try:
        with open('metrics.json', 'r') as f:
            history = json.load(f)
        
        report.append("## Training Progress\n")
        report.append(f"- **Total training cycles:** {len(history['loss'])}\n")
        report.append(f"- **Current complexity level:** {history['complexity'][-1] if history['complexity'] else 1}\n")
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
        if torch.cuda.is_available():
            checkpoint = torch.load('model_checkpoint.pth')
        else:
            checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
        
        hidden_sizes = checkpoint.get('hidden_sizes', [])
        report.append("## Model Architecture\n")
        report.append(f"- **Hidden layers:** {len(hidden_sizes)}\n")
        report.append(f"- **Layer sizes:** {hidden_sizes}\n")
        
        # Calculate total parameters
        model = SelfImprovingModel(hidden_sizes=hidden_sizes)
        model.load_state_dict(checkpoint['model_state_dict'])
        total_params = sum(p.numel() for p in model.parameters())
        report.append(f"- **Total parameters:** {total_params:,}\n\n")
        
        report.append("## Next Steps\n")
        report.append("- Model continues to train automatically every 6 hours\n")
        report.append("- Architecture evolves when performance plateaus\n")
        report.append("- Synthetic data complexity increases progressively\n")
        
    except Exception as e:
        report.append(f"\n⚠️ Error generating report: {str(e)}\n")
    
    report_text = ''.join(report)
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

if __name__ == "__main__":
    generate_report()
