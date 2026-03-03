"""Interactive Visualization Dashboard

Creates comprehensive visualizations for:
1. Training dynamics and loss landscapes
2. Quantum optimization trajectories
3. Causal graphs with strength indicators
4. Nobel research progress
5. Meta-learning insights
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class VisualizationDashboard:
    """Create comprehensive research visualizations"""
    
    def __init__(self):
        self.figures = []
        
    def plot_training_dynamics(self, metrics: Dict, save_path: str = 'training_dynamics.png'):
        """Plot training and validation loss over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Training Dynamics Dashboard', fontsize=16, fontweight='bold')
        
        # Loss curves
        ax = axes[0, 0]
        if 'loss' in metrics and 'val_loss' in metrics:
            epochs = range(len(metrics['loss']))
            ax.plot(epochs, metrics['loss'], label='Training Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, metrics['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax = axes[0, 1]
        if 'learning_rate' in metrics:
            ax.plot(metrics['learning_rate'], linewidth=2, color='purple', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Gradient norms
        ax = axes[1, 0]
        if 'grad_norm' in metrics:
            ax.plot(metrics['grad_norm'], linewidth=2, color='red', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Magnitudes')
            ax.grid(True, alpha=0.3)
        
        # Performance improvement
        ax = axes[1, 1]
        if 'val_loss' in metrics and len(metrics['val_loss']) > 1:
            improvements = []
            for i in range(1, len(metrics['val_loss'])):
                prev = metrics['val_loss'][i-1]
                curr = metrics['val_loss'][i]
                if prev > 0:
                    improvement = (prev - curr) / prev * 100
                    improvements.append(improvement)
            
            if improvements:
                ax.bar(range(len(improvements)), improvements, alpha=0.7, color='green')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.axhline(y=3, color='red', linestyle='--', linewidth=1, label='Auto-commit threshold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Improvement (%)')
                ax.set_title('Per-Epoch Improvement')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training dynamics saved to {save_path}")
        self.figures.append(fig)
        
    def plot_quantum_optimization(self, state_file: str = 'quantum_optimizer_state.json',
                                 save_path: str = 'quantum_optimization.png'):
        """Visualize quantum optimization trajectory"""
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('⚡ Quantum-Inspired Optimization', fontsize=16, fontweight='bold')
            
            # Energy landscape
            ax = axes[0]
            if 'energy_history' in state:
                energy = state['energy_history']
                iterations = range(len(energy))
                
                ax.plot(iterations, energy, linewidth=2, alpha=0.8, color='blue')
                ax.axhline(y=state['best_energy'], color='red', linestyle='--', 
                          linewidth=2, label=f"Best: {state['best_energy']:.4f}")
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Energy (Loss)')
                ax.set_title('Quantum Annealing Trajectory')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Annotate quantum tunneling events (energy increases)
                tunneling_events = []
                for i in range(1, len(energy)):
                    if energy[i] > energy[i-1]:
                        tunneling_events.append(i)
                
                if tunneling_events:
                    ax.scatter(tunneling_events, [energy[i] for i in tunneling_events],
                             color='orange', s=100, marker='*', zorder=5, 
                             label=f'Tunneling ({len(tunneling_events)} events)')
                    ax.legend()
            
            # Temperature schedule
            ax = axes[1]
            # Recreate temperature schedule
            n_steps = len(state.get('energy_history', []))
            if n_steps > 0:
                T_start = 10.0
                T_end = 0.01
                temperatures = [T_start * (T_end / T_start) ** (i / n_steps) 
                              for i in range(n_steps)]
                
                ax.plot(temperatures, linewidth=2, color='red', alpha=0.8)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Temperature')
                ax.set_title('Annealing Schedule (Cooling)')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Quantum optimization plot saved to {save_path}")
            self.figures.append(fig)
            
        except Exception as e:
            print(f"Could not plot quantum optimization: {e}")
    
    def plot_causal_graph(self, graph_file: str = 'causal_graph.json',
                         save_path: str = 'causal_graph.png'):
        """Visualize discovered causal relationships"""
        try:
            with open(graph_file, 'r') as f:
                data = json.load(f)
            
            graph = data.get('graph', {})
            edges = graph.get('edges', [])
            
            if not edges:
                print("No causal edges to plot")
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.suptitle('🧠 Causal Discovery Graph', fontsize=16, fontweight='bold')
            
            # Extract nodes
            nodes = set()
            for parent, child, weight in edges:
                nodes.add(parent)
                nodes.add(child)
            
            nodes = list(nodes)
            n_nodes = len(nodes)
            
            # Position nodes in circle
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            positions = {node: (np.cos(angle), np.sin(angle)) 
                        for node, angle in zip(nodes, angles)}
            
            # Draw edges with thickness proportional to causal strength
            for parent, child, weight in edges:
                x1, y1 = positions[parent]
                x2, y2 = positions[child]
                
                # Arrow from parent to child
                dx = x2 - x1
                dy = y2 - y1
                
                # Color by strength
                color = plt.cm.RdYlGn(weight)
                linewidth = 1 + 5 * weight  # Scale with strength
                
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', lw=linewidth, 
                                         color=color, alpha=0.7))
                
                # Add weight label
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Draw nodes
            for node, (x, y) in positions.items():
                ax.scatter(x, y, s=2000, c='lightblue', edgecolors='black', 
                          linewidth=2, zorder=10)
                ax.text(x, y, node, ha='center', va='center', fontsize=10,
                       fontweight='bold')
            
            # Styling
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Edge width and color indicate causal strength', 
                        fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Causal graph saved to {save_path}")
            self.figures.append(fig)
            
        except Exception as e:
            print(f"Could not plot causal graph: {e}")
    
    def plot_nobel_progress(self, state_file: str = 'nobel_research_state.json',
                           save_path: str = 'nobel_progress.png'):
        """Visualize Nobel Prize potential and research progress"""
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            assessment = state.get('nobel_assessment', {})
            criteria = assessment.get('criteria', {})
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('🌟 Nobel Prize Research Progress', fontsize=16, fontweight='bold')
            
            # Overall potential gauge
            ax = axes[0, 0]
            potential = assessment.get('nobel_potential', 0)
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            r = 1
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Background arc
            ax.plot(x, y, linewidth=10, color='lightgray', alpha=0.3)
            
            # Progress arc
            progress_theta = np.linspace(0, np.pi * potential, 100)
            progress_x = r * np.cos(progress_theta)
            progress_y = r * np.sin(progress_theta)
            
            color = 'red' if potential < 0.3 else 'yellow' if potential < 0.7 else 'green'
            ax.plot(progress_x, progress_y, linewidth=10, color=color, alpha=0.8)
            
            # Needle
            needle_angle = np.pi * (1 - potential)
            needle_x = [0, r * 0.8 * np.cos(needle_angle)]
            needle_y = [0, r * 0.8 * np.sin(needle_angle)]
            ax.plot(needle_x, needle_y, linewidth=3, color='black')
            
            # Text
            ax.text(0, -0.3, f"{potential:.1%}", ha='center', va='center',
                   fontsize=24, fontweight='bold')
            ax.text(0, -0.5, 'Nobel Potential', ha='center', va='center',
                   fontsize=12)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.7, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Overall Assessment')
            
            # Criteria radar chart
            ax = axes[0, 1]
            if criteria:
                categories = list(criteria.keys())
                values = list(criteria.values())
                
                # Radar chart
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
                values_plot = values + [values[0]]  # Complete the circle
                angles_plot = np.concatenate([angles, [angles[0]]])
                
                ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='blue', alpha=0.8)
                ax.fill(angles_plot, values_plot, alpha=0.25, color='blue')
                ax.set_xticks(angles)
                ax.set_xticklabels([c.replace('_', '\n').title() for c in categories], 
                                  fontsize=8)
                ax.set_ylim(0, 1)
                ax.set_title('Criteria Breakdown')
                ax.grid(True)
            
            # Discovery timeline
            ax = axes[1, 0]
            discoveries = state.get('discoveries', [])
            
            if discoveries:
                types = [d['type'] for d in discoveries]
                type_counts = {t: types.count(t) for t in set(types)}
                
                colors_map = {'breakthrough': 'gold', 'major': 'green', 
                             'minor': 'blue', 'null': 'gray'}
                colors = [colors_map.get(t, 'gray') for t in type_counts.keys()]
                
                ax.bar(type_counts.keys(), type_counts.values(), color=colors, alpha=0.7)
                ax.set_xlabel('Discovery Type')
                ax.set_ylabel('Count')
                ax.set_title(f'Discoveries ({len(discoveries)} total)')
                ax.grid(True, alpha=0.3, axis='y')
            
            # Hypothesis tracking
            ax = axes[1, 1]
            hypotheses = state.get('hypotheses', [])
            experiments = state.get('experiments', [])
            
            stages = ['Hypotheses', 'Experiments', 'Discoveries']
            counts = [len(hypotheses), len(experiments), len(discoveries)]
            colors = ['lightblue', 'orange', 'green']
            
            ax.bar(stages, counts, color=colors, alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title('Research Pipeline')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add text on bars
            for i, (stage, count) in enumerate(zip(stages, counts)):
                ax.text(i, count + 0.5, str(count), ha='center', 
                       fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Nobel progress plot saved to {save_path}")
            self.figures.append(fig)
            
        except Exception as e:
            print(f"Could not plot Nobel progress: {e}")
    
    def generate_comprehensive_report(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATION DASHBOARD")
        print("="*60 + "\n")
        
        # Load and plot training dynamics
        try:
            with open('metrics.json', 'r') as f:
                metrics = json.load(f)
            self.plot_training_dynamics(metrics)
        except Exception as e:
            print(f"Could not plot training dynamics: {e}")
        
        # Plot quantum optimization
        self.plot_quantum_optimization()
        
        # Plot causal graph
        self.plot_causal_graph()
        
        # Plot Nobel progress
        self.plot_nobel_progress()
        
        print("\n" + "="*60)
        print("VISUALIZATION DASHBOARD COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  - training_dynamics.png")
        print("  - quantum_optimization.png")
        print("  - causal_graph.png")
        print("  - nobel_progress.png")
        print("\n🎉 Dashboard ready for analysis!")


if __name__ == "__main__":
    dashboard = VisualizationDashboard()
    dashboard.generate_comprehensive_report()
