"""
Standalone visualization for Port Congestion Simulation with DeepSeek-V3
Run this after the simulation to generate comprehensive plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SimulationVisualizer:
    """Visualize simulation results"""
    
    def __init__(self, results_file=None):
        self.results = None
        self.analysis = None
        self.config = None
        
        if results_file and Path(results_file).exists():
            self.load_from_file(results_file)
        else:
            self.load_latest_results()
    
    def load_latest_results(self):
        """Load most recent simulation results"""
        results_dir = Path("simulation_results")
        if not results_dir.exists():
            print("❌ No simulation_results directory found")
            return
        
        json_files = list(results_dir.glob("simulation_results_*.json"))
        if not json_files:
            print("❌ No results files found")
            return
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        self.load_from_file(latest_file)
    
    def load_from_file(self, filepath):
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = data.get('config', {})
        self.results = data
        self.analysis = data.get('analysis', {})
        
        # Convert to numpy arrays
        self.weekly_costs = np.array(self.results['weekly_costs'])
        self.weekly_day_counts = np.array(self.results['weekly_day_counts'])
        self.agent_costs = [np.array(costs) for costs in self.results['agent_costs']]
        self.agent_actions = [np.array(actions) for actions in self.results['agent_actions']]
        
        print(f"✅ Loaded results from {filepath}")
        print(f"   Weeks: {len(self.weekly_costs)}")
        print(f"   Agents: {len(self.agent_costs)}")
        print(f"   RL Agents: {self.config.get('n_rl_agents', 9)}")
        print(f"   DeepSeek Agents: {self.config.get('n_llm_agents', 2)}")
    
    def create_all_visualizations(self, save_dir="simulation_results"):
        """Create all visualizations"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Create all plots
        self.plot_cost_evolution(save_path)
        self.plot_agent_comparison(save_path)
        self.plot_congestion_heatmap(save_path)
        self.plot_day_preferences(save_path)
        self.plot_cost_distribution(save_path)
        self.create_summary_report(save_path)
        
        print(f"\n✅ All visualizations saved to {save_path}/")
    
    def plot_cost_evolution(self, save_path):
        """Plot cost evolution over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall cost trend
        ax = axes[0, 0]
        weeks = range(1, len(self.weekly_costs) + 1)
        window = 8
        rolling_avg = pd.Series(self.weekly_costs).rolling(window).mean()
        
        ax.plot(weeks, self.weekly_costs, 'o-', alpha=0.3, markersize=3, label='Weekly Cost')
        ax.plot(weeks, rolling_avg, 'r-', linewidth=2, label=f'{window}-Week Average')
        ax.set_xlabel('Week')
        ax.set_ylabel('Average Cost ($)')
        ax.set_title('System Cost Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. RL vs DeepSeek comparison
        ax = axes[0, 1]
        n_rl = self.config.get('n_rl_agents', 9)
        
        rl_weekly = np.mean([self.agent_costs[i] for i in range(n_rl)], axis=0)
        ds_weekly = np.mean([self.agent_costs[i] for i in range(n_rl, len(self.agent_costs))], axis=0)
        
        rl_smooth = pd.Series(rl_weekly).rolling(window).mean()
        ds_smooth = pd.Series(ds_weekly).rolling(window).mean()
        
        ax.plot(weeks, rl_smooth, 'b-', linewidth=2, label='Q-Learning')
        ax.plot(weeks, ds_smooth, 'r-', linewidth=2, label='DeepSeek-V3')
        ax.fill_between(weeks, rl_smooth, ds_smooth, alpha=0.2, color='gray')
        ax.set_xlabel('Week')
        ax.set_ylabel('Average Cost ($)')
        ax.set_title('RL vs DeepSeek: Cost Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative cost
        ax = axes[1, 0]
        rl_cumulative = np.cumsum(rl_weekly)
        ds_cumulative = np.cumsum(ds_weekly)
        
        ax.fill_between(weeks, rl_cumulative, alpha=0.3, color='blue', label='Q-Learning')
        ax.fill_between(weeks, ds_cumulative, alpha=0.3, color='red', label='DeepSeek-V3')
        ax.plot(weeks, rl_cumulative, 'b-', linewidth=2)
        ax.plot(weeks, ds_cumulative, 'r-', linewidth=2)
        ax.set_xlabel('Week')
        ax.set_ylabel('Cumulative Cost ($)')
        ax.set_title('Cumulative Cost Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Improvement percentage
        ax = axes[1, 1]
        improvement = (rl_weekly - ds_weekly) / rl_weekly * 100
        improvement_smooth = pd.Series(improvement).rolling(window).mean()
        
        colors = ['green' if x > 0 else 'red' for x in improvement]
        ax.bar(weeks, improvement, alpha=0.3, color=colors)
        ax.plot(weeks, improvement_smooth, 'k-', linewidth=2, label='Trend')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Week')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('DeepSeek Improvement Over RL (Positive = Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'cost_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: cost_evolution.png")
    
    def plot_agent_comparison(self, save_path):
        """Compare individual agent performance"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Bar chart
        ax = axes[0]
        agent_perf = self.analysis['agent_performance']
        agent_labels = [f"{p['type']}_{p['id']}" for p in agent_perf]
        total_costs = [p['total_cost'] for p in agent_perf]
        colors = ['#3498db' if p['type'] == 'RL' else '#e74c3c' for p in agent_perf]
        
        bars = ax.bar(range(len(agent_labels)), total_costs, color=colors, alpha=0.7)
        ax.set_xticks(range(len(agent_labels)))
        ax.set_xticklabels(agent_labels, rotation=45, ha='right')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Individual Agent Performance')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, cost in zip(bars, total_costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost/1000:.0f}k', ha='center', va='bottom', fontsize=8)
        
        # 2. Box plot
        ax = axes[1]
        n_rl = self.config.get('n_rl_agents', 9)
        
        rl_costs = [cost for i in range(n_rl) for cost in self.agent_costs[i]]
        ds_costs = [cost for i in range(n_rl, len(self.agent_costs)) for cost in self.agent_costs[i]]
        
        bp = ax.boxplot([rl_costs, ds_costs], 
                        labels=['Q-Learning Agents', 'DeepSeek-V3 Agents'],
                        patch_artist=True)
        
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Cost per Week ($)')
        ax.set_title('Cost Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'agent_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: agent_comparison.png")
    
    def plot_congestion_heatmap(self, save_path):
        """Plot congestion heatmap over time"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Congestion heatmap
        ax = axes[0]
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Create heatmap data
        heatmap_data = self.weekly_day_counts.T
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Week')
        ax.set_ylabel('Day of Week')
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels(days)
        ax.set_title('Congestion Heatmap (Darker = More Congested)')
        plt.colorbar(im, ax=ax, label='Vessels per Day')
        
        # 2. Average congestion by day
        ax = axes[1]
        avg_congestion = np.mean(self.weekly_day_counts, axis=0)
        
        bars = ax.bar(days, avg_congestion, color=plt.cm.YlOrRd(avg_congestion / max(avg_congestion)))
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Vessels per Day')
        ax.set_title('Average Congestion by Day of Week')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, avg_congestion):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path / 'congestion_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: congestion_heatmap.png")
    
    def plot_day_preferences(self, save_path):
        """Plot day preference patterns"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        n_rl = self.config.get('n_rl_agents', 9)
        
        # Calculate preferences
        rl_counts = np.zeros(7)
        ds_counts = np.zeros(7)
        
        for i in range(n_rl):
            rl_counts += np.bincount(self.agent_actions[i], minlength=7)
        
        for i in range(n_rl, len(self.agent_actions)):
            ds_counts += np.bincount(self.agent_actions[i], minlength=7)
        
        rl_probs = rl_counts / rl_counts.sum()
        ds_probs = ds_counts / ds_counts.sum()
        
        # Plot
        x = np.arange(len(days))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rl_probs, width, label='Q-Learning', color='#3498db', alpha=0.7)
        bars2 = ax.bar(x + width/2, ds_probs, width, label='DeepSeek-V3', color='#e74c3c', alpha=0.7)
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Selection Probability')
        ax.set_title('Day Selection Preferences: RL vs DeepSeek')
        ax.set_xticks(x)
        ax.set_xticklabels(days, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path / 'day_preferences.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: day_preferences.png")
    
    def plot_cost_distribution(self, save_path):
        """Plot cost distribution statistics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_rl = self.config.get('n_rl_agents', 9)
        
        # Calculate statistics
        rl_costs = [np.sum(self.agent_costs[i]) for i in range(n_rl)]
        ds_costs = [np.sum(self.agent_costs[i]) for i in range(n_rl, len(self.agent_costs))]
        
        # Create DataFrame for boxplot
        data = pd.DataFrame({
            'Agent Type': ['Q-Learning'] * len(rl_costs) + ['DeepSeek-V3'] * len(ds_costs),
            'Total Cost': rl_costs + ds_costs
        })
        
        # Boxplot
        bp = data.boxplot(by='Agent Type', column='Total Cost', ax=ax, grid=True)
        ax.set_title('Total Cost Distribution by Agent Type')
        ax.set_ylabel('Total Cost ($)')
        ax.set_xlabel('')
        plt.suptitle('')  # Remove automatic title
        
        # Add swarm plot for individual points
        for i, agent_type in enumerate(['Q-Learning', 'DeepSeek-V3']):
            costs = data[data['Agent Type'] == agent_type]['Total Cost']
            x_jitter = np.random.normal(i, 0.04, len(costs))
            ax.scatter(x_jitter, costs, alpha=0.6, s=30, c='black', zorder=3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'cost_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: cost_distribution.png")
    
    def create_summary_report(self, save_path):
        """Create a text summary report"""
        n_rl = self.config.get('n_rl_agents', 9)
        
        # Calculate metrics
        rl_total = np.sum([np.sum(self.agent_costs[i]) for i in range(n_rl)])
        ds_total = np.sum([np.sum(self.agent_costs[i]) for i in range(n_rl, len(self.agent_costs))])
        
        rl_avg_per_agent = rl_total / n_rl
        ds_avg_per_agent = ds_total / (len(self.agent_costs) - n_rl)
        
        improvement = ((rl_avg_per_agent - ds_avg_per_agent) / rl_avg_per_agent) * 100
        
        # Create report
        report = f"""
{'='*60}
PORT CONGESTION SIMULATION REPORT
DeepSeek-V3 vs Q-Learning
{'='*60}

SIMULATION CONFIGURATION
{'-'*60}
Total Weeks: {self.config.get('n_weeks', 52)}
Q-Learning Agents: {n_rl}
DeepSeek-V3 Agents: {len(self.agent_costs) - n_rl}
DeepSeek Model: {self.config.get('deepseek_model', 'deepseek-chat')}
Timestamp: {self.config.get('timestamp', 'N/A')}

PERFORMANCE METRICS
{'-'*60}
Total System Cost: ${rl_total + ds_total:,.2f}
Average Weekly Cost: ${np.mean(self.weekly_costs):,.2f}

Q-LEARNING AGENTS (n={n_rl})
{'-'*60}
Total Cost (All Agents): ${rl_total:,.2f}
Average per Agent: ${rl_avg_per_agent:,.2f}
Best Agent: ${min([np.sum(self.agent_costs[i]) for i in range(n_rl)]):,.2f}
Worst Agent: ${max([np.sum(self.agent_costs[i]) for i in range(n_rl)]):,.2f}
Std Dev: ${np.std([np.sum(self.agent_costs[i]) for i in range(n_rl)]):,.2f}

DEEPSEEK-V3 AGENTS (n={len(self.agent_costs) - n_rl})
{'-'*60}
Total Cost (All Agents): ${ds_total:,.2f}
Average per Agent: ${ds_avg_per_agent:,.2f}
Best Agent: ${min([np.sum(self.agent_costs[i]) for i in range(n_rl, len(self.agent_costs))]):,.2f}
Worst Agent: ${max([np.sum(self.agent_costs[i]) for i in range(n_rl, len(self.agent_costs))]):,.2f}
Std Dev: ${np.std([np.sum(self.agent_costs[i]) for i in range(n_rl, len(self.agent_costs))]):,.2f}

COMPARISON
{'-'*60}
DeepSeek Performance vs Q-Learning: {improvement:+.1f}%
{'✅ DeepSeek performed BETTER' if improvement > 0 else '❌ Q-Learning performed BETTER' if improvement < 0 else '🤝 Similar performance'}

CONGESTION PATTERNS
{'-'*60}
Most Congested Day: {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][np.argmax(np.mean(self.weekly_day_counts, axis=0))]}
Least Congested Day: {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][np.argmin(np.mean(self.weekly_day_counts, axis=0))]}
Average Congestion: {np.mean(self.weekly_day_counts):.1f} vessels/day

{'='*60}
End of Report
{'='*60}
"""
        
        # Save report
        report_file = save_path / 'simulation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"  ✓ Saved: simulation_report.txt")
        print(report)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to generate visualizations"""
    print("\n" + "="*60)
    print("PORT CONGESTION SIMULATION VISUALIZER")
    print("="*60)
    
    # Create visualizer
    visualizer = SimulationVisualizer()
    
    if visualizer.results is None:
        print("\n❌ No simulation results found!")
        print("\nPlease run the simulation first:")
        print("  python deepseek_simulation.py")
        return
    
    # Generate all visualizations
    visualizer.create_all_visualizations()
    
    print("\n✅ Visualization complete!")
    print("\nYou can now view the generated plots in the 'simulation_results' folder.")

if __name__ == "__main__":
    main()