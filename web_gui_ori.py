"""
Marine Port Optimization - Web GUI with RL Agents and Live Plots
Includes comprehensive visualization capabilities
"""

from flask import Flask, render_template_string, jsonify, request, send_file
import numpy as np
import pandas as pd
import random
import time
import threading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web serving
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

app = Flask(__name__)
app.config['DEBUG'] = False

# Global variables to store results for plotting
simulation_results = None

# ============================================================================
# RL AGENT IMPLEMENTATION
# ============================================================================

class RLAgent:
    """RL Agent that learns optimal weekly attendance patterns"""
    
    def __init__(self, agent_id, alpha=0.1, gamma=0.9, epsilon=0.1, strategy_mutation_rate=0.5):
        self.agent_id = agent_id
        self.agent_name = f"RLAgent_{agent_id}"
        
        # RL Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.strategy_mutation_rate = strategy_mutation_rate
        
        # Initialize with predefined strategies (7-day binary vectors)
        self.strategies = self._initialize_strategies()
        self.Q = np.ones(len(self.strategies)) * 0.5
        
        self.current_week = -1
        self.weekly_strategy = None
        self.weekly_strategy_idx = None
        self.reward_history = []
        self.strategy_history = []
        self.created_strategies = []
        
        # Tracking metrics
        self.total_success = 0
        self.total_attempts = 0
        self.mutation_attempts = 0
        self.mutation_successes = 0
        
    def _initialize_strategies(self):
        strategies = [
            [1,0,1,0,1,0,0],  # Mon, Wed, Fri
            [0,1,0,1,0,1,0],  # Tue, Thu, Sat
            [1,0,0,1,0,0,1],  # Mon, Thu, Sun
            [0,0,1,0,1,0,1],  # Wed, Fri, Sun
            [1,1,0,0,0,1,0]   # Mon, Tue, Sat
        ]
        
        for _ in range(2):
            strategies.append(self._generate_random_strategy())
        
        return strategies
    
    def _generate_random_strategy(self):
        num_days = random.choice([3, 4])
        strategy = [0] * 7
        days = random.sample(range(7), num_days)
        for day in days:
            strategy[day] = 1
        return strategy
    
    def _select_weekly_strategy(self, week_num):
        if len(self.strategies) == 0:
            self.weekly_strategy = [0] * 7
            self.weekly_strategy_idx = 0
            return
            
        if random.random() < self.epsilon:
            self.weekly_strategy_idx = random.randint(0, len(self.strategies)-1)
        else:
            if len(self.Q) > 0:
                self.weekly_strategy_idx = np.argmax(self.Q)
            else:
                self.weekly_strategy_idx = random.randint(0, len(self.strategies)-1)
        
        self.weekly_strategy = self.strategies[self.weekly_strategy_idx].copy()
        self.strategy_history.append(self.weekly_strategy_idx)
        self.current_week = week_num
    
    def decide(self, day):
        week_num = day // 7
        day_of_week = day % 7
        
        if week_num != self.current_week:
            self._select_weekly_strategy(week_num)
        
        if self.weekly_strategy is not None:
            return self.weekly_strategy[day_of_week]
        return 0
    
    def observe_reward(self, reward):
        self.reward_history.append(reward)
        
        if reward > 0:
            self.total_success += 1
        self.total_attempts += 1
        
        if self.weekly_strategy_idx is not None and len(self.Q) > 0:
            if self.weekly_strategy_idx < len(self.Q):
                future = np.max(self.Q) if len(self.Q) > 0 else 0
                td_error = reward + self.gamma * future - self.Q[self.weekly_strategy_idx]
                self.Q[self.weekly_strategy_idx] += self.alpha * td_error
                self.Q = np.clip(self.Q, -10, 10)
        
        if len(self.reward_history) % 7 == 0 and len(self.reward_history) > 7:
            self._try_mutation()
    
    def _try_mutation(self):
        self.mutation_attempts += 1
        
        if random.random() < self.strategy_mutation_rate and len(self.strategies) > 0:
            if self._create_new_strategy():
                self.mutation_successes += 1
    
    def _create_new_strategy(self):
        if len(self.strategies) == 0:
            return False
        
        parent_idx = np.argmax(self.Q)
        if parent_idx >= len(self.strategies):
            parent_idx = 0
            
        parent_strategy = self.strategies[parent_idx].copy()
        
        new_strategy = []
        for decision in parent_strategy:
            if random.random() < 0.3:
                new_strategy.append(1 - decision)
            else:
                new_strategy.append(decision)
        
        days_count = sum(new_strategy)
        if days_count < 3:
            zero_days = [i for i, d in enumerate(new_strategy) if d == 0]
            to_add = 3 - days_count
            add_days = random.sample(zero_days, min(to_add, len(zero_days)))
            for day in add_days:
                new_strategy[day] = 1
        elif days_count > 4:
            one_days = [i for i, d in enumerate(new_strategy) if d == 1]
            to_remove = days_count - 4
            remove_days = random.sample(one_days, min(to_remove, len(one_days)))
            for day in remove_days:
                new_strategy[day] = 0
        
        for existing_strat in self.strategies:
            if new_strategy == existing_strat:
                return False
        
        self.strategies.append(new_strategy)
        initial_q = self.Q[parent_idx] * 0.9
        initial_q = np.clip(initial_q, 0.1, 2.0)
        self.Q = np.append(self.Q, initial_q)
        
        return True
    
    def success_rate(self):
        if self.total_attempts == 0:
            return 0
        return self.total_success / self.total_attempts

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.current_day = 0
        self.ships = []
        self.daily_arrivals = None
        self.daily_success = None
        self.config = None
        self.auto_step_thread = None
        self.attendance_history = []
        self.daily_rewards = {'rl': [], 'llm': []}
        
    def initialize(self, config):
        self.config = config
        self.current_day = 0
        self.ships = []
        self.is_running = True
        self.is_paused = False
        self.attendance_history = []
        self.daily_rewards = {'rl': [], 'llm': []}
        
        num_adaptive = min(config.get('num_adaptive_ships', config['total_ships']), config['total_ships'])
        
        for i in range(config['total_ships']):
            if i < num_adaptive:
                ship = RLAgent(agent_id=i, alpha=0.15, gamma=0.9, epsilon=0.15, strategy_mutation_rate=0.6)
            else:
                ship = RLAgent(agent_id=i, alpha=0.08, gamma=0.9, epsilon=0.05, strategy_mutation_rate=0.4)
            self.ships.append(ship)
        
        self.daily_arrivals = np.zeros(config['total_days'], dtype=int)
        self.daily_success = np.zeros(config['total_days'], dtype=bool)
        
        return True
    
    def run_day(self):
        if self.current_day >= self.config['total_days']:
            self.is_running = False
            return False
        
        decisions = [ship.decide(self.current_day) for ship in self.ships]
        num_arrivals = sum(decisions)
        self.daily_arrivals[self.current_day] = num_arrivals
        self.attendance_history.append(num_arrivals)
        
        capacity_limit = self.config['total_ships'] * (self.config.get('capacity', 60) / 100)
        is_congested = num_arrivals > capacity_limit
        
        # Track daily rewards by agent type
        rl_rewards = []
        llm_rewards = []
        
        for ship, decision in zip(self.ships, decisions):
            if decision == 1:
                if not is_congested:
                    reward = 3
                    ship.observe_reward(3)
                else:
                    reward = -2
                    ship.observe_reward(-2)
            else:
                if is_congested:
                    reward = 1
                    ship.observe_reward(1)
                else:
                    reward = 0
                    ship.observe_reward(0)
            
            # Track by type (all are RL agents in this version)
            if ship.epsilon > 0.1:
                rl_rewards.append(reward)
            else:
                llm_rewards.append(reward)
        
        # Store average daily rewards
        self.daily_rewards['rl'].append(np.mean(rl_rewards) if rl_rewards else 0)
        self.daily_rewards['llm'].append(np.mean(llm_rewards) if llm_rewards else 0)
        
        self.daily_success[self.current_day] = (num_arrivals == 1)
        self.current_day += 1
        return True
    
    def auto_step(self):
        while self.is_running:
            if not self.is_paused and self.current_day < self.config['total_days']:
                self.run_day()
                time.sleep(self.config.get('delay', 0.1))
            else:
                time.sleep(0.1)
    
    def start_auto_step(self):
        self.auto_step_thread = threading.Thread(target=self.auto_step)
        self.auto_step_thread.daemon = True
        self.auto_step_thread.start()
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def stop(self):
        self.is_running = False
    
    def get_results(self):
        """Get simulation results for plotting"""
        return {
            'attendance_history': self.attendance_history,
            'daily_rewards': self.daily_rewards,
            'daily_arrivals': self.daily_arrivals[:self.current_day].tolist(),
            'daily_success': self.daily_success[:self.current_day].tolist(),
            'ships': self.ships,
            'config': self.config,
            'current_day': self.current_day
        }
    
    def get_current_metrics(self):
        if self.current_day == 0 or not self.config:
            return {
                'efficiency': 0, 'avg_arrivals': 0, 'adaptive_success': 0, 'conservative_success': 0,
                'days_with_1': 0, 'days_with_0': 0, 'days_with_2plus': 0, 'current_day': 0,
                'total_days': self.config['total_days'] if self.config else 0,
                'daily_arrivals': [], 'daily_success': [], 'is_running': self.is_running,
                'is_paused': self.is_paused, 'avg_q_value': 0, 'total_strategies': 0
            }
        
        days_so_far = self.current_day
        days_with_1 = np.sum(self.daily_success[:days_so_far])
        efficiency = days_with_1 / days_so_far if days_so_far > 0 else 0
        
        adaptive_ships = [s for s in self.ships if s.epsilon > 0.1]
        conservative_ships = [s for s in self.ships if s.epsilon <= 0.1]
        
        adaptive_rates = [s.success_rate() for s in adaptive_ships]
        conservative_rates = [s.success_rate() for s in conservative_ships]
        
        avg_q_value = np.mean([np.max(s.Q) if len(s.Q) > 0 else 0 for s in self.ships])
        total_strategies = sum([len(s.strategies) for s in self.ships])
        
        return {
            'efficiency': float(efficiency),
            'avg_arrivals': float(np.mean(self.daily_arrivals[:days_so_far])) if days_so_far > 0 else 0,
            'adaptive_success': float(np.mean(adaptive_rates)) if adaptive_rates else 0,
            'conservative_success': float(np.mean(conservative_rates)) if conservative_rates else 0,
            'days_with_1': int(days_with_1),
            'days_with_0': int(np.sum(self.daily_arrivals[:days_so_far] == 0)),
            'days_with_2plus': int(np.sum(self.daily_arrivals[:days_so_far] >= 2)),
            'current_day': int(self.current_day),
            'total_days': int(self.config['total_days']),
            'daily_arrivals': self.daily_arrivals[:days_so_far].tolist(),
            'daily_success': self.daily_success[:days_so_far].tolist(),
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'avg_q_value': float(avg_q_value),
            'total_strategies': int(total_strategies)
        }

simulation = SimulationEngine()

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_performance_plot():
    """Create performance comparison plot"""
    results = simulation.get_results()
    
    if results['current_day'] == 0:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cumulative Rewards
    ax1 = axes[0, 0]
    days = range(results['current_day'])
    
    # Calculate cumulative rewards
    rl_cumulative = np.cumsum(results['daily_rewards']['rl'])
    llm_cumulative = np.cumsum(results['daily_rewards']['llm'])
    
    ax1.plot(days, rl_cumulative, 'blue', linewidth=2, label='Adaptive Ships')
    ax1.plot(days, llm_cumulative, 'green', linewidth=2, label='Conservative Ships')
    ax1.set_title('Cumulative Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily Attendance
    ax2 = axes[0, 1]
    capacity_limit = results['config']['total_ships'] * (results['config'].get('capacity', 60) / 100)
    
    ax2.plot(days, results['attendance_history'], 'purple', alpha=0.5, linewidth=1, label='Daily Attendance')
    ax2.axhline(y=capacity_limit, color='red', linestyle='--', linewidth=2, label=f'Capacity ({capacity_limit:.0f})')
    
    # Add moving average
    if len(results['attendance_history']) >= 7:
        moving_avg = pd.Series(results['attendance_history']).rolling(window=7).mean()
        ax2.plot(days, moving_avg, 'orange', linewidth=2, label='7-day MA')
    
    ax2.set_title('Attendance Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Attendance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Attendance Distribution
    ax3 = axes[1, 0]
    bins = np.linspace(0, results['config']['total_ships'], 21)
    ax3.hist(results['attendance_history'], bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=capacity_limit, color='red', linestyle='--', linewidth=2, label=f'Capacity ({capacity_limit:.0f})')
    ax3.axvline(x=np.mean(results['attendance_history']), color='green', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(results["attendance_history"]):.1f})')
    ax3.set_title('Attendance Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Attendance')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Daily Rewards Comparison
    ax4 = axes[1, 1]
    
    # Calculate moving averages
    rl_ma = pd.Series(results['daily_rewards']['rl']).rolling(window=7).mean()
    llm_ma = pd.Series(results['daily_rewards']['llm']).rolling(window=7).mean()
    
    ax4.plot(days, rl_ma, 'blue', linewidth=2, label='Adaptive Ships (MA)')
    ax4.plot(days, llm_ma, 'green', linewidth=2, label='Conservative Ships (MA)')
    
    # Add advantage area
    advantage = np.array(llm_ma) - np.array(rl_ma)
    ax4.fill_between(days, 0, advantage, where=advantage > 0, color='green', alpha=0.2, label='Conservative Better')
    ax4.fill_between(days, 0, advantage, where=advantage < 0, color='blue', alpha=0.2, label='Adaptive Better')
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax4.set_title('Daily Rewards (7-day Moving Average)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Average Daily Reward')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64 for web display
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_detailed_analysis_plot():
    """Create more detailed analysis plots"""
    results = simulation.get_results()
    
    if results['current_day'] == 0:
        return None
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Weekly Performance
    ax1 = plt.subplot(3, 3, 1)
    weeks = results['current_day'] // 7
    weekly_avg = []
    
    for week in range(weeks):
        start = week * 7
        end = min(start + 7, results['current_day'])
        week_attendance = results['attendance_history'][start:end]
        if week_attendance:
            weekly_avg.append(np.mean(week_attendance))
    
    if weekly_avg:
        ax1.bar(range(len(weekly_avg)), weekly_avg, color='skyblue', alpha=0.7)
        ax1.axhline(y=capacity_limit, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Weekly Average Attendance', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Average Attendance')
        ax1.grid(True, alpha=0.3)
    
    # 2. Day of Week Analysis
    ax2 = plt.subplot(3, 3, 2)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_attendance = {i: [] for i in range(7)}
    
    for day, att in enumerate(results['attendance_history']):
        day_attendance[day % 7].append(att)
    
    day_avg = [np.mean(day_attendance[i]) for i in range(7)]
    day_std = [np.std(day_attendance[i]) for i in range(7)]
    
    bars = ax2.bar(day_names, day_avg, yerr=day_std, capsize=5, color='lightcoral', alpha=0.8)
    ax2.axhline(y=capacity_limit, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Average Attendance by Day', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Attendance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Congested Days per Week
    ax3 = plt.subplot(3, 3, 3)
    congested_per_week = []
    
    for week in range(weeks):
        start = week * 7
        end = min(start + 7, results['current_day'])
        week_attendance = results['attendance_history'][start:end]
        congested = sum(1 for att in week_attendance if att > capacity_limit)
        congested_per_week.append(congested)
    
    if congested_per_week:
        ax3.bar(range(len(congested_per_week)), congested_per_week, color='salmon', alpha=0.7)
        ax3.axhline(y=np.mean(congested_per_week), color='blue', linestyle='--', linewidth=2,
                   label=f'Avg: {np.mean(congested_per_week):.1f}')
        ax3.set_title('Congested Days per Week', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Congested Days')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance Ratio
    ax4 = plt.subplot(3, 3, 4)
    performance_ratio = []
    
    for i in range(min(len(results['daily_rewards']['rl']), len(results['daily_rewards']['llm']))):
        if results['daily_rewards']['rl'][i] != 0:
            ratio = results['daily_rewards']['llm'][i] / results['daily_rewards']['rl'][i]
            performance_ratio.append(ratio)
        else:
            performance_ratio.append(1.0)
    
    days_range = range(len(performance_ratio))
    ax4.plot(days_range, performance_ratio, 'orange', alpha=0.5, linewidth=1, label='Daily Ratio')
    
    # Add moving average
    if len(performance_ratio) >= 7:
        ratio_ma = pd.Series(performance_ratio).rolling(window=7).mean()
        ax4.plot(days_range, ratio_ma, 'red', linewidth=2, label='7-day MA')
    
    ax4.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Equal Performance')
    ax4.set_title('Performance Ratio (Conservative/Adaptive)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Efficiency Progress
    ax5 = plt.subplot(3, 3, 5)
    efficiency_progress = []
    successes = 0
    
    for i in range(results['current_day']):
        if results['daily_success'][i]:
            successes += 1
        efficiency_progress.append(successes / (i + 1))
    
    ax5.plot(days_range[:results['current_day']], efficiency_progress, 'green', linewidth=2)
    ax5.set_title('Port Efficiency Over Time', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Day')
    ax5.set_ylabel('Efficiency')
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3)
    
    # 6. Strategy Evolution (if available)
    ax6 = plt.subplot(3, 3, 6)
    strategy_counts = [len(ship.strategies) for ship in simulation.ships]
    adaptive_counts = [s for s in strategy_counts if simulation.ships[i].epsilon > 0.1 for i, s in enumerate(strategy_counts)]
    conservative_counts = [s for s in strategy_counts if simulation.ships[i].epsilon <= 0.1 for i, s in enumerate(strategy_counts)]
    
    if adaptive_counts and conservative_counts:
        ax6.boxplot([adaptive_counts, conservative_counts], labels=['Adaptive', 'Conservative'])
        ax6.set_title('Strategies per Ship', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Number of Strategies')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Q-Value Distribution
    ax7 = plt.subplot(3, 3, 7)
    q_values = [np.max(ship.Q) for ship in simulation.ships]
    ax7.hist(q_values, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(x=np.mean(q_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(q_values):.2f}')
    ax7.set_title('Q-Value Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Max Q-Value')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Reward Heatmap (last 50 days)
    ax8 = plt.subplot(3, 3, 8)
    last_n = min(50, results['current_day'])
    if last_n > 0:
        reward_matrix = []
        for i in range(last_n):
            week = i // 7
            dow = i % 7
            reward_matrix.append([results['daily_rewards']['rl'][-last_n + i] if i < len(results['daily_rewards']['rl']) else 0])
        
        im = ax8.imshow(reward_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax8.set_title('Recent Reward Heatmap', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Recent Days')
        ax8.set_ylabel('Day Index')
        plt.colorbar(im, ax=ax8, label='Reward')
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    total_reward_adaptive = sum(results['daily_rewards']['rl'])
    total_reward_conservative = sum(results['daily_rewards']['llm'])
    optimal_days = sum(results['daily_success'])
    congestion_rate = (sum(1 for att in results['attendance_history'] if att > capacity_limit) / results['current_day']) * 100
    
    stats_text = f"""
    SIMULATION SUMMARY
    
    Days Completed: {results['current_day']}/{results['config']['total_days']}
    Port Efficiency: {(optimal_days/results['current_day']*100):.1f}%
    
    Reward Summary:
    • Adaptive Ships: {total_reward_adaptive:.0f}
    • Conservative: {total_reward_conservative:.0f}
    • Difference: {total_reward_conservative - total_reward_adaptive:+.0f}
    
    Attendance:
    • Optimal Days: {optimal_days}
    • Congestion Rate: {congestion_rate:.1f}%
    • Avg Attendance: {np.mean(results['attendance_history']):.1f}
    
    Learning Progress:
    • Avg Q-Value: {np.mean(q_values):.2f}
    • Total Strategies: {sum([len(ship.strategies) for ship in simulation.ships])}
    """
    
    ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

# ============================================================================
# FLASK WEB APPLICATION
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Marine Port Optimization - RL Agents with Visualizations</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .content { display: flex; padding: 20px; gap: 20px; flex-wrap: wrap; }
        .sidebar {
            width: 320px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
        }
        .main-content { flex: 1; min-width: 300px; }
        .control-group {
            margin-bottom: 20px;
        }
        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 10px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover:not(:disabled) { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-start { background: #28a745; color: white; }
        .btn-pause { background: #ffc107; color: #333; }
        .btn-stop { background: #dc3545; color: white; }
        .btn-refresh { background: #17a2b8; color: white; }
        .metrics {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .metric-label { font-weight: bold; color: #555; }
        .metric-value { color: #667eea; font-weight: bold; }
        .plot-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .plot-image {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 10px;
        }
        .status-running { background: #28a745; color: white; }
        .status-paused { background: #ffc107; color: #333; }
        .status-stopped { background: #dc3545; color: white; }
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px;
            margin-top: 15px;
            border-radius: 5px;
            font-size: 12px;
        }
        .tab-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab-button {
            background: #e0e0e0;
            color: #333;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .tab-button.active {
            background: #667eea;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        @media (max-width: 768px) {
            .content { flex-direction: column; }
            .sidebar { width: 100%; }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Marine Port Optimization with RL Agents</h1>
            <p>Real-time simulation with comprehensive visualizations</p>
        </div>
        <div class="content">
            <div class="sidebar">
                <h3>⚙️ Controls</h3>
                <div class="control-group">
                    <label>Total Ships:</label>
                    <input type="number" id="total_ships" value="20" min="5" max="100">
                </div>
                <div class="control-group">
                    <label>Total Days:</label>
                    <input type="number" id="total_days" value="365" min="10" max="730">
                </div>
                <div class="control-group">
                    <label>Adaptive Ships:</label>
                    <input type="number" id="adaptive_ships" value="10" min="0" max="100">
                </div>
                <div class="control-group">
                    <label>Port Capacity (%):</label>
                    <input type="number" id="capacity" value="60" min="10" max="90">
                </div>
                <div class="control-group">
                    <label>Speed (ms):</label>
                    <input type="range" id="speed" min="0" max="500" value="50" step="10">
                    <span id="speedValue">50</span>ms
                </div>
                <button class="btn-start" onclick="startSimulation()" id="startBtn">▶ Start</button>
                <button class="btn-pause" onclick="pauseSimulation()" id="pauseBtn" disabled>⏸ Pause</button>
                <button class="btn-stop" onclick="stopSimulation()" id="stopBtn" disabled>⏹ Stop</button>
                <button class="btn-refresh" onclick="refreshPlots()" id="refreshBtn">🔄 Refresh Plots</button>
                <div id="statusBadge" class="status-badge status-stopped">● Stopped</div>
                
                <div class="info-box">
                    <strong>🧠 RL Learning:</strong><br>
                    • 7-day attendance patterns<br>
                    • Q-learning with exploration<br>
                    • Adaptive ships (ε=0.15)<br>
                    • Conservative ships (ε=0.05)
                </div>
                
                <div class="metrics">
                    <h3>📊 Live Metrics</h3>
                    <div class="metric-item"><span class="metric-label">Day:</span><span class="metric-value" id="day">0/0</span></div>
                    <div class="metric-item"><span class="metric-label">Efficiency:</span><span class="metric-value" id="efficiency">0%</span></div>
                    <div class="metric-item"><span class="metric-label">Adaptive:</span><span class="metric-value" id="adaptive_success">0%</span></div>
                    <div class="metric-item"><span class="metric-label">Conservative:</span><span class="metric-value" id="conservative_success">0%</span></div>
                    <div class="metric-item"><span class="metric-label">Optimal Days:</span><span class="metric-value" id="optimal_days">0</span></div>
                    <div class="metric-item"><span class="metric-label">Avg Q-Value:</span><span class="metric-value" id="avg_q">0.00</span></div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="switchTab('overview')">📈 Overview</button>
                    <button class="tab-button" onclick="switchTab('detailed')">📊 Detailed Analysis</button>
                </div>
                
                <div id="overview-tab" class="tab-content active">
                    <div class="plot-container">
                        <h3>Performance Overview</h3>
                        <img id="performance-plot" class="plot-image" src="" alt="Performance Plot">
                    </div>
                </div>
                
                <div id="detailed-tab" class="tab-content">
                    <div class="plot-container">
                        <h3>Detailed Analysis</h3>
                        <img id="detailed-plot" class="plot-image" src="" alt="Detailed Analysis">
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let updateInterval = null;
        let currentTab = 'overview';
        
        function switchTab(tab) {
            currentTab = tab;
            
            // Update tab buttons
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Update tab content
            document.getElementById('overview-tab').classList.remove('active');
            document.getElementById('detailed-tab').classList.remove('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
            
            // Refresh plots for the new tab
            refreshPlots();
        }
        
        async function startSimulation() {
            const config = {
                total_ships: parseInt(document.getElementById('total_ships').value),
                total_days: parseInt(document.getElementById('total_days').value),
                num_adaptive_ships: parseInt(document.getElementById('adaptive_ships').value),
                capacity: parseInt(document.getElementById('capacity').value),
                delay: parseInt(document.getElementById('speed').value) / 1000.0
            };
            
            const response = await fetch('/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                document.getElementById('startBtn').disabled = true;
                document.getElementById('pauseBtn').disabled = false;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('statusBadge').className = 'status-badge status-running';
                document.getElementById('statusBadge').textContent = '● Running';
                
                if (updateInterval) clearInterval(updateInterval);
                updateInterval = setInterval(updateMetrics, 500);
                updateInterval = setInterval(refreshPlots, 2000);
            }
        }
        
        async function pauseSimulation() {
            const response = await fetch('/pause', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                const btn = document.getElementById('pauseBtn');
                const statusDiv = document.getElementById('statusBadge');
                
                if (data.paused) {
                    btn.textContent = '▶ Resume';
                    statusDiv.className = 'status-badge status-paused';
                    statusDiv.textContent = '⏸ Paused';
                } else {
                    btn.textContent = '⏸ Pause';
                    statusDiv.className = 'status-badge status-running';
                    statusDiv.textContent = '● Running';
                }
            }
        }
        
        async function stopSimulation() {
            await fetch('/stop', { method: 'POST' });
            if (updateInterval) clearInterval(updateInterval);
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('pauseBtn').disabled = true;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('pauseBtn').textContent = '⏸ Pause';
            document.getElementById('statusBadge').className = 'status-badge status-stopped';
            document.getElementById('statusBadge').textContent = '● Stopped';
        }
        
        async function refreshPlots() {
            if (currentTab === 'overview') {
                const response = await fetch('/plot/performance');
                const data = await response.json();
                if (data.image) {
                    document.getElementById('performance-plot').src = `data:image/png;base64,${data.image}`;
                }
            } else {
                const response = await fetch('/plot/detailed');
                const data = await response.json();
                if (data.image) {
                    document.getElementById('detailed-plot').src = `data:image/png;base64,${data.image}`;
                }
            }
        }
        
        async function updateMetrics() {
            try {
                const response = await fetch('/metrics');
                const data = await response.json();
                
                if (!data) return;
                
                document.getElementById('day').textContent = `${data.current_day}/${data.total_days}`;
                document.getElementById('efficiency').textContent = `${(data.efficiency * 100).toFixed(1)}%`;
                document.getElementById('adaptive_success').textContent = `${(data.adaptive_success * 100).toFixed(1)}%`;
                document.getElementById('conservative_success').textContent = `${(data.conservative_success * 100).toFixed(1)}%`;
                document.getElementById('optimal_days').textContent = data.days_with_1;
                document.getElementById('avg_q').textContent = data.avg_q_value.toFixed(3);
                
                if (data.current_day >= data.total_days && data.current_day > 0) {
                    clearInterval(updateInterval);
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('pauseBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('statusBadge').className = 'status-badge status-stopped';
                    document.getElementById('statusBadge').textContent = '● Complete';
                }
            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }
        
        document.getElementById('speed').addEventListener('input', function() {
            document.getElementById('speedValue').textContent = this.value;
        });
        
        // Initial refresh
        setTimeout(refreshPlots, 1000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start', methods=['POST'])
def start_simulation():
    global simulation
    try:
        config = request.json
        simulation = SimulationEngine()
        if simulation.initialize(config):
            simulation.start_auto_step()
            return jsonify({'status': 'started'})
        return jsonify({'status': 'error'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/pause', methods=['POST'])
def pause_simulation():
    global simulation
    if simulation.is_running:
        if simulation.is_paused:
            simulation.resume()
        else:
            simulation.pause()
        return jsonify({'paused': simulation.is_paused})
    return jsonify({'paused': False})

@app.route('/stop', methods=['POST'])
def stop_simulation():
    global simulation
    simulation.stop()
    return jsonify({'status': 'stopped'})

@app.route('/metrics')
def get_metrics():
    global simulation
    return jsonify(simulation.get_current_metrics())

@app.route('/plot/performance')
def get_performance_plot():
    """Return performance plot as base64 image"""
    img_base64 = create_performance_plot()
    if img_base64:
        return jsonify({'image': img_base64})
    return jsonify({'image': None})

@app.route('/plot/detailed')
def get_detailed_plot():
    """Return detailed analysis plot as base64 image"""
    img_base64 = create_detailed_analysis_plot()
    if img_base64:
        return jsonify({'image': img_base64})
    return jsonify({'image': None})

def find_free_port(start_port=5000, max_attempts=10):
    """Find a free port to run the server on"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return None

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    print("="*60)
    print("MARINE PORT OPTIMIZATION - RL AGENT SIMULATION WITH PLOTS")
    print("="*60)
    
    port = find_free_port(5000, 20)
    if port is None:
        print("❌ Could not find an available port.")
        exit(1)
    
    local_ip = get_local_ip()
    
    print(f"\n🌟 Server starting on port {port}...")
    print(f"\n📱 Open your browser and go to:")
    print(f"   → http://{local_ip}:{port}")
    print(f"   → http://127.0.0.1:{port}")
    
    print("\n📊 Features:")
    print("   • Real-time RL agent simulation")
    print("   • Live performance metrics")
    print("   • Interactive plots and visualizations")
    print("   • Performance comparison between adaptive and conservative ships")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)