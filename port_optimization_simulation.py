"""
Marine Port Optimization - Single Berth Scenario (FIXED)
365 days, 1 ship per day capacity
Pure El Farol Minority Game
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimulationConfig:
    total_ships: int = 100  # Total ships in the fleet
    total_days: int = 365  # One year simulation
    port_capacity: int = 1  # Can dock exactly 1 ship per day
    random_seed: int = 42

# ============================================================================
# SHIP AGENTS
# ============================================================================

class Ship:
    """Individual ship agent deciding which day to arrive"""
    
    def __init__(self, ship_id: int, strategy: str, memory_days: int = 30):
        self.ship_id = ship_id
        self.strategy = strategy
        self.memory_days = memory_days
        self.arrival_history = []  # Days they arrived
        self.success_history = []  # Whether they docked successfully
        self.total_success = 0
        self.total_attempts = 0
        
    def decide(self, congestion_history: np.ndarray, current_day: int) -> int:
        """
        Decide which future day to arrive
        Returns: day index (0 to total_days-1)
        """
        total_days = len(congestion_history)
        
        # Handle empty history (first day)
        if total_days == 0:
            return 0  # All ships arrive on day 0 initially
        
        if self.strategy == 'random':
            # Pure random: choose any future day
            return random.randint(0, total_days - 1)
        
        elif self.strategy == 'contrarian':
            # Avoid days that were recently crowded
            if total_days > self.memory_days:
                recent = congestion_history[-self.memory_days:]
                # Find days with lowest congestion (0 or 1 ships)
                low_congestion_days = np.where(recent <= 1)[0]
                if len(low_congestion_days) > 0:
                    # Convert relative to absolute day
                    future_day = current_day + random.choice(low_congestion_days)
                    return min(future_day, total_days - 1)
            return random.randint(0, total_days - 1)
        
        elif self.strategy == 'pattern_learner':
            # Learn patterns: which days of week are best
            if total_days >= 7:
                # Analyze day-of-week patterns
                dow_performance = np.zeros(7)
                for day in range(total_days):
                    dow = day % 7
                    if congestion_history[day] == 1:
                        dow_performance[dow] += 1  # Good day
                    elif congestion_history[day] > 1:
                        dow_performance[dow] -= 1  # Bad day
                
                # Choose best day of week
                best_dow = np.argmax(dow_performance)
                # Find next occurrence of best_dow
                days_ahead = (best_dow - (current_day % 7) + 7) % 7
                target_day = current_day + days_ahead
                return min(target_day, total_days - 1)
            return random.randint(0, total_days - 1)
        
        elif self.strategy == 'q_learner':
            # Simplified Q-learning
            if not hasattr(self, 'q_table'):
                self.q_table = np.zeros(7)  # Learn day-of-week preferences
            if random.random() < 0.1:  # Exploration
                dow = random.randint(0, 6)
            else:
                dow = np.argmax(self.q_table)
            
            days_ahead = (dow - (current_day % 7) + 7) % 7
            target_day = current_day + days_ahead
            return min(target_day, total_days - 1)
        
        elif self.strategy == 'llm_like':
            # Simulates LLM behavior: analyzes patterns and avoids crowds
            if total_days > 20:
                # Calculate congestion score for each future day
                future_horizon = min(30, total_days - current_day)
                if future_horizon > 0:
                    scores = []
                    for offset in range(future_horizon):
                        day = current_day + offset
                        if day < total_days:
                            # Lower score is better (less congestion)
                            recent_congestion = congestion_history[max(0, day-7):day+1]
                            score = np.mean(recent_congestion) if len(recent_congestion) > 0 else 0
                            # Add some randomness (LLM temperature)
                            score += random.uniform(0, 0.5)
                            scores.append(score)
                        else:
                            scores.append(float('inf'))
                    
                    if scores:
                        best_offset = np.argmin(scores)
                        return min(current_day + best_offset, total_days - 1)
            return random.randint(0, total_days - 1)
        
        else:  # conservative
            # Choose days that historically had 0 or 1 ships
            if total_days > 10:
                good_days = [i for i, c in enumerate(congestion_history[-50:]) if c <= 1]
                if good_days:
                    offset = random.choice(good_days)
                    future_day = current_day + offset
                    return min(future_day, total_days - 1)
            return random.randint(0, total_days - 1)
    
    def update(self, success: bool):
        """Update ship's internal state"""
        self.total_attempts += 1
        if success:
            self.total_success += 1
    
    def success_rate(self) -> float:
        return self.total_success / self.total_attempts if self.total_attempts > 0 else 0

# ============================================================================
# PORT SIMULATION
# ============================================================================

class PortSimulation:
    """Single-berth port simulation"""
    
    def __init__(self, config: SimulationConfig, num_llm_ships: int = 0):
        self.config = config
        self.num_llm_ships = num_llm_ships
        self.ships = []
        self.daily_arrivals = np.zeros(config.total_days, dtype=int)
        self.daily_success = np.zeros(config.total_days, dtype=bool)
        self.daily_waiting = np.zeros(config.total_days, dtype=int)
        
        self._initialize_ships()
    
    def _initialize_ships(self):
        """Create ships with different strategies"""
        strategies = ['random', 'contrarian', 'pattern_learner', 'conservative']
        
        for i in range(self.config.total_ships):
            if i < self.num_llm_ships:
                # LLM ships use more sophisticated strategy
                strategy = 'llm_like'
            else:
                strategy = random.choice(strategies)
            
            self.ships.append(Ship(i, strategy))
        
        # Add Q-learners as a small percentage for diversity
        num_q = max(1, int(self.config.total_ships * 0.1))
        for i in range(num_q):
            idx = self.num_llm_ships + i
            if idx < self.config.total_ships:
                self.ships[idx].strategy = 'q_learner'
        
        print(f"Initialized {self.num_llm_ships} LLM-like ships, {self.config.total_ships - self.num_llm_ships} heuristic ships")
    
    def run_day(self, day: int):
        """Simulate a single day of port operations"""
        # Ships decide which day to arrive (they plan ahead)
        arrivals_today = []
        
        for ship in self.ships:
            # Ships can only decide for future days
            if day < self.config.total_days:
                target_day = ship.decide(self.daily_arrivals[:day], day)
                # Ensure target_day is within bounds
                target_day = min(max(0, target_day), self.config.total_days - 1)
                if target_day == day:
                    arrivals_today.append(ship)
        
        # Process arrivals
        num_arrivals = len(arrivals_today)
        self.daily_arrivals[day] = num_arrivals
        
        # Determine who docks (only 1 ship can dock)
        if num_arrivals == 1:
            # Perfect: exactly one ship
            self.daily_success[day] = True
            arrivals_today[0].update(True)
            self.daily_waiting[day] = 0
            
        elif num_arrivals == 0:
            # No ships - capacity wasted
            self.daily_success[day] = False
            self.daily_waiting[day] = 0
            
        else:  # num_arrivals >= 2
            # Congestion: only 1 docks, others wait
            self.daily_success[day] = True  # One ship docks successfully
            
            # Choose randomly which ship docks
            docking_ship = random.choice(arrivals_today)
            for ship in arrivals_today:
                if ship == docking_ship:
                    ship.update(True)
                else:
                    ship.update(False)
                    self.daily_waiting[day] += 1
    
    def run_full_simulation(self):
        """Run all 365 days"""
        print(f"\n{'='*60}")
        print(f"Running simulation with {self.num_llm_ships} LLM ships...")
        print(f"{'='*60}")
        
        for day in range(self.config.total_days):
            self.run_day(day)
            
            if (day + 1) % 50 == 0:
                avg_arrivals = np.mean(self.daily_arrivals[:day+1])
                success_rate = np.mean(self.daily_success[:day+1])
                print(f"Day {day+1}/{self.config.total_days} | Avg arrivals: {avg_arrivals:.2f} | Success rate: {success_rate:.2%}")
        
        return self.get_metrics()
    
    def get_metrics(self):
        """Calculate performance metrics"""
        # Key metrics
        total_successful_days = np.sum(self.daily_success)
        success_rate = total_successful_days / self.config.total_days
        
        # Arrival distribution
        days_with_0 = np.sum(self.daily_arrivals == 0)
        days_with_1 = np.sum(self.daily_arrivals == 1)
        days_with_2plus = np.sum(self.daily_arrivals >= 2)
        
        # Efficiency: % of days with exactly 1 ship
        efficiency = days_with_1 / self.config.total_days
        
        # Congestion metric
        avg_arrivals = np.mean(self.daily_arrivals)
        total_waiting_days = np.sum(self.daily_waiting)
        
        # Ship performance
        ship_success_rates = [s.success_rate() for s in self.ships]
        
        # Separate LLM vs heuristic performance
        llm_rates = ship_success_rates[:self.num_llm_ships] if self.num_llm_ships > 0 else []
        heuristic_rates = ship_success_rates[self.num_llm_ships:]
        
        return {
            'success_rate': success_rate,
            'efficiency': efficiency,
            'days_with_0': days_with_0,
            'days_with_1': days_with_1,
            'days_with_2plus': days_with_2plus,
            'avg_arrivals': avg_arrivals,
            'total_waiting_days': total_waiting_days,
            'avg_ship_success': np.mean(ship_success_rates),
            'llm_success_rate': np.mean(llm_rates) if llm_rates else 0,
            'heuristic_success_rate': np.mean(heuristic_rates) if heuristic_rates else 0,
            'optimal_days_achieved': days_with_1,
            'wasted_capacity_days': days_with_0,
            'congestion_days': days_with_2plus
        }

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment():
    """Test different numbers of LLM ships (0 to 30)"""
    config = SimulationConfig(
        total_ships=100,
        total_days=365,
        port_capacity=1,
        random_seed=42
    )
    
    # Test LLM counts
    llm_counts = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    results = []
    
    for n_llm in llm_counts:
        sim = PortSimulation(config, num_llm_ships=n_llm)
        metrics = sim.run_full_simulation()
        metrics['num_llm_ships'] = n_llm
        metrics['llm_percentage'] = n_llm / config.total_ships * 100
        results.append(metrics)
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results_df: pd.DataFrame):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Efficiency (days with exactly 1 ship)
    ax1 = axes[0, 0]
    ax1.plot(results_df['llm_percentage'], results_df['efficiency'], 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Theoretical Maximum')
    ax1.set_xlabel('LLM Ships (%)')
    ax1.set_ylabel('Efficiency (% days with exactly 1 ship)')
    ax1.set_title('Port Utilization Efficiency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Arrival distribution
    ax2 = axes[0, 1]
    ax2.plot(results_df['llm_percentage'], results_df['days_with_1'], 'g-', linewidth=2, label='Optimal (1 ship)', marker='o')
    ax2.plot(results_df['llm_percentage'], results_df['days_with_0'], 'r--', linewidth=2, label='Wasted (0 ships)', marker='s')
    ax2.plot(results_df['llm_percentage'], results_df['days_with_2plus'], 'orange', linewidth=2, label='Congested (2+ ships)', marker='^')
    ax2.set_xlabel('LLM Ships (%)')
    ax2.set_ylabel('Number of Days')
    ax2.set_title('Daily Arrival Distribution (out of 365 days)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ship success rates
    ax3 = axes[1, 0]
    ax3.plot(results_df['llm_percentage'], results_df['llm_success_rate'], 'b-', linewidth=2, label='LLM Ships', marker='o')
    ax3.plot(results_df['llm_percentage'], results_df['heuristic_success_rate'], 'r-', linewidth=2, label='Heuristic Ships', marker='s')
    ax3.set_xlabel('LLM Ships (%)')
    ax3.set_ylabel('Ship Success Rate')
    ax3.set_title('Ship Performance Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Congestion metrics
    ax4 = axes[1, 1]
    ax4.plot(results_df['llm_percentage'], results_df['avg_arrivals'], 'purple', linewidth=2, marker='d')
    ax4.axhline(y=1.0, color='green', linestyle='--', label='Target (1 ship/day)')
    ax4.set_xlabel('LLM Ships (%)')
    ax4.set_ylabel('Average Daily Arrivals')
    ax4.set_title('Average Congestion Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('single_berth_port_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

def print_detailed_analysis(results_df: pd.DataFrame):
    """Print comprehensive analysis"""
    print("\n" + "="*70)
    print("SINGLE BERTH PORT OPTIMIZATION - DETAILED ANALYSIS")
    print("="*70)
    
    # Find optimal configuration
    optimal = results_df.loc[results_df['efficiency'].idxmax()]
    
    print(f"\n★ OPTIMAL CONFIGURATION:")
    print(f"   LLM Ships: {optimal['num_llm_ships']} ({optimal['llm_percentage']:.1f}%)")
    print(f"   Efficiency: {optimal['efficiency']:.2%} of days had exactly 1 ship")
    print(f"   → {optimal['days_with_1']} optimal days out of 365")
    print(f"   → {optimal['days_with_0']} wasted capacity days")
    print(f"   → {optimal['days_with_2plus']} congested days")
    
    # Compare with baseline (0% LLM)
    baseline = results_df[results_df['num_llm_ships'] == 0].iloc[0]
    improvement = (optimal['efficiency'] - baseline['efficiency']) / baseline['efficiency'] * 100
    
    print(f"\n📊 IMPROVEMENT VS BASELINE (0% LLM):")
    print(f"   Baseline efficiency: {baseline['efficiency']:.2%}")
    print(f"   Optimal efficiency: {optimal['efficiency']:.2%}")
    print(f"   Improvement: +{improvement:.1f}%")
    
    # Critical threshold (where performance drops)
    print(f"\n⚠ CRITICAL THRESHOLDS:")
    high_llm = results_df[results_df['num_llm_ships'] >= 50]
    if len(high_llm) > 0:
        worst = high_llm.loc[high_llm['efficiency'].idxmin()]
        print(f"   At {worst['llm_percentage']:.1f}% LLM: Efficiency drops to {worst['efficiency']:.2%}")
        print(f"   → {worst['congestion_days']} congested days (vs {optimal['congestion_days']} at optimum)")
    
    # Ship performance
    print(f"\n🚢 SHIP PERFORMANCE:")
    print(f"   Best LLM ship success rate: {optimal['llm_success_rate']:.2%}")
    print(f"   Best heuristic success rate: {optimal['heuristic_success_rate']:.2%}")
    
    if optimal['llm_success_rate'] > optimal['heuristic_success_rate']:
        print(f"   → LLM ships outperform heuristics at optimal penetration")
    else:
        print(f"   → Heuristics perform better at optimal penetration")
    
    # Practical recommendation
    print(f"\n💡 RECOMMENDATION:")
    if optimal['llm_percentage'] <= 20:
        print(f"   Deploy {optimal['num_llm_ships']} LLM-powered ships ({optimal['llm_percentage']:.1f}% of fleet)")
        print(f"   This achieves {optimal['efficiency']:.1%} port utilization")
    else:
        print(f"   Limit LLM deployment to {optimal['num_llm_ships']} ships ({optimal['llm_percentage']:.1f}%)")
        print(f"   Beyond this, herding causes congestion collapse")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("SINGLE BERTH MARINE PORT OPTIMIZATION")
    print("El Farol Minority Game - 365 Days, 1 Ship/Day Capacity")
    print("="*70)
    
    # Run experiment
    results_df = run_experiment()
    
    # Display summary
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    display_cols = ['num_llm_ships', 'llm_percentage', 'efficiency', 'days_with_1', 
                    'days_with_0', 'days_with_2plus', 'avg_arrivals']
    print(results_df[display_cols].to_string(index=False))
    
    # Save results
    results_df.to_csv("single_berth_results.csv", index=False)
    print("\n✓ Results saved to single_berth_results.csv")
    
    # Plot
    plot_results(results_df)
    
    # Detailed analysis
    print_detailed_analysis(results_df)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()