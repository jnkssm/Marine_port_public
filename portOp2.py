"""
Marine Port Optimization - GUI Simulation with Real Ollama LLM
Interactive simulation with real-time visualization and control
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import pandas as pd
import random
import requests
import time
import threading
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import json

# ============================================================================
# OLLAMA LLM INTEGRATION
# ============================================================================

class OllamaLLM:
    """Interface to Ollama LLM for ship decision making"""
    
    def __init__(self, model: str = "llama2", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self.total_calls = 0
        self.total_time = 0
        
    def query(self, prompt: str, temperature: float = 0.7, callback=None) -> str:
        """Send query to Ollama and get response"""
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": 100
                },
                timeout=30
            )
            
            self.total_calls += 1
            elapsed = time.time() - start_time
            self.total_time += elapsed
            
            if response.status_code == 200:
                result = response.json()["response"].strip()
                if callback:
                    callback(result, elapsed)
                return result
            else:
                error_msg = f"API error: {response.status_code}"
                if callback:
                    callback(error_msg, elapsed)
                return "0"
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if callback:
                callback(error_msg, 0)
            return "0"
    
    def get_stats(self):
        return f"Calls: {self.total_calls}, Avg: {self.total_time/self.total_calls:.2f}s" if self.total_calls > 0 else "No calls"

# ============================================================================
# SHIP AGENT
# ============================================================================

class Ship:
    """Individual ship agent"""
    
    def __init__(self, ship_id: int, strategy: str, llm: OllamaLLM = None, log_callback=None):
        self.ship_id = ship_id
        self.strategy = strategy
        self.llm = llm
        self.log_callback = log_callback
        self.total_success = 0
        self.total_attempts = 0
        self.planned_arrival = None
        self.decision_history = []
        
    def decide_with_llm(self, congestion_history: np.ndarray, current_day: int, total_days: int) -> int:
        """Use LLM to decide arrival day"""
        
        recent_days = min(30, len(congestion_history))
        recent_congestion = congestion_history[-recent_days:] if len(congestion_history) > 0 else []
        
        if len(recent_congestion) > 0:
            avg_congestion = np.mean(recent_congestion)
            days_with_1 = np.sum(recent_congestion == 1)
            days_with_0 = np.sum(recent_congestion == 0)
            days_with_2plus = np.sum(recent_congestion >= 2)
        else:
            avg_congestion = 0
            days_with_1 = 0
            days_with_0 = 0
            days_with_2plus = 0
        
        prompt = f"""You are an AI ship captain optimizing port arrivals.

PORT RULES:
- Port docks EXACTLY 1 ship per day
- Multiple ships same day → congestion (only 1 docks)
- No ships → wasted capacity
- Goal: Exactly 1 ship each day

STATUS:
- Today: Day {current_day} of {total_days}
- Choose future day between {current_day} and {total_days-1}

RECENT CONGESTION (last {recent_days} days):
{recent_congestion.tolist() if len(recent_congestion) > 0 else "No data"}

STATS:
- Avg arrivals: {avg_congestion:.2f}
- Perfect days (1 ship): {days_with_1}
- Wasted days (0 ships): {days_with_0}
- Congested days (2+ ships): {days_with_2plus}

STRATEGY:
1. Avoid historically congested days
2. Prefer days with exactly 1 ship
3. Consider day-of-week patterns
4. Spread out arrivals

Respond with ONLY the day number (between {current_day} and {total_days-1}):
"""
        
        def handle_response(response, elapsed):
            if self.log_callback:
                self.log_callback(f"Ship {self.ship_id} LLM response ({elapsed:.2f}s): {response[:50]}")
        
        response = self.llm.query(prompt, temperature=0.5, callback=handle_response)
        
        # Parse response
        try:
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                chosen_day = int(numbers[0])
                chosen_day = max(current_day, min(chosen_day, total_days - 1))
            else:
                chosen_day = random.randint(current_day, total_days - 1)
        except:
            chosen_day = random.randint(current_day, total_days - 1)
        
        self.decision_history.append({'day': current_day, 'chosen': chosen_day})
        return chosen_day
    
    def decide_with_heuristic(self, congestion_history: np.ndarray, current_day: int, total_days: int) -> int:
        """Heuristic strategies"""
        
        if current_day >= total_days - 1:
            return total_days - 1
        
        if self.strategy == 'random':
            return random.randint(current_day, total_days - 1)
        
        elif self.strategy == 'contrarian':
            if len(congestion_history) > 7:
                look_ahead = min(14, total_days - current_day)
                scores = []
                for offset in range(look_ahead):
                    day = current_day + offset
                    similar_days = [i for i in range(len(congestion_history)) 
                                  if i % 7 == day % 7][-5:]
                    if similar_days:
                        avg_congestion = np.mean([congestion_history[i] for i in similar_days])
                        scores.append(avg_congestion + random.uniform(0, 0.2))
                    else:
                        scores.append(random.uniform(0, 1))
                best_offset = np.argmin(scores)
                return min(current_day + best_offset, total_days - 1)
            return random.randint(current_day, total_days - 1)
        
        elif self.strategy == 'pattern_learner':
            if len(congestion_history) >= 14:
                dow_performance = np.zeros(7)
                for day in range(max(0, len(congestion_history)-30), len(congestion_history)):
                    dow = day % 7
                    if congestion_history[day] == 1:
                        dow_performance[dow] += 1
                    elif congestion_history[day] > 1:
                        dow_performance[dow] -= 0.5
                best_dow = np.argmax(dow_performance)
                days_ahead = (best_dow - (current_day % 7) + 7) % 7
                return min(current_day + days_ahead, total_days - 1)
            return random.randint(current_day, total_days - 1)
        
        else:  # conservative
            if len(congestion_history) > 10:
                good_days = []
                for i in range(max(0, len(congestion_history)-30), len(congestion_history)):
                    if congestion_history[i] <= 1:
                        for offset in range(1, 8):
                            future_day = current_day + offset
                            if future_day < total_days and (future_day % 7) == (i % 7):
                                good_days.append(offset)
                if good_days:
                    offset = random.choice(good_days)
                    return min(current_day + offset, total_days - 1)
            return random.randint(current_day, total_days - 1)
    
    def decide(self, congestion_history: np.ndarray, current_day: int, total_days: int) -> int:
        """Main decision method"""
        if self.strategy == 'llm_real' and self.llm is not None:
            return self.decide_with_llm(congestion_history, current_day, total_days)
        else:
            return self.decide_with_heuristic(congestion_history, current_day, total_days)
    
    def update(self, success: bool):
        self.total_attempts += 1
        if success:
            self.total_success += 1
    
    def success_rate(self) -> float:
        return self.total_success / self.total_attempts if self.total_attempts > 0 else 0

# ============================================================================
# PORT SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """Main simulation engine that runs in a separate thread"""
    
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.current_day = 0
        self.ships = []
        self.daily_arrivals = None
        self.daily_success = None
        self.daily_waiting = None
        self.config = None
        self.llm = None
        self.callbacks = {}
        
    def set_callbacks(self, on_update=None, on_complete=None, on_log=None):
        self.callbacks = {
            'on_update': on_update,
            'on_complete': on_complete,
            'on_log': on_log
        }
    
    def log(self, message):
        if self.callbacks.get('on_log'):
            self.callbacks['on_log'](message)
    
    def initialize(self, config):
        self.config = config
        self.current_day = 0
        self.ships = []
        
        if config['num_llm_ships'] > 0:
            self.llm = OllamaLLM(model=config['ollama_model'], url=config['ollama_url'])
            self.log(f"Initialized LLM with model: {config['ollama_model']}")
        
        # Create ships
        strategies = ['random', 'contrarian', 'pattern_learner', 'conservative']
        for i in range(config['total_ships']):
            if i < config['num_llm_ships']:
                strategy = 'llm_real'
                ship = Ship(i, strategy, self.llm, self.log)
            else:
                strategy = random.choice(strategies)
                ship = Ship(i, strategy, None, self.log)
            self.ships.append(ship)
        
        self.daily_arrivals = np.zeros(config['total_days'], dtype=int)
        self.daily_success = np.zeros(config['total_days'], dtype=bool)
        self.daily_waiting = np.zeros(config['total_days'], dtype=int)
        
        self.log(f"Initialized {config['num_llm_ships']} LLM ships, {config['total_ships'] - config['num_llm_ships']} heuristic ships")
    
    def run_day(self):
        """Run a single simulation day"""
        total_days = self.config['total_days']
        
        # Ships decide arrival days
        if self.current_day == 0:
            for ship in self.ships:
                target_day = random.randint(0, min(29, total_days - 1))
                ship.planned_arrival = target_day
        else:
            for ship in self.ships:
                if not hasattr(ship, 'planned_arrival') or ship.planned_arrival <= self.current_day:
                    if self.current_day < total_days - 1:
                        target_day = ship.decide(
                            self.daily_arrivals[:self.current_day], 
                            self.current_day, 
                            total_days
                        )
                        if target_day <= self.current_day:
                            target_day = self.current_day + 1 if self.current_day + 1 < total_days else self.current_day
                        ship.planned_arrival = target_day
                    else:
                        ship.planned_arrival = self.current_day
        
        # Process arrivals
        arrivals_today = [ship for ship in self.ships 
                         if hasattr(ship, 'planned_arrival') and ship.planned_arrival == self.current_day]
        
        num_arrivals = len(arrivals_today)
        self.daily_arrivals[self.current_day] = num_arrivals
        
        if num_arrivals == 1:
            self.daily_success[self.current_day] = True
            arrivals_today[0].update(True)
            self.daily_waiting[self.current_day] = 0
            
        elif num_arrivals == 0:
            self.daily_success[self.current_day] = False
            self.daily_waiting[self.current_day] = 0
            
        else:  # Congestion
            self.daily_success[self.current_day] = True
            docking_ship = random.choice(arrivals_today)
            
            for ship in arrivals_today:
                if ship == docking_ship:
                    ship.update(True)
                else:
                    ship.update(False)
                    self.daily_waiting[self.current_day] += 1
                    
                    if self.current_day < total_days - 1:
                        new_target = ship.decide(
                            self.daily_arrivals[:self.current_day + 1], 
                            self.current_day + 1, 
                            total_days
                        )
                        if new_target <= self.current_day:
                            new_target = self.current_day + 1 if self.current_day + 1 < total_days else self.current_day
                        ship.planned_arrival = new_target
    
    def run(self):
        """Main simulation loop"""
        self.is_running = True
        self.is_paused = False
        
        for day in range(self.config['total_days']):
            if not self.is_running:
                break
            
            while self.is_paused and self.is_running:
                time.sleep(0.1)
            
            self.current_day = day
            self.run_day()
            
            # Calculate metrics for update
            metrics = self.get_current_metrics()
            
            if self.callbacks.get('on_update'):
                self.callbacks['on_update'](day + 1, self.config['total_days'], metrics)
            
            # Add delay for visualization
            time.sleep(self.config.get('delay', 0.1))
        
        if self.is_running and self.callbacks.get('on_complete'):
            final_metrics = self.get_final_metrics()
            self.callbacks['on_complete'](final_metrics)
        
        self.is_running = False
    
    def get_current_metrics(self):
        """Get current simulation metrics"""
        if self.current_day == 0:
            return {'efficiency': 0, 'avg_arrivals': 0, 'llm_success': 0, 'heuristic_success': 0}
        
        days_so_far = self.current_day + 1
        days_with_1 = np.sum(self.daily_success[:days_so_far])
        efficiency = days_with_1 / days_so_far
        
        # Calculate ship success rates
        llm_rates = [s.success_rate() for s in self.ships if s.strategy == 'llm_real']
        heuristic_rates = [s.success_rate() for s in self.ships if s.strategy != 'llm_real']
        
        return {
            'efficiency': efficiency,
            'avg_arrivals': np.mean(self.daily_arrivals[:days_so_far]),
            'llm_success': np.mean(llm_rates) if llm_rates else 0,
            'heuristic_success': np.mean(heuristic_rates) if heuristic_rates else 0,
            'days_with_1': days_with_1,
            'days_with_0': np.sum(self.daily_arrivals[:days_so_far] == 0),
            'days_with_2plus': np.sum(self.daily_arrivals[:days_so_far] >= 2),
            'total_waiting': np.sum(self.daily_waiting[:days_so_far])
        }
    
    def get_final_metrics(self):
        """Get final simulation metrics"""
        return self.get_current_metrics()
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def stop(self):
        self.is_running = False

# ============================================================================
# GUI APPLICATION
# ============================================================================

class PortSimulationGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Marine Port Optimization - AI Simulation")
        self.root.geometry("1400x900")
        
        self.simulation = SimulationEngine()
        self.simulation.set_callbacks(
            on_update=self.update_display,
            on_complete=self.simulation_complete,
            on_log=self.add_log
        )
        
        self.simulation_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="Simulation Controls", padding="10")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Configuration inputs
        ttk.Label(left_panel, text="Total Ships:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.total_ships_var = tk.IntVar(value=20)
        ttk.Spinbox(left_panel, from_=5, to=100, textvariable=self.total_ships_var, width=10).grid(row=0, column=1, pady=5)
        
        ttk.Label(left_panel, text="Total Days:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.total_days_var = tk.IntVar(value=100)
        ttk.Spinbox(left_panel, from_=10, to=365, textvariable=self.total_days_var, width=10).grid(row=1, column=1, pady=5)
        
        ttk.Label(left_panel, text="LLM Ships:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.llm_ships_var = tk.IntVar(value=5)
        ttk.Spinbox(left_panel, from_=0, to=100, textvariable=self.llm_ships_var, width=10).grid(row=2, column=1, pady=5)
        
        ttk.Label(left_panel, text="Sim Speed (ms):", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.speed_var = tk.IntVar(value=100)
        ttk.Scale(left_panel, from_=0, to=500, variable=self.speed_var, orient=tk.HORIZONTAL).grid(row=3, column=1, pady=5, sticky=(tk.W, tk.E))
        ttk.Label(left_panel, textvariable=self.speed_var).grid(row=3, column=2, padx=5)
        
        ttk.Label(left_panel, text="Ollama Model:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="llama2")
        model_combo = ttk.Combobox(left_panel, textvariable=self.model_var, values=["llama2", "mistral", "phi", "tinyllama"])
        model_combo.grid(row=4, column=1, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Control buttons
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.start_btn = ttk.Button(left_panel, text="▶ Start Simulation", command=self.start_simulation)
        self.start_btn.grid(row=6, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        self.pause_btn = ttk.Button(left_panel, text="⏸ Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_btn.grid(row=7, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        self.stop_btn = ttk.Button(left_panel, text="⏹ Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.grid(row=8, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        # Metrics display
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        metrics_frame = ttk.LabelFrame(left_panel, text="Current Metrics", padding="5")
        metrics_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.metrics_vars = {
            'day': tk.StringVar(value="Day: 0/0"),
            'efficiency': tk.StringVar(value="Efficiency: 0%"),
            'avg_arrivals': tk.StringVar(value="Avg Arrivals: 0.00"),
            'llm_success': tk.StringVar(value="LLM Success: 0%"),
            'heuristic_success': tk.StringVar(value="Heuristic Success: 0%"),
            'days_optimal': tk.StringVar(value="Optimal Days: 0"),
            'days_wasted': tk.StringVar(value="Wasted Days: 0"),
            'days_congested': tk.StringVar(value="Congested Days: 0")
        }
        
        row = 0
        for key, var in self.metrics_vars.items():
            ttk.Label(metrics_frame, textvariable=var, font=('Arial', 9)).grid(row=row, column=0, sticky=tk.W, pady=2)
            row += 1
        
        # Right panel - Visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup plots
        self.setup_plots()
        
        # Bottom panel - Log
        log_frame = ttk.LabelFrame(main_frame, text="Simulation Log", padding="5")
        log_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure log colors
        self.log_text.tag_config("INFO", foreground="blue")
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("LLM", foreground="purple")
        
    def setup_plots(self):
        """Setup matplotlib subplots"""
        # Clear existing plots
        self.fig.clear()
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        
        # Initialize empty plots
        self.ax1.set_title('Port Efficiency Over Time')
        self.ax1.set_xlabel('Day')
        self.ax1.set_ylabel('Efficiency')
        self.ax1.set_ylim([0, 1])
        self.efficiency_line, = self.ax1.plot([], [], 'b-', linewidth=2)
        
        self.ax2.set_title('Daily Arrivals')
        self.ax2.set_xlabel('Day')
        self.ax2.set_ylabel('Number of Ships')
        self.arrivals_bar = self.ax2.bar([], [])
        
        self.ax3.set_title('Ship Performance')
        self.ax3.set_xlabel('Ship Type')
        self.ax3.set_ylabel('Success Rate')
        self.performance_bars = self.ax3.bar([0, 1], [0, 0])
        self.ax3.set_xticks([0, 1])
        self.ax3.set_xticklabels(['LLM Ships', 'Heuristic Ships'])
        self.ax3.set_ylim([0, 1])
        
        self.ax4.set_title('Arrival Distribution')
        self.ax4.set_xlabel('Arrival Count')
        self.ax4.set_ylabel('Days')
        self.distribution_bars = self.ax4.bar([0, 1, 2], [0, 0, 0])
        self.ax4.set_xticks([0, 1, 2])
        self.ax4.set_xticklabels(['0 Ships', '1 Ship', '2+ Ships'])
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def update_plots(self, day, total_days, metrics, daily_arrivals, daily_success):
        """Update all plots with current simulation data"""
        # Plot 1: Efficiency over time
        days_range = list(range(1, day + 1))
        efficiency_history = [np.sum(daily_success[:i+1]) / (i+1) for i in range(day)]
        self.efficiency_line.set_data(days_range[:len(efficiency_history)], efficiency_history)
        self.ax1.set_xlim([0, total_days])
        self.ax1.relim()
        self.ax1.autoscale_view(scalex=True, scaley=False)
        
        # Plot 2: Daily arrivals (last 50 days)
        start = max(0, day - 50)
        x = list(range(start, day))
        y = daily_arrivals[start:day]
        self.ax2.clear()
        self.ax2.bar(x, y, color='steelblue', alpha=0.7)
        self.ax2.set_title('Daily Arrivals (Last 50 Days)')
        self.ax2.set_xlabel('Day')
        self.ax2.set_ylabel('Number of Ships')
        self.ax2.axhline(y=1, color='green', linestyle='--', label='Target (1 ship)', alpha=0.7)
        self.ax2.legend()
        
        # Plot 3: Ship performance
        self.ax3.clear()
        self.ax3.bar([0, 1], [metrics['llm_success'], metrics['heuristic_success']], 
                    color=['purple', 'orange'], alpha=0.7)
        self.ax3.set_title('Ship Performance')
        self.ax3.set_xlabel('Ship Type')
        self.ax3.set_ylabel('Success Rate')
        self.ax3.set_xticks([0, 1])
        self.ax3.set_xticklabels(['LLM Ships', 'Heuristic Ships'])
        self.ax3.set_ylim([0, 1])
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Arrival distribution
        self.ax4.clear()
        self.ax4.bar([0, 1, 2], [metrics['days_with_0'], metrics['days_with_1'], metrics['days_with_2plus']],
                    color=['red', 'green', 'orange'], alpha=0.7)
        self.ax4.set_title('Arrival Distribution (All Days)')
        self.ax4.set_xlabel('Arrival Count')
        self.ax4.set_ylabel('Number of Days')
        self.ax4.set_xticks([0, 1, 2])
        self.ax4.set_xticklabels(['0 Ships\n(Wasted)', '1 Ship\n(Optimal)', '2+ Ships\n(Congested)'])
        self.ax4.grid(True, alpha=0.3, axis='y')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_display(self, day, total_days, metrics):
        """Update all display elements"""
        # Update metrics
        self.metrics_vars['day'].set(f"Day: {day}/{total_days}")
        self.metrics_vars['efficiency'].set(f"Efficiency: {metrics['efficiency']:.1%}")
        self.metrics_vars['avg_arrivals'].set(f"Avg Arrivals: {metrics['avg_arrivals']:.2f}")
        self.metrics_vars['llm_success'].set(f"LLM Success: {metrics['llm_success']:.1%}")
        self.metrics_vars['heuristic_success'].set(f"Heuristic Success: {metrics['heuristic_success']:.1%}")
        self.metrics_vars['days_optimal'].set(f"Optimal Days: {metrics['days_with_1']}")
        self.metrics_vars['days_wasted'].set(f"Wasted Days: {metrics['days_with_0']}")
        self.metrics_vars['days_congested'].set(f"Congested Days: {metrics['days_with_2plus']}")
        
        # Update plots
        self.update_plots(day, total_days, metrics, 
                         self.simulation.daily_arrivals, 
                         self.simulation.daily_success)
        
        # Update progress bar (optional - could add)
        
    def simulation_complete(self, metrics):
        """Handle simulation completion"""
        self.add_log("=" * 50, "SUCCESS")
        self.add_log("SIMULATION COMPLETE!", "SUCCESS")
        self.add_log("=" * 50, "SUCCESS")
        self.add_log(f"Final Port Efficiency: {metrics['efficiency']:.1%}", "SUCCESS")
        self.add_log(f"Optimal Days: {metrics['days_with_1']}/{self.simulation.config['total_days']}", "SUCCESS")
        self.add_log(f"LLM Ship Success Rate: {metrics['llm_success']:.1%}", "INFO")
        self.add_log(f"Heuristic Ship Success Rate: {metrics['heuristic_success']:.1%}", "INFO")
        
        # Show completion dialog
        messagebox.showinfo("Simulation Complete", 
                           f"Simulation finished!\n\nPort Efficiency: {metrics['efficiency']:.1%}\n"
                           f"Optimal Days: {metrics['days_with_1']}\n"
                           f"LLM Success: {metrics['llm_success']:.1%}\n"
                           f"Heuristic Success: {metrics['heuristic_success']:.1%}")
        
        # Reset buttons
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        
    def add_log(self, message, tag="INFO"):
        """Add message to log window"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, formatted_msg, tag)
        self.log_text.see(tk.END)
        
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        # Get configuration
        config = {
            'total_ships': self.total_ships_var.get(),
            'total_days': self.total_days_var.get(),
            'num_llm_ships': self.llm_ships_var.get(),
            'delay': self.speed_var.get() / 1000.0,  # Convert to seconds
            'ollama_model': self.model_var.get(),
            'ollama_url': "http://localhost:11434"
        }
        
        # Validate LLM ships don't exceed total ships
        if config['num_llm_ships'] > config['total_ships']:
            messagebox.showerror("Error", "LLM ships cannot exceed total ships!")
            return
        
        # Clear previous simulation
        self.log_text.delete(1.0, tk.END)
        self.add_log("Initializing simulation...", "INFO")
        self.add_log(f"Configuration: {config['total_ships']} ships, {config['total_days']} days", "INFO")
        self.add_log(f"LLM Ships: {config['num_llm_ships']} (Model: {config['ollama_model']})", "LLM")
        
        # Initialize simulation
        self.simulation.initialize(config)
        
        # Start simulation in thread
        self.simulation_thread = threading.Thread(target=self.simulation.run)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Update button states
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.add_log("Simulation started!", "SUCCESS")
        
    def pause_simulation(self):
        """Pause/resume simulation"""
        if self.simulation.is_paused:
            self.simulation.resume()
            self.pause_btn.config(text="⏸ Pause")
            self.add_log("Simulation resumed", "INFO")
        else:
            self.simulation.pause()
            self.pause_btn.config(text="▶ Resume")
            self.add_log("Simulation paused", "INFO")
    
    def stop_simulation(self):
        """Stop simulation"""
        self.simulation.stop()
        self.add_log("Stopping simulation...", "ERROR")
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    # Check Ollama availability
    try:
        test_llm = OllamaLLM()
        response = test_llm.query("Hello", temperature=0)
        print("✓ Ollama is available!")
    except:
        print("⚠ Ollama not detected. LLM ships will fall back to heuristics.")
        print("  To use real LLM, start Ollama: 'ollama serve'")
    
    # Launch GUI
    app = PortSimulationGUI()
    app.run()

if __name__ == "__main__":
    main()