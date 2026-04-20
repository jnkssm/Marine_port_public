"""
Simulation engine for port optimization.

NEW: MultiRunEngine – runs N independent simulations and aggregates results
     for mean ± SD reliability analysis.

Agent types:
  • "adaptive"     → LLMAgent  (Q-learning + Ollama)
  • "adaptive"     → RLAgent   (pure Q-learning when LLM disabled)
  • "conservative" → RLAgent   (conservative slot, always pure RL)
"""

import numpy as np
import threading
import time
import pandas as pd
import copy

from rl_agent import RLAgent, LLMAgent, get_agent_profile, PERSONA_DESCRIPTIONS
from config   import ADAPTIVE_PARAMS, CONSERVATIVE_PARAMS, REWARDS


class SimulationEngine:
    """Single-run simulation engine."""

    def __init__(self):
        self.is_running  = False
        self.is_paused   = False
        self.current_day = 0
        self.ships       = []
        self.config      = None
        self.auto_step_thread = None
        self.attendance_history = []
        self.daily_arrivals     = None
        self.daily_success      = None
        self.daily_rewards      = {'adaptive': [], 'conservative': []}
        self.all_llm_logs       = pd.DataFrame()

    def initialize(self, config: dict) -> bool:
        self.config             = config
        self.current_day        = 0
        self.ships              = []
        self.is_running         = True
        self.is_paused          = False
        self.attendance_history = []
        self.daily_rewards      = {'adaptive': [], 'conservative': []}
        self.all_llm_logs       = pd.DataFrame()
        
        self.daily_arrivals = np.zeros(config['total_days'], dtype=int)
        self.daily_success  = np.zeros(config['total_days'], dtype=bool)

        total_ships  = config['total_ships']
        num_adaptive = min(config.get('num_adaptive_ships', total_ships), total_ships)
        use_llm      = config.get('use_llm', False)
        capacity     = config.get('capacity', 60)
        llm_timeout  = config.get('llm_timeout', 120)

        for i in range(total_ships):
            if i < num_adaptive:
                if use_llm:
                    ship = LLMAgent(agent_id=i, capacity=capacity,
                                    llm_timeout=llm_timeout, **ADAPTIVE_PARAMS)
                    ship.total_ships = total_ships
                else:
                    ship = RLAgent(agent_id=i, **ADAPTIVE_PARAMS)
                    ship.agent_type = "adaptive"
            else:
                ship = RLAgent(agent_id=i, **CONSERVATIVE_PARAMS)
                ship.agent_type = "conservative"
            ship.global_attendance_history = self.attendance_history
            self.ships.append(ship)

        self.daily_arrivals = np.zeros(config['total_days'], dtype=int)
        self.daily_success  = np.zeros(config['total_days'], dtype=bool)
        return True

    def run_day(self) -> bool:
        if self.current_day >= self.config['total_days']:
            self.is_running = False
            return False

        decisions    = [ship.decide(self.current_day) for ship in self.ships]
        num_arrivals = sum(decisions)
        self.daily_arrivals[self.current_day] = num_arrivals
        self.attendance_history.append(num_arrivals)

        capacity_limit = (self.config['total_ships']
                          * (self.config.get('capacity', 60) / 100))
        is_congested = num_arrivals > capacity_limit

        adaptive_rewards, conservative_rewards = [], []

        for ship, decision in zip(self.ships, decisions):
            if decision == 1:
                reward = (REWARDS['attend_non_congested'] if not is_congested
                          else REWARDS['attend_congested'])
            else:
                reward = (REWARDS['stay_congested'] if is_congested
                          else REWARDS['stay_non_congested'])
            ship.observe_reward(reward)
            atype = getattr(ship, 'agent_type', 'rl')
            if atype == 'adaptive':
                adaptive_rewards.append(reward)
            else:
                conservative_rewards.append(reward)

        self.daily_rewards['adaptive'].append(
            np.mean(adaptive_rewards) if adaptive_rewards else 0)
        self.daily_rewards['conservative'].append(
            np.mean(conservative_rewards) if conservative_rewards else 0)

        self.daily_success[self.current_day] = (num_arrivals == 1)
        self.current_day += 1
        self._collect_llm_logs()
        return True

    def _collect_llm_logs(self) -> None:
        for ship in self.ships:
            if not hasattr(ship, 'llm_log_df') or ship.llm_log_df.empty:
                continue
            if self.all_llm_logs.empty:
                self.all_llm_logs = ship.llm_log_df.copy()
            else:
                existing = set(self.all_llm_logs['timestamp'].tolist())
                new_rows = ship.llm_log_df[~ship.llm_log_df['timestamp'].isin(existing)]
                if not new_rows.empty:
                    self.all_llm_logs = pd.concat(
                        [self.all_llm_logs, new_rows], ignore_index=True)

    def auto_step(self) -> None:
        while self.is_running:
            if not self.is_paused and self.current_day < self.config['total_days']:
                self.run_day()
                time.sleep(self.config.get('delay', 0.1))
            else:
                time.sleep(0.1)

    def start_auto_step(self) -> None:
        self.auto_step_thread = threading.Thread(target=self.auto_step, daemon=True)
        self.auto_step_thread.start()

    def pause(self)  -> None: self.is_paused = True
    def resume(self) -> None: self.is_paused = False
    def stop(self)   -> None: self.is_running = False

    def get_results(self) -> dict:
        """Get simulation results safely, handling uninitialized state."""
        self._collect_llm_logs()
        
        # Handle uninitialized or empty simulation
        if self.config is None or self.current_day == 0:
            return {
                'attendance_history': [],
                'daily_rewards': {'adaptive': [], 'conservative': []},
                'daily_arrivals': [],
                'daily_success': [],
                'ships': self.ships,
                'config': self.config or {},
                'current_day': self.current_day,
                'llm_logs': self.all_llm_logs,
            }
        
        attendance_snapshot = list(self.attendance_history)
        return {
            'attendance_history': attendance_snapshot,
            'daily_rewards':      self.daily_rewards,
            'daily_arrivals':     self.daily_arrivals[:self.current_day].tolist() if self.daily_arrivals is not None else [],
            'daily_success':      self.daily_success[:self.current_day].tolist() if self.daily_success is not None else [],
            'ships':              self.ships,
            'config':             self.config,
            'current_day':        len(attendance_snapshot),
            'llm_logs':           self.all_llm_logs,
        }

    def get_llm_logs(self) -> pd.DataFrame:
        self._collect_llm_logs()
        return self.all_llm_logs

    def get_agent_personas(self) -> list:
        """Return persona profiles for all ships."""
        return [get_agent_profile(s) for s in self.ships]

    def get_current_metrics(self) -> dict:
        if self.current_day == 0 or not self.config:
            return self._empty_metrics()
        n = self.current_day
        days_with_1 = int(np.sum(self.daily_success[:n])) if self.daily_success is not None else 0
        efficiency  = days_with_1 / n if n > 0 else 0
        
        adaptive_ships     = [s for s in self.ships if getattr(s, 'agent_type', '') == 'adaptive']
        conservative_ships = [s for s in self.ships if getattr(s, 'agent_type', '') in ('conservative', 'rl')]
        adaptive_rates     = [s.success_rate() for s in adaptive_ships]
        conservative_rates = [s.success_rate() for s in conservative_ships]
        q_values           = [float(np.max(s.Q)) if len(s.Q) > 0 else 0 for s in self.ships]
        llm_ships          = [s for s in self.ships if getattr(s, 'use_llm', False)]
        total_llm_attempts = sum(s.llm_attempts  for s in llm_ships)
        total_llm_success  = sum(s.llm_successes for s in llm_ships)
        return {
            'efficiency':           float(efficiency),
            'avg_arrivals':         float(np.mean(self.daily_arrivals[:n])),
            'adaptive_success':     float(np.mean(adaptive_rates))     if adaptive_rates     else 0,
            'conservative_success': float(np.mean(conservative_rates)) if conservative_rates else 0,
            'days_with_1':          days_with_1,
            'days_with_0':          int(np.sum(self.daily_arrivals[:n] == 0)),
            'days_with_2plus':      int(np.sum(self.daily_arrivals[:n] >= 2)),
            'current_day':          int(self.current_day),
            'total_days':           int(self.config['total_days']),
            'daily_arrivals':       self.daily_arrivals[:n].tolist(),
            'daily_success':        self.daily_success[:n].tolist(),
            'is_running':           self.is_running,
            'is_paused':            self.is_paused,
            'avg_q_value':          float(np.mean(q_values)),
            'total_strategies':     int(sum(len(s.strategies) for s in self.ships)),
            'total_llm_attempts':   total_llm_attempts,
            'total_llm_successes':  total_llm_success,
            'llm_success_rate':     total_llm_success / max(1, total_llm_attempts),
        }

    def _empty_metrics(self) -> dict:
        return {
            'efficiency': 0, 'avg_arrivals': 0,
            'adaptive_success': 0, 'conservative_success': 0,
            'days_with_1': 0, 'days_with_0': 0, 'days_with_2plus': 0,
            'current_day': 0,
            'total_days': self.config['total_days'] if self.config else 0,
            'daily_arrivals': [], 'daily_success': [],
            'is_running': self.is_running, 'is_paused': self.is_paused,
            'avg_q_value': 0, 'total_strategies': 0,
            'total_llm_attempts': 0, 'total_llm_successes': 0,
            'llm_success_rate': 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Run Engine
# ══════════════════════════════════════════════════════════════════════════════

class MultiRunEngine:
    """
    Runs N independent simulations and aggregates results for reliability
    analysis (mean ± SD across runs).

    Each run's summary is stored in self.run_summaries.
    Aggregated stats are computed on demand via get_aggregated_stats().
    """

    def __init__(self):
        self.is_running       = False
        self.is_paused        = False
        self.current_run      = 0
        self.total_runs       = 10
        self.run_summaries    = []       # list of per-run dicts
        self.run_personas     = []       # list of per-run persona lists
        self.config           = None
        self.progress_log     = []       # human-readable progress lines
        self._thread          = None
        self._stop_flag       = False
        self._pause_flag      = False

    def initialize(self, config: dict, total_runs: int = 10) -> bool:
        self.config        = config
        self.total_runs    = total_runs
        self.current_run   = 0
        self.run_summaries = []
        self.run_personas  = []
        self.progress_log  = []
        self.is_running    = False
        self.is_paused     = False
        self._stop_flag    = False
        self._pause_flag   = False
        return True

    def _run_one(self, run_idx: int) -> dict:
        """Execute a single complete simulation and return summary metrics."""
        engine = SimulationEngine()
        # Disable UI delay for batch runs
        cfg = dict(self.config)
        cfg['delay'] = 0
        engine.initialize(cfg)

        while engine.current_day < cfg['total_days'] and not self._stop_flag:
            while self._pause_flag and not self._stop_flag:
                time.sleep(0.05)
            engine.run_day()

        results = engine.get_results()
        n = results['current_day']
        if n == 0:
            return {}

        attendance = list(results['attendance_history'])
        n = len(attendance)

        capacity_limit = (cfg['total_ships'] * cfg.get('capacity', 60) / 100)
        congested_days = sum(1 for a in attendance if a > capacity_limit)

        adap_ships = [s for s in results['ships'] if getattr(s, 'agent_type', '') == 'adaptive']
        cons_ships = [s for s in results['ships'] if getattr(s, 'agent_type', '') == 'conservative']

        def _mean_cum(ships):
            if not ships:
                return 0.0
            return float(np.mean([sum(s.reward_history) for s in ships]))

        def _mean_success(ships):
            if not ships:
                return 0.0
            return float(np.mean([s.success_rate() for s in ships]))

        personas = engine.get_agent_personas()

        summary = {
            'run':                   run_idx + 1,
            'days':                  n,
            'efficiency':            float(sum(results['daily_success']) / n),
            'avg_attendance':        float(np.mean(attendance)),
            'congested_days':        congested_days,
            'congestion_rate':       congested_days / n,
            'adaptive_cum_reward':   _mean_cum(adap_ships),
            'conservative_cum_reward': _mean_cum(cons_ships),
            'adaptive_success_rate': _mean_success(adap_ships),
            'conservative_success_rate': _mean_success(cons_ships),
            'avg_q_value':           float(np.mean([float(np.max(s.Q)) for s in results['ships'] if len(s.Q) > 0])),
            'total_strategies':      int(sum(len(s.strategies) for s in results['ships'])),
            'llm_calls':             sum(getattr(s, 'llm_attempts', 0) for s in results['ships']),
            'llm_success_rate':      (
                sum(getattr(s, 'llm_successes', 0) for s in results['ships']) /
                max(1, sum(getattr(s, 'llm_attempts', 0) for s in results['ships']))
            ),
            'attendance_series':     attendance,  # full time-series for plotting
        }
        return summary, personas

    def _run_all(self) -> None:
        self.is_running = True
        self.progress_log.append(f"Starting {self.total_runs} simulation runs...")

        for i in range(self.total_runs):
            if self._stop_flag:
                break
            self.current_run = i + 1
            self.progress_log.append(f"Run {i+1}/{self.total_runs} started...")
            result = self._run_one(i)
            if result and not self._stop_flag:
                summary, personas = result
                self.run_summaries.append(summary)
                self.run_personas.append(personas)
                self.progress_log.append(
                    f"Run {i+1} done — efficiency {summary['efficiency']*100:.1f}%, "
                    f"adap reward {summary['adaptive_cum_reward']:+.1f}"
                )

        self.progress_log.append(f"All runs complete. {len(self.run_summaries)} successful.")
        self.is_running = False

    def start(self) -> None:
        self._stop_flag  = False
        self._pause_flag = False
        self._thread = threading.Thread(target=self._run_all, daemon=True)
        self._thread.start()

    def pause(self)  -> None:
        self._pause_flag = True
        self.is_paused   = True

    def resume(self) -> None:
        self._pause_flag = False
        self.is_paused   = False

    def stop(self) -> None:
        self._stop_flag = True
        self.is_running = False

    def get_progress(self) -> dict:
        return {
            'current_run':  self.current_run,
            'total_runs':   self.total_runs,
            'is_running':   self.is_running,
            'is_paused':    self.is_paused,
            'runs_complete': len(self.run_summaries),
            'log':          self.progress_log[-20:],
        }

    def get_raw_table(self) -> list:
        """Return per-run table rows for display."""
        rows = []
        for s in self.run_summaries:
            rows.append({
                'run':           s['run'],
                'efficiency':    round(s['efficiency'] * 100, 1),
                'adap_reward':   round(s['adaptive_cum_reward'], 1),
                'cons_reward':   round(s['conservative_cum_reward'], 1),
                'adap_success':  round(s['adaptive_success_rate'] * 100, 1),
                'cons_success':  round(s['conservative_success_rate'] * 100, 1),
                'congestion':    round(s['congestion_rate'] * 100, 1),
                'avg_attend':    round(s['avg_attendance'], 2),
                'avg_q':         round(s['avg_q_value'], 3),
                'strategies':    s['total_strategies'],
                'llm_calls':     s['llm_calls'],
            })
        return rows

    def get_aggregated_stats(self) -> dict:
        """Compute mean ± SD across all completed runs."""
        if not self.run_summaries:
            return {}

        keys = [
            'efficiency', 'adaptive_cum_reward', 'conservative_cum_reward',
            'adaptive_success_rate', 'conservative_success_rate',
            'congestion_rate', 'avg_attendance', 'avg_q_value',
            'total_strategies', 'llm_calls',
        ]
        agg = {}
        for k in keys:
            vals = [s[k] for s in self.run_summaries if k in s]
            if vals:
                agg[k] = {
                    'mean': float(np.mean(vals)),
                    'std':  float(np.std(vals)),
                    'min':  float(np.min(vals)),
                    'max':  float(np.max(vals)),
                }
        return agg

    def get_attendance_series_stats(self) -> dict:
        """
        Return mean ± SD attendance per day (for plotting),
        padded/truncated to the minimum run length.
        """
        if not self.run_summaries:
            return {'mean': [], 'upper': [], 'lower': []}
        series = [s['attendance_series'] for s in self.run_summaries]
        min_len = min(len(s) for s in series)
        matrix  = np.array([s[:min_len] for s in series], dtype=float)
        mean    = matrix.mean(axis=0)
        sd      = matrix.std(axis=0)
        return {
            'mean':  mean.tolist(),
            'upper': (mean + sd).tolist(),
            'lower': (mean - sd).tolist(),
        }

    def get_persona_summary(self) -> dict:
        """
        Aggregate persona counts across all runs, split by agent type.
        Returns dict with 'rl' and 'llm' sub-dicts mapping persona → count.
        """
        rl_counts  = {}
        llm_counts = {}
        for run_profiles in self.run_personas:
            for p in run_profiles:
                persona = p.get('persona', 'Unknown')
                if p.get('use_llm'):
                    llm_counts[persona] = llm_counts.get(persona, 0) + 1
                else:
                    rl_counts[persona]  = rl_counts.get(persona, 0) + 1
        return {'rl': rl_counts, 'llm': llm_counts}

    def get_persona_metric_table(self) -> list:
        """
        Per-persona average metrics across all runs and agents.
        Returns list of dicts suitable for a table.
        """
        if not self.run_personas:
            return []
        all_profiles = [p for run in self.run_personas for p in run]
        df = pd.DataFrame(all_profiles)
        if df.empty:
            return []
        numeric_cols = ['avg_reward', 'reward_std', 'cumulative', 'go_ratio',
                        'q_spread', 'success_rate', 'strat_count', 'trend']
        result = []
        for persona, grp in df.groupby('persona'):
            row = {'persona': persona, 'count': len(grp),
                   'type': 'LLM' if grp['use_llm'].any() else 'RL'}
            for c in numeric_cols:
                if c in grp.columns:
                    row[c + '_mean'] = round(float(grp[c].mean()), 3)
                    row[c + '_std']  = round(float(grp[c].std()),  3)
            result.append(row)
        return result