"""
RL Agent implementation for port optimization.

Contains two agent types:
  • RLAgent  – pure Q-learning with strategy mutation (no LLM)
  • LLMAgent – Q-learning + Ollama LLM strategy generation

PERSONA CLASSIFICATION:
  Each agent exposes get_persona() which analyses:
    • go_ratio        – fraction of days the agent chose to attend
    • reward_variance – stability of rewards (low = consistent, high = volatile)
    • q_spread        – max–min Q spread (high = discriminating learner)
    • strategy_count  – how many distinct strategies discovered
    • recent_trend    – slope of last-N rewards (positive = improving)

  Persona labels (per agent_type bucket):
    RL agents  → Bold Rusher | Steady Planner | Cautious Waiter | Rigid Follower
    LLM agents → Adaptive Explorer | Strategic Learner | Opportunist | Overcautious

FIXES applied vs original:
  FIX 2a – New LLM strategies initialise at a HIGHER Q (0.95× best).
  FIX 2b – Duplicate strategies get a Q BOOST instead of being dropped.
  FIX 2c – Fallback mutator flips 2-3 bits (not 1).
  FIX 2d – Prompt shows recent weekly reward per strategy.
  FIX 2e – Prompt explicitly lists existing strategies to avoid.
  FIX 2f – Variety guard: inject random strategy after N consecutive duplicates.
"""

import numpy as np
import random
import re
import pandas as pd
import requests
import json
from datetime import datetime

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

_DEFAULT_CAPACITY = 60


def call_ollama(prompt: str, model: str = OLLAMA_MODEL,
                timeout: int = 120) -> tuple:
    try:
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.5, "num_predict": 80},
        }
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(10, timeout))
        resp.raise_for_status()
        raw_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_text += chunk.get("response", "")
            if chunk.get("done", False):
                break
        return raw_text.strip(), ""
    except requests.exceptions.ConnectionError:
        return "", "[Ollama not reachable – run: ollama serve]"
    except requests.exceptions.Timeout:
        return "", f"[Ollama timed out after {timeout}s]"
    except Exception as exc:
        return "", f"[Ollama error: {exc}]"


_BASE_STRATEGIES = [
    [1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0],
]


def _random_strategy() -> list:
    s = [0] * 7
    for d in random.sample(range(7), random.choice([3, 4])):
        s[d] = 1
    return s


def _fix_day_count(s: list) -> None:
    cnt = sum(s)
    if cnt < 3:
        zeros = [i for i, v in enumerate(s) if v == 0]
        for i in random.sample(zeros, min(3 - cnt, len(zeros))):
            s[i] = 1
    elif cnt > 4:
        ones = [i for i, v in enumerate(s) if v == 1]
        for i in random.sample(ones, min(cnt - 4, len(ones))):
            s[i] = 0


def _init_strategies() -> list:
    strats = [s.copy() for s in _BASE_STRATEGIES]
    for _ in range(2):
        strats.append(_random_strategy())
    return strats


def _diverse_mutate(parent: list) -> list:
    new_s  = parent.copy()
    n_flip = random.randint(2, 3)
    for pos in random.sample(range(7), n_flip):
        new_s[pos] = 1 - new_s[pos]
    _fix_day_count(new_s)
    return new_s


# ── Persona classification helpers ────────────────────────────────────────────

_RL_PERSONAS = {
    "Bold Rusher":      "High go-ratio, volatile rewards. Attends frequently regardless of congestion.",
    "Steady Planner":   "Balanced go-ratio, low variance. Consistent, medium-risk strategy.",
    "Cautious Waiter":  "Low go-ratio, avoids congestion well. Stays home more than peers.",
    "Rigid Follower":   "Few strategies explored. Sticks to initial patterns without adapting.",
}

_LLM_PERSONAS = {
    "Adaptive Explorer":   "High strategy diversity, improving trend. LLM drives genuine exploration.",
    "Strategic Learner":   "Moderate diversity, positive reward trend. Balanced LLM guidance.",
    "Opportunist":         "High go-ratio spikes. Chases short-term gains, reward-volatile.",
    "Overcautious":        "Low go-ratio, minimal strategy growth. LLM calls rarely change behaviour.",
}

PERSONA_DESCRIPTIONS = {**_RL_PERSONAS, **_LLM_PERSONAS}


def _recent_trend(rewards: list, n: int = 21) -> float:
    """Linear slope of the last n rewards (positive = improving)."""
    data = rewards[-n:] if len(rewards) >= n else rewards
    if len(data) < 2:
        return 0.0
    x = np.arange(len(data), dtype=float)
    return float(np.polyfit(x, data, 1)[0])


def classify_persona(agent) -> str:
    """Return a persona label string for the given agent."""
    rh = agent.reward_history
    if not rh:
        return "Unknown"

    decisions_attend = sum(
        v for s in agent.strategies for v in s
    ) / max(1, len(agent.strategies) * 7)
    go_ratio      = agent.total_success / max(1, agent.total_attempts)
    reward_var    = float(np.var(rh)) if len(rh) > 1 else 0.0
    q_spread      = float(np.max(agent.Q) - np.min(agent.Q)) if len(agent.Q) > 1 else 0.0
    strat_count   = len(agent.strategies)
    trend         = _recent_trend(rh)

    is_llm = getattr(agent, 'use_llm', False)

    if is_llm:
        diversity_score = strat_count / max(7, strat_count)
        if diversity_score > 0.6 and trend > 0:
            return "Adaptive Explorer"
        elif trend > 0 and reward_var < 0.3:
            return "Strategic Learner"
        elif go_ratio > 0.6 and reward_var > 0.3:
            return "Opportunist"
        else:
            return "Overcautious"
    else:
        if go_ratio > 0.55 and reward_var > 0.25:
            return "Bold Rusher"
        elif q_spread < 0.15 and strat_count < 9:
            return "Rigid Follower"
        elif go_ratio < 0.4:
            return "Cautious Waiter"
        else:
            return "Steady Planner"


def get_agent_profile(agent) -> dict:
    """Return a rich profile dict for persona display."""
    rh = agent.reward_history
    n  = len(rh)
    go_decisions = [
        agent.weekly_strategy[d % 7]
        for d in range(n)
        if agent.weekly_strategy is not None
    ] if agent.weekly_strategy else []

    # day-level go-ratio from reward history structure
    # Use strategies attendance average as proxy
    avg_attend = np.mean([sum(s) / 7 for s in agent.strategies]) if agent.strategies else 0

    return {
        "agent_id":       agent.agent_id,
        "agent_type":     getattr(agent, 'agent_type', 'rl'),
        "use_llm":        getattr(agent, 'use_llm', False),
        "persona":        classify_persona(agent),
        "go_ratio":       round(avg_attend, 3),
        "avg_reward":     round(float(np.mean(rh)), 3) if rh else 0,
        "reward_std":     round(float(np.std(rh)), 3)  if rh else 0,
        "cumulative":     round(float(np.sum(rh)), 1)  if rh else 0,
        "q_spread":       round(float(np.max(agent.Q) - np.min(agent.Q)), 3) if len(agent.Q) > 1 else 0,
        "best_q":         round(float(np.max(agent.Q)), 3) if len(agent.Q) > 0 else 0,
        "strat_count":    len(agent.strategies),
        "trend":          round(_recent_trend(rh), 4),
        "success_rate":   round(agent.success_rate(), 3),
        "llm_attempts":   getattr(agent, 'llm_attempts', 0),
        "llm_successes":  getattr(agent, 'llm_successes', 0),
        "mutation_rate":  round(
            getattr(agent, 'mutation_successes', 0) / max(1, getattr(agent, 'mutation_attempts', 1)), 3
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pure RL Agent
# ══════════════════════════════════════════════════════════════════════════════

class RLAgent:
    use_llm    = False
    agent_type = "rl"

    def __init__(self, agent_id,
                 alpha=0.1, gamma=0.9, epsilon=0.1,
                 strategy_mutation_rate=0.5,
                 **_kwargs):
        self.agent_id   = agent_id
        self.agent_name = f"rl_Agent_{agent_id}"
        self.alpha                  = alpha
        self.gamma                  = gamma
        self.epsilon                = epsilon
        self.strategy_mutation_rate = strategy_mutation_rate
        self.strategies = _init_strategies()
        self.Q          = np.ones(len(self.strategies)) * 0.5
        self.current_week        = -1
        self.weekly_strategy     = None
        self.weekly_strategy_idx = None
        self.reward_history      = []
        self.strategy_history    = []
        self.created_strategies  = []
        self.global_attendance_history = []
        self.total_success      = 0
        self.total_attempts     = 0
        self.mutation_attempts  = 0
        self.mutation_successes = 0
        self.llm_attempts  = 0
        self.llm_successes = 0
        self.llm_log_df    = pd.DataFrame()

    def _select_weekly_strategy(self, week_num: int) -> None:
        if not self.strategies:
            self.weekly_strategy     = [0] * 7
            self.weekly_strategy_idx = 0
            return
        if random.random() < self.epsilon:
            self.weekly_strategy_idx = random.randint(0, len(self.strategies) - 1)
        else:
            self.weekly_strategy_idx = int(np.argmax(self.Q))
        self.weekly_strategy = self.strategies[self.weekly_strategy_idx].copy()
        self.strategy_history.append(self.weekly_strategy_idx)
        self.current_week = week_num

    def decide(self, day: int) -> int:
        week_num    = day // 7
        day_of_week = day % 7
        if week_num != self.current_week:
            self._select_weekly_strategy(week_num)
        return self.weekly_strategy[day_of_week] if self.weekly_strategy else 0

    def observe_reward(self, reward: float) -> None:
        self.reward_history.append(reward)
        if reward > 0:
            self.total_success += 1
        self.total_attempts += 1
        if self.weekly_strategy_idx is not None and self.weekly_strategy_idx < len(self.Q):
            future   = float(np.max(self.Q))
            td_error = reward + self.gamma * future - self.Q[self.weekly_strategy_idx]
            self.Q[self.weekly_strategy_idx] += self.alpha * td_error
            self.Q = np.clip(self.Q, -10, 10)
        if len(self.reward_history) % 7 == 0:
            self._try_mutation(len(self.reward_history) // 7)

    def _try_mutation(self, current_week: int) -> None:
        if current_week <= 1:
            return
        if random.random() > self.strategy_mutation_rate:
            return
        self.mutation_attempts += 1
        if self._create_new_strategy():
            self.mutation_successes += 1

    def _create_new_strategy(self) -> bool:
        if not self.strategies:
            return False
        parent_idx = int(np.argmax(self.Q))
        parent_idx = min(parent_idx, len(self.strategies) - 1)
        parent     = self.strategies[parent_idx].copy()
        new_s      = [1 - d if random.random() < 0.3 else d for d in parent]
        _fix_day_count(new_s)
        if new_s in self.strategies:
            return False
        self.strategies.append(new_s)
        self.Q = np.append(self.Q, np.clip(self.Q[parent_idx] * 0.9, 0.1, 2.0))
        self.created_strategies.append({
            'day': len(self.reward_history), 'strategy': new_s.copy(),
            'creation_method': 'rl_mutation', 'initial_q': float(self.Q[-1]),
        })
        return True

    def success_rate(self) -> float:
        return self.total_success / max(1, self.total_attempts)

    def get_persona_profile(self) -> dict:
        return get_agent_profile(self)

    def get_stats(self) -> dict:
        return {
            'agent_id': self.agent_id, 'agent_type': self.agent_type,
            'success_rate': self.success_rate(), 'strategies_count': len(self.strategies),
            'mutations': self.mutation_successes,
            'best_q': float(np.max(self.Q)) if len(self.Q) > 0 else 0,
            'avg_q':  float(np.mean(self.Q)) if len(self.Q) > 0 else 0,
            'llm_attempts': 0, 'llm_successes': 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LLM Agent
# ══════════════════════════════════════════════════════════════════════════════

class LLMAgent:
    use_llm    = True
    agent_type = "adaptive"
    _DUPLICATE_PATIENCE = 3

    def __init__(self, agent_id,
                 alpha=0.1, gamma=0.9, epsilon=0.1,
                 strategy_mutation_rate=0.5,
                 capacity: int = _DEFAULT_CAPACITY,
                 llm_timeout: int = 120,
                 **_kwargs):
        self.agent_id   = agent_id
        self.agent_name = f"adaptive_Agent_{agent_id}"
        self.alpha                  = alpha
        self.gamma                  = gamma
        self.epsilon                = epsilon
        self.strategy_mutation_rate = strategy_mutation_rate
        self.capacity               = capacity
        self.llm_timeout            = llm_timeout
        self.strategies = _init_strategies()
        self.Q          = np.ones(len(self.strategies)) * 0.5
        self.current_week        = -1
        self.weekly_strategy     = None
        self.weekly_strategy_idx = None
        self.reward_history      = []
        self.strategy_history    = []
        self.created_strategies  = []
        self.global_attendance_history = []
        self.total_success      = 0
        self.total_attempts     = 0
        self.mutation_attempts  = 0
        self.mutation_successes = 0
        self.llm_attempts  = 0
        self.llm_successes = 0
        self.llm_log_df    = pd.DataFrame()
        self._consecutive_duplicates = 0

    def _select_weekly_strategy(self, week_num: int) -> None:
        if not self.strategies:
            self.weekly_strategy     = [0] * 7
            self.weekly_strategy_idx = 0
            return
        if random.random() < self.epsilon:
            self.weekly_strategy_idx = random.randint(0, len(self.strategies) - 1)
        else:
            self.weekly_strategy_idx = int(np.argmax(self.Q))
        self.weekly_strategy = self.strategies[self.weekly_strategy_idx].copy()
        self.strategy_history.append(self.weekly_strategy_idx)
        self.current_week = week_num

    def decide(self, day: int) -> int:
        week_num    = day // 7
        day_of_week = day % 7
        if week_num != self.current_week:
            self._select_weekly_strategy(week_num)
        return self.weekly_strategy[day_of_week] if self.weekly_strategy else 0

    def observe_reward(self, reward: float) -> None:
        self.reward_history.append(reward)
        if reward > 0:
            self.total_success += 1
        self.total_attempts += 1
        if self.weekly_strategy_idx is not None and self.weekly_strategy_idx < len(self.Q):
            future   = float(np.max(self.Q))
            td_error = reward + self.gamma * future - self.Q[self.weekly_strategy_idx]
            self.Q[self.weekly_strategy_idx] += self.alpha * td_error
            self.Q = np.clip(self.Q, -10, 10)
        if len(self.reward_history) % 7 == 0:
            self._try_mutation(len(self.reward_history) // 7)

    def _try_mutation(self, current_week: int) -> None:
        if current_week <= 1:
            return
        if random.random() > self.strategy_mutation_rate:
            return
        self.mutation_attempts += 1
        if self._create_strategy_with_llm(current_week):
            self.mutation_successes += 1

    def _strategy_info(self) -> str:
        days  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pairs = sorted(zip(self.strategies, self.Q), key=lambda x: x[1], reverse=True)
        lines = []
        for rank, (strat, q) in enumerate(pairs, 1):
            attending = ', '.join(days[i] for i, v in enumerate(strat) if v == 1) or 'None'
            lines.append(f"  Rank {rank}: {strat}  →  [{attending}]  Q={q:.3f}")
        last7 = self.reward_history[-7:] if len(self.reward_history) >= 7 else self.reward_history
        if last7:
            rparts = [f"{days[i] if i < 7 else f'D{i}'}: {r:+.1f}" for i, r in enumerate(last7)]
            lines.append("  Last-week rewards : " + " | ".join(rparts))
            lines.append(f"  Last-week avg     : {np.mean(last7):.3f}")
        lines.append("\n  EXISTING strategies (do NOT repeat these):")
        for s in self.strategies:
            lines.append(f"    {s}")
        return "\n".join(lines)

    def _build_prompt(self, current_week: int) -> str:
        strategy_info = self._strategy_info()
        attendance    = list(self.global_attendance_history)
        total_ships   = getattr(self, 'total_ships', None)
        if total_ships is None and attendance:
            total_ships = max(attendance)
        elif total_ships is None:
            total_ships = 20
        capacity_pct   = self.capacity
        capacity_ships = int(total_ships * capacity_pct / 100)
        return f"""You are an AI ship agent optimising your port-arrival schedule to maximise rewards.

ENVIRONMENT PARAMETERS:
• Total ships in simulation : {total_ships}
• Port capacity threshold   : {capacity_pct}%  ({capacity_ships} ships)
• A day is CONGESTED when   : arrivals > {capacity_ships} ships

REWARD SYSTEM:
• Arrive on a NON-CONGESTED day   : +1
• Arrive on a CONGESTED day       : -1
• Stay home on a CONGESTED day    : +0.5
• Stay home on a NON-CONGESTED day:  0

MY CURRENT STRATEGY PERFORMANCE:
{strategy_info}

Historical total arrivals per day: {attendance}

Suggest ONE new weekly schedule that can maximize your rewards. Prefer less-congested days.

STRICT OUTPUT FORMAT — ONLY a Python list of 7 binary integers.
Example: [1,0,1,0,1,0,0]"""

    def _create_strategy_with_llm(self, current_week: int) -> bool:
        self.llm_attempts += 1
        prompt = self._build_prompt(current_week)
        raw_text, error = call_ollama(prompt, model=OLLAMA_MODEL, timeout=self.llm_timeout)
        display_raw  = raw_text if raw_text else error
        new_strategy = self._parse_llm_response(raw_text)
        best_idx = int(np.argmax(self.Q))
        best_q   = float(self.Q[best_idx])
        fallback_used = False
        if new_strategy is None:
            parent        = self.strategies[min(best_idx, len(self.strategies) - 1)].copy()
            new_strategy  = _diverse_mutate(parent)
            fallback_used = True
        is_duplicate = new_strategy in self.strategies
        if is_duplicate:
            self._consecutive_duplicates += 1
        else:
            self._consecutive_duplicates = 0
        if self._consecutive_duplicates >= self._DUPLICATE_PATIENCE:
            for _ in range(20):
                candidate = _random_strategy()
                if candidate not in self.strategies:
                    new_strategy = candidate
                    is_duplicate = False
                    self._consecutive_duplicates = 0
                    break
        self._log_llm_interaction(
            prompt=prompt, raw_response=display_raw, parsed=new_strategy,
            initial_q=best_q * 0.95, week=current_week, fallback=fallback_used,
        )
        if not is_duplicate:
            self.strategies.append(new_strategy)
            self.Q = np.append(self.Q, np.clip(best_q * 0.95, 0.1, 2.0))
        else:
            idx = self.strategies.index(new_strategy)
            self.Q[idx] = np.clip(self.Q[idx] * 1.05, -10, 10)
        self.llm_successes += 1
        return True

    def _parse_llm_response(self, text: str):
        if not text:
            return None
        for match in reversed(re.findall(r'\[[01,\s]+\]', text)):
            try:
                strat = eval(match)
                if len(strat) == 7 and all(x in (0, 1) for x in strat) and 2 <= sum(strat) <= 5:
                    return list(strat)
            except Exception:
                continue
        return None

    def _log_llm_interaction(self, prompt, raw_response, parsed, initial_q, week, fallback):
        row = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'agent_id': self.agent_id, 'agent_type': self.agent_type,
            'week': week, 'day': len(self.reward_history),
            'capacity_pct': self.capacity,
            'prompt': prompt, 'raw_llm_response': raw_response,
            'parsed_strategy': str(parsed), 'fallback_used': fallback,
            'initial_q_value': initial_q, 'success': True,
        }])
        self.llm_log_df = (row if self.llm_log_df.empty
                           else pd.concat([self.llm_log_df, row], ignore_index=True))

    def get_llm_log(self) -> pd.DataFrame:
        return self.llm_log_df

    def get_persona_profile(self) -> dict:
        return get_agent_profile(self)

    def success_rate(self) -> float:
        return self.total_success / max(1, self.total_attempts)

    def get_stats(self) -> dict:
        return {
            'agent_id': self.agent_id, 'agent_type': self.agent_type,
            'success_rate': self.success_rate(), 'strategies_count': len(self.strategies),
            'mutations': self.mutation_successes,
            'best_q': float(np.max(self.Q)) if len(self.Q) > 0 else 0,
            'avg_q':  float(np.mean(self.Q)) if len(self.Q) > 0 else 0,
            'llm_attempts': self.llm_attempts, 'llm_successes': self.llm_successes,
        }