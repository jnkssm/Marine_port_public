"""
Configuration parameters for the simulation
"""

# Simulation parameters
TOTAL_AGENTS = 20
CAPACITY = 60  # Percentage
NUM_DAYS = 365

# RL Agent parameters
RL_PARAMS = {
    'alpha': 0.1,      # Learning rate
    'gamma': 0.9,      # Discount factor
    'epsilon': 0.1,    # Exploration rate
    'strategy_mutation_rate': 0.5
}

# Adaptive vs Conservative parameters
ADAPTIVE_PARAMS = {
    'alpha': 0.15,
    'gamma': 0.9,
    'epsilon': 0.15,
    'strategy_mutation_rate': 1 #pure LLM response
}

CONSERVATIVE_PARAMS = {
    'alpha': 0.08,
    'gamma': 0.9,
    'epsilon': 0.05,
    'strategy_mutation_rate': 0.4
}

# Reward system
REWARDS = {
    'attend_non_congested': 3,   # BEST
    'attend_congested': -2,      # WORST
    'stay_congested': 1,         # GOOD
    'stay_non_congested': 0      # NEUTRAL
}

# Web server configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'auto_find_port': True
}

# Plotting style
PLOT_STYLE = {
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'figsize': (12, 10),
    'dpi': 100
}