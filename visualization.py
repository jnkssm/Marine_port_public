"""
Visualization and plotting functions.

Key changes:
  – All plots use attendance_history snapshot to avoid race conditions.
  – New create_multirun_plot() for averaged multi-run results with SD bands.
  – Existing overview and detailed plots preserved.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io
import base64
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE['style'])
sns.set_palette(PLOT_STYLE['palette'])


def _agent_cumulative_rewards(ships, agent_type: str):
    agents = [s for s in ships if getattr(s, 'agent_type', '') == agent_type]
    if not agents:
        return np.empty((0, 0))
    max_len = max(len(a.reward_history) for a in agents)
    if max_len == 0:
        return np.empty((0, 0))
    matrix = np.zeros((len(agents), max_len))
    for i, agent in enumerate(agents):
        h = np.array(agent.reward_history, dtype=float)
        matrix[i, :len(h)] = h
    return np.cumsum(matrix, axis=1)


def _agent_weekly_rewards(ships, agent_type: str):
    agents = [s for s in ships if getattr(s, 'agent_type', '') == agent_type]
    if not agents:
        return np.empty((0, 0))
    n_weeks = max(len(a.reward_history) // 7 for a in agents)
    if n_weeks == 0:
        return np.empty((0, 0))
    matrix = np.zeros((len(agents), n_weeks))
    for i, agent in enumerate(agents):
        h = np.array(agent.reward_history, dtype=float)
        for w in range(n_weeks):
            matrix[i, w] = h[w * 7:(w + 1) * 7].sum()
    return matrix


def _plot_mean_sd(ax, data_2d, color, label, alpha_band=0.20):
    if data_2d.size == 0:
        return
    mean = data_2d.mean(axis=0)
    x    = np.arange(len(mean))
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    if data_2d.shape[0] > 1:
        sd = data_2d.std(axis=0)
        ax.fill_between(x, mean - sd, mean + sd,
                        color=color, alpha=alpha_band, label=f'{label} ±1 SD')


def create_performance_plot(results):
    if results['current_day'] == 0:
        return None

    ships = results['ships']
    attendance_history = list(results['attendance_history'])
    n_days = len(attendance_history)

    adap_cum = _agent_cumulative_rewards(ships, 'adaptive')
    cons_cum = _agent_cumulative_rewards(ships, 'conservative')
    adap_wk  = _agent_weekly_rewards(ships, 'adaptive')
    cons_wk  = _agent_weekly_rewards(ships, 'conservative')

    fig, axes = plt.subplots(2, 2, figsize=PLOT_STYLE['figsize'])

    ax1 = axes[0, 0]
    _plot_mean_sd(ax1, adap_cum, color='#5b7fde', label='LLM / Adaptive agents (mean)')
    _plot_mean_sd(ax1, cons_cum, color='#3aaa5c', label='Conservative RL agents (mean)')
    ax1.set_title('Cumulative Rewards — Mean ± 1 SD per Agent', fontweight='bold')
    ax1.set_xlabel('Day'); ax1.set_ylabel('Cumulative Reward')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    capacity_limit = (results['config']['total_ships']
                      * results['config'].get('capacity', 60) / 100)
    days = range(n_days)
    ax2.plot(days, attendance_history, color='#9b59b6', alpha=0.45, linewidth=1, label='Daily')
    ax2.axhline(y=capacity_limit, color='red', linestyle='--', linewidth=2,
                label=f'Capacity ({capacity_limit:.0f})')
    if n_days >= 7:
        ma = pd.Series(attendance_history).rolling(window=7).mean()
        ax2.plot(days, ma, color='orange', linewidth=2, label='7-day MA')
    ax2.set_title('Attendance Over Time', fontweight='bold')
    ax2.set_xlabel('Day'); ax2.set_ylabel('Attendance')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    bins = np.linspace(0, results['config']['total_ships'], 21)
    ax3.hist(attendance_history, bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=capacity_limit, color='red', linestyle='--', linewidth=2, label='Capacity')
    ax3.axvline(x=np.mean(attendance_history), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(attendance_history):.1f}')
    ax3.set_title('Attendance Distribution', fontweight='bold')
    ax3.set_xlabel('Attendance'); ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    def _bar_weekly(matrix, color, label, offset):
        if matrix.size == 0:
            return
        mean = matrix.mean(axis=0)
        sd   = matrix.std(axis=0) if matrix.shape[0] > 1 else np.zeros_like(mean)
        x    = np.arange(len(mean)) + offset
        ax4.bar(x, mean, width=0.38, color=color, alpha=0.75, label=label,
                yerr=sd, capsize=3, error_kw=dict(elinewidth=1, ecolor='black', capthick=1.2))
    _bar_weekly(adap_wk, '#5b7fde', 'Adaptive (mean ± SD)',     -0.20)
    _bar_weekly(cons_wk, '#3aaa5c', 'Conservative (mean ± SD)', +0.20)
    ax4.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
    ax4.set_title('Weekly Total Reward — Mean ± 1 SD per Agent', fontweight='bold')
    ax4.set_xlabel('Week'); ax4.set_ylabel('Total Reward (week)')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return _fig_to_b64(fig)


def create_detailed_analysis_plot(results):
    if results['current_day'] == 0:
        return None

    ships = results['ships']
    attendance_history = list(results['attendance_history'])
    n_days = len(attendance_history)
    capacity_limit = (results['config']['total_ships']
                      * results['config'].get('capacity', 60) / 100)
    weeks = n_days // 7

    adap_cum = _agent_cumulative_rewards(ships, 'adaptive')
    cons_cum = _agent_cumulative_rewards(ships, 'conservative')
    adap_wk  = _agent_weekly_rewards(ships, 'adaptive')
    cons_wk  = _agent_weekly_rewards(ships, 'conservative')

    fig = plt.figure(figsize=(18, 14))

    ax1 = plt.subplot(3, 3, 1)
    weekly_avg = [
        np.mean(attendance_history[w * 7:min((w + 1) * 7, n_days)])
        for w in range(weeks)
    ]
    if weekly_avg:
        ax1.bar(range(len(weekly_avg)), weekly_avg, color='skyblue', alpha=0.7)
        ax1.axhline(y=capacity_limit, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Weekly Average Attendance', fontweight='bold')
        ax1.set_xlabel('Week'); ax1.set_ylabel('Attendance'); ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_data  = [[] for _ in range(7)]
    for d, att in enumerate(attendance_history):
        day_data[d % 7].append(att)
    day_avg = [np.mean(dd) for dd in day_data]
    day_std = [np.std(dd)  for dd in day_data]
    ax2.bar(day_names, day_avg, yerr=day_std, capsize=5, color='lightcoral', alpha=0.8)
    ax2.axhline(y=capacity_limit, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Average Attendance by Day', fontweight='bold')
    ax2.set_xlabel('Day'); ax2.set_ylabel('Attendance'); ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(3, 3, 3)
    congested = [
        sum(1 for att in attendance_history[w * 7:min((w + 1) * 7, n_days)]
            if att > capacity_limit)
        for w in range(weeks)
    ]
    if congested:
        ax3.bar(range(len(congested)), congested, color='salmon', alpha=0.7)
        ax3.axhline(y=np.mean(congested), color='blue', linestyle='--', linewidth=2,
                    label=f'Avg: {np.mean(congested):.1f}')
        ax3.set_title('Congested Days per Week', fontweight='bold')
        ax3.set_xlabel('Week'); ax3.set_ylabel('Congested Days')
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis='y')

    ax4 = plt.subplot(3, 3, 4)
    for mat, col in [(adap_cum, '#5b7fde'), (cons_cum, '#3aaa5c')]:
        if mat.size == 0:
            continue
        for row in mat:
            ax4.plot(row, color=col, linewidth=0.5, alpha=0.15)
    _plot_mean_sd(ax4, adap_cum, '#5b7fde', 'Adaptive mean',     alpha_band=0.18)
    _plot_mean_sd(ax4, cons_cum, '#3aaa5c', 'Conservative mean', alpha_band=0.18)
    ax4.set_title('Cumulative Rewards — Mean ± 1 SD\n(faint = individual agents)',
                  fontweight='bold', fontsize=9)
    ax4.set_xlabel('Day'); ax4.set_ylabel('Cumulative Reward')
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    for mat, col, lbl in [(adap_wk, '#5b7fde', 'Adaptive'), (cons_wk, '#3aaa5c', 'Conservative')]:
        if mat.size == 0:
            continue
        mean = mat.mean(axis=0)
        sd   = mat.std(axis=0) if mat.shape[0] > 1 else np.zeros_like(mean)
        x    = np.arange(len(mean))
        ax5.plot(x, mean, color=col, linewidth=2, label=f'{lbl} mean')
        ax5.fill_between(x, mean - sd, mean + sd, color=col, alpha=0.18, label='±1 SD')
    ax5.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
    ax5.set_title('Weekly Total Reward — Mean ± 1 SD per Agent', fontweight='bold')
    ax5.set_xlabel('Week'); ax5.set_ylabel('Total Reward (week)')
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    adap_counts = [len(s.strategies) for s in ships if s.agent_type == 'adaptive']
    cons_counts = [len(s.strategies) for s in ships if s.agent_type == 'conservative']
    if adap_counts and cons_counts:
        ax6.boxplot([adap_counts, cons_counts], labels=['Adaptive', 'Conservative'])
        ax6.set_title('Strategies per Ship', fontweight='bold')
        ax6.set_ylabel('Number of Strategies'); ax6.grid(True, alpha=0.3, axis='y')

    ax7 = plt.subplot(3, 3, 7)
    q_values = [float(np.max(s.Q)) for s in ships if len(s.Q) > 0]
    ax7.hist(q_values, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(x=np.mean(q_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(q_values):.2f}')
    ax7.set_title('Q-Value Distribution', fontweight='bold')
    ax7.set_xlabel('Max Q-Value'); ax7.set_ylabel('Frequency')
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    last_n = min(50, n_days)
    adap_agents = [s for s in ships if s.agent_type == 'adaptive']
    cons_agents = [s for s in ships if s.agent_type == 'conservative']
    rows, row_labels = [], []
    for ag in adap_agents[:5]:
        h = np.array(ag.reward_history[-last_n:], dtype=float)
        pad = np.zeros(last_n - len(h))
        rows.append(np.concatenate([pad, h])); row_labels.append(f'A{ag.agent_id}')
    for ag in cons_agents[:5]:
        h = np.array(ag.reward_history[-last_n:], dtype=float)
        pad = np.zeros(last_n - len(h))
        rows.append(np.concatenate([pad, h])); row_labels.append(f'C{ag.agent_id}')
    if rows:
        im = ax8.imshow(rows, aspect='auto', cmap='RdYlGn',
                        interpolation='nearest', vmin=-1, vmax=1)
        ax8.set_yticks(range(len(row_labels)))
        ax8.set_yticklabels(row_labels, fontsize=7)
        ax8.set_title('Reward Heatmap — Last 50 Days\n(top 5 per group)',
                      fontweight='bold', fontsize=9)
        ax8.set_xlabel('Recent Days')
        plt.colorbar(im, ax=ax8, label='Reward', shrink=0.8)

    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    adap_total = adap_cum[:, -1].mean() if adap_cum.size else 0
    cons_total = cons_cum[:, -1].mean() if cons_cum.size else 0
    adap_sd    = adap_cum[:, -1].std()  if adap_cum.size and adap_cum.shape[0] > 1 else 0
    cons_sd    = cons_cum[:, -1].std()  if cons_cum.size and cons_cum.shape[0] > 1 else 0
    n_adap = len(adap_agents); n_cons = len(cons_agents)
    stats_text = (
        f"  SIMULATION SUMMARY\n\n"
        f"  Days : {n_days}/{results['config']['total_days']}\n"
        f"  Eff  : {sum(results['daily_success']) / n_days * 100:.1f}%\n\n"
        f"  Cumulative reward (mean ± SD)\n"
        f"  Adaptive ({n_adap} agents)\n    {adap_total:+.1f} ± {adap_sd:.1f}\n"
        f"  Conservative ({n_cons} agents)\n    {cons_total:+.1f} ± {cons_sd:.1f}\n\n"
        f"  Avg Attendance : {np.mean(attendance_history):.2f}\n"
        f"  Avg Q-Value    : {np.mean(q_values):.3f}\n"
        f"  Total Strats   : {sum(len(s.strategies) for s in ships)}"
    )
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return _fig_to_b64(fig)


def create_multirun_plot(multi_engine) -> str:
    """
    4-panel figure summarising the multi-run experiment.

    Panel 1 – Mean ± SD attendance across all runs (time series).
    Panel 2 – Per-run cumulative reward for adaptive vs conservative (scatter + CI).
    Panel 3 – Box plots of key metrics across runs.
    Panel 4 – Reliability: coefficient of variation per metric.
    """
    if not multi_engine.run_summaries:
        return None

    summaries = multi_engine.run_summaries
    agg       = multi_engine.get_aggregated_stats()
    series    = multi_engine.get_attendance_series_stats()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: attendance time series mean ± SD ─────────────────────────────
    ax1 = axes[0, 0]
    if series['mean']:
        x    = np.arange(len(series['mean']))
        mean = np.array(series['mean'])
        upper = np.array(series['upper'])
        lower = np.array(series['lower'])
        ax1.plot(x, mean, color='#5b7fde', linewidth=2, label='Mean attendance')
        ax1.fill_between(x, lower, upper, color='#5b7fde', alpha=0.2, label='±1 SD')
        cfg = multi_engine.config
        cap = cfg['total_ships'] * cfg.get('capacity', 60) / 100
        ax1.axhline(y=cap, color='red', linestyle='--', linewidth=1.5,
                    label=f'Capacity ({cap:.0f})')
    ax1.set_title(f'Attendance Over Time — Mean ± 1 SD ({len(summaries)} runs)',
                  fontweight='bold')
    ax1.set_xlabel('Day'); ax1.set_ylabel('Ships attending')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Panel 2: per-run cumulative reward scatter ────────────────────────────
    ax2 = axes[0, 1]
    runs       = [s['run'] for s in summaries]
    adap_rews  = [s['adaptive_cum_reward']      for s in summaries]
    cons_rews  = [s['conservative_cum_reward']  for s in summaries]
    ax2.plot(runs, adap_rews, 'o-', color='#5b7fde', linewidth=1.5,
             markersize=6, label='Adaptive (per run)')
    ax2.plot(runs, cons_rews, 's-', color='#3aaa5c', linewidth=1.5,
             markersize=6, label='Conservative (per run)')
    if 'adaptive_cum_reward' in agg:
        m = agg['adaptive_cum_reward']['mean']
        s = agg['adaptive_cum_reward']['std']
        ax2.axhline(y=m, color='#5b7fde', linestyle='--', linewidth=1)
        ax2.fill_between([min(runs), max(runs)], m - s, m + s,
                         color='#5b7fde', alpha=0.08)
    if 'conservative_cum_reward' in agg:
        m = agg['conservative_cum_reward']['mean']
        s = agg['conservative_cum_reward']['std']
        ax2.axhline(y=m, color='#3aaa5c', linestyle='--', linewidth=1)
        ax2.fill_between([min(runs), max(runs)], m - s, m + s,
                         color='#3aaa5c', alpha=0.08)
    ax2.set_title('Cumulative Reward per Run', fontweight='bold')
    ax2.set_xlabel('Run #'); ax2.set_ylabel('Total reward (mean per agent)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Panel 3: box plots of key metrics ─────────────────────────────────────
    ax3 = axes[1, 0]
    metric_data = {
        'Efficiency\n(%)':    [s['efficiency'] * 100       for s in summaries],
        'Adap success\n(%)':  [s['adaptive_success_rate'] * 100 for s in summaries],
        'Cons success\n(%)':  [s['conservative_success_rate'] * 100 for s in summaries],
        'Congestion\n(%)':    [s['congestion_rate'] * 100   for s in summaries],
    }
    bp = ax3.boxplot(metric_data.values(), patch_artist=True, notch=False,
                     medianprops=dict(color='black', linewidth=2))
    colors_box = ['#5b7fde', '#3aaa5c', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax3.set_xticks(range(1, len(metric_data) + 1))
    ax3.set_xticklabels(metric_data.keys(), fontsize=9)
    ax3.set_title(f'Key Metrics Distribution ({len(summaries)} runs)', fontweight='bold')
    ax3.set_ylabel('Value (%)'); ax3.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: coefficient of variation (reliability) ───────────────────────
    ax4 = axes[1, 1]
    cv_metrics = {
        'Efficiency':     ('efficiency',                1),
        'Adap reward':    ('adaptive_cum_reward',        1),
        'Cons reward':    ('conservative_cum_reward',    1),
        'Adap success':   ('adaptive_success_rate',      1),
        'Cons success':   ('conservative_success_rate',  1),
        'Congestion':     ('congestion_rate',            1),
    }
    cv_names, cv_vals, cv_colors = [], [], []
    for label, (key, _) in cv_metrics.items():
        if key in agg and agg[key]['mean'] != 0:
            cv = abs(agg[key]['std'] / agg[key]['mean']) * 100
            cv_names.append(label)
            cv_vals.append(cv)
            cv_colors.append('#e74c3c' if cv > 20 else '#f39c12' if cv > 10 else '#3aaa5c')
    if cv_names:
        bars = ax4.barh(cv_names, cv_vals, color=cv_colors, alpha=0.75)
        ax4.axvline(x=10, color='orange', linestyle='--', linewidth=1.5,
                    label='10% CV (moderate variability)')
        ax4.axvline(x=20, color='red', linestyle='--', linewidth=1.5,
                    label='20% CV (high variability)')
        for bar, val in zip(bars, cv_vals):
            ax4.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}%', va='center', fontsize=9)
    ax4.set_title('Coefficient of Variation — Reliability Check', fontweight='bold')
    ax4.set_xlabel('CV (%) — lower = more reliable')
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return _fig_to_b64(fig)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return b64


def save_plots_to_file(results, output_dir='outputs/plots'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for name, fn in [('performance_plot', create_performance_plot),
                     ('detailed_analysis', create_detailed_analysis_plot)]:
        img = fn(results)
        if img:
            with open(f'{output_dir}/{name}.png', 'wb') as f:
                f.write(base64.b64decode(img))
    print(f'Plots saved to {output_dir}/')