"""
Web GUI for marine port optimization simulation with LLM tracking.
New features:
  • Agent Persona tab — deep per-agent analysis with persona classification
  • Multi-Run tab     — run N simulations, view raw per-run table + avg plots
"""

from flask import Flask, render_template_string, jsonify, request, send_file
from simulation_engine import SimulationEngine, MultiRunEngine
from visualization import (create_performance_plot, create_detailed_analysis_plot,
                            create_multirun_plot)
from config import WEB_CONFIG
from rl_agent import OLLAMA_URL, OLLAMA_MODEL, PERSONA_DESCRIPTIONS
import socket
import pandas as pd
import requests
import io
import html as html_module

app = Flask(__name__)
app.config['DEBUG'] = WEB_CONFIG['debug']

simulation   = SimulationEngine()
multi_engine = MultiRunEngine()


def ollama_is_alive() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = r.json().get('models', [])
        model_available = any(OLLAMA_MODEL in model.get('name', '') for model in models)
        if not model_available:
            print(f"Warning: Model {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")
        return True
    except Exception:
        return False


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Marine Port Optimization - RL Agents with LLM</title>
    <meta charset="UTF-8">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            padding: 20px; min-height: 100vh;
        }
        .container { max-width:1800px; margin:0 auto; background:white; border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,.3); overflow:hidden; }
        .header { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:25px; text-align:center; }
        .header h1 { font-size:1.8em; margin-bottom:6px; }
        .ollama-badge { display:inline-block; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:bold; margin-top:6px; }
        .ollama-ok  { background:#28a745; color:white; }
        .ollama-err { background:#dc3545; color:white; }
        .content { display:flex; padding:20px; gap:20px; flex-wrap:wrap; }
        .sidebar { width:300px; background:#f8f9fa; border-radius:15px; padding:18px; flex-shrink:0; }
        .main-content { flex:1; min-width:300px; }
        .control-group { margin-bottom:12px; }
        .control-group label { display:block; margin-bottom:4px; font-weight:bold; font-size:12px; }
        .control-group .hint { font-size:10px; color:#888; margin-top:2px; }
        input[type=number], input[type=range] { width:100%; padding:7px; border:2px solid #e0e0e0; border-radius:8px; }
        .section-title { font-size:11px; font-weight:bold; text-transform:uppercase; color:#667eea; letter-spacing:.5px; margin:14px 0 5px; }
        button { width:100%; padding:10px; margin-top:7px; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:13px; }
        .btn-start  { background:#28a745; color:white; }
        .btn-pause  { background:#ffc107; color:#333; }
        .btn-stop   { background:#dc3545; color:white; }
        .btn-refresh{ background:#17a2b8; color:white; }
        .btn-export { background:#6c757d; color:white; }
        .btn-multirun { background:#764ba2; color:white; }
        .metrics { background:white; border-radius:10px; padding:12px; margin-top:16px; }
        .metric-item { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #e0e0e0; font-size:12px; }
        .plot-container { background:white; border-radius:10px; padding:16px; margin-bottom:16px; }
        .plot-image { width:100%; height:auto; border-radius:8px; }
        .status-badge { display:inline-block; padding:4px 10px; border-radius:5px; margin-top:8px; font-size:12px; }
        .status-running { background:#28a745; color:white; }
        .status-paused  { background:#ffc107; color:#333; }
        .status-stopped { background:#dc3545; color:white; }
        .tab-buttons { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
        .tab-button { background:#e0e0e0; padding:9px 16px; border:none; border-radius:8px; cursor:pointer; transition:all .2s; font-size:13px; }
        .tab-button:hover { background:#c0c0c0; }
        .tab-button.active { background:#667eea; color:white; }
        .tab-content { display:none; }
        .tab-content.active { display:block; }
        .tag-adaptive-rl { background: #e3f5ec; color: #1a6e3c; font-size: 11px; padding: 3px 8px; border-radius: 10px; }

        .dot-llm  { background:#667eea; }
        .dot-rl   { background:#3aaa5c; }     /* Adaptive RL (no LLM) */
        .dot-cons { background:#ffc107; }    

        /* ── Persona styles ── */
        .persona-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px,1fr)); gap:14px; margin-top:12px; }
        .persona-card { background:white; border:1.5px solid #e0e0e0; border-radius:12px; padding:14px; }
        .persona-card.llm  { border-color:#5b7fde; }
        .persona-card.rl   { border-color:#3aaa5c; }
        .persona-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
        .persona-name { font-size:14px; font-weight:bold; }
        .persona-tag { font-size:11px; padding:3px 8px; border-radius:10px; }
        .persona-card.adaptive-rl { border-color: #3aaa5c; border-left: 4px solid #3aaa5c; }

        .tag-llm  { background:#e3ecff; color:#2c5fcb; }
        .tag-rl   { background:#e3f5ec; color:#1a6e3c; }
        .persona-label { font-size:13px; color:#764ba2; font-weight:bold; margin-bottom:4px; }
        .persona-desc  { font-size:11px; color:#666; margin-bottom:10px; line-height:1.4; }
        .persona-metrics { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
        .pm-item { background:#f8f9fa; border-radius:6px; padding:5px 8px; }
        .pm-label { font-size:10px; color:#888; }
        .pm-value { font-size:13px; font-weight:bold; color:#333; }
        .persona-section-label { font-size:12px; font-weight:bold; text-transform:uppercase; color:#667eea; margin:16px 0 8px; letter-spacing:.5px; }
        .decision-bar { display:flex; height:12px; border-radius:6px; overflow:hidden; margin:6px 0 2px; }
        .bar-go   { background:#5b7fde; }
        .bar-stay { background:#e0e0e0; }
        .bar-label { display:flex; justify-content:space-between; font-size:10px; color:#888; }

        /* ── Multi-run styles ── */
        .multirun-controls { background:#f0f4ff; border-radius:10px; padding:16px; margin-bottom:16px; }
        .progress-log { background:#1e1e1e; color:#a8ff78; font-family:monospace; font-size:12px; padding:12px; border-radius:8px; height:160px; overflow-y:auto; margin-top:10px; }
        .run-table { width:100%; border-collapse:collapse; font-size:12px; }
        .run-table th { background:#667eea; color:white; padding:9px 8px; text-align:center; }
        .run-table td { border:1px solid #ddd; padding:7px 8px; text-align:center; }
        .run-table tr:nth-child(even) { background:#f9f9f9; }
        .run-table tr.avg-row { background:#fff3cd; font-weight:bold; }
        .run-table tr.avg-row td { border-top:2px solid #ffc107; }
        .agg-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px,1fr)); gap:10px; margin-bottom:16px; }
        .agg-card { background:#f8f9fa; border-radius:10px; padding:12px; border-left:4px solid #667eea; }
        .agg-metric { font-size:11px; color:#666; }
        .agg-mean { font-size:20px; font-weight:bold; color:#333; }
        .agg-sd { font-size:11px; color:#888; }
        .agg-cv { font-size:11px; margin-top:3px; }
        .cv-good   { color:#28a745; }
        .cv-medium { color:#ffc107; }
        .cv-high   { color:#dc3545; }
        .multirun-progress-bar { height:8px; background:#e0e0e0; border-radius:4px; margin:8px 0; }
        .multirun-progress-fill { height:8px; background:linear-gradient(90deg,#667eea,#764ba2); border-radius:4px; transition:width .5s; }

        /* ── LLM table ── */
        .llm-table { width:100%; border-collapse:collapse; font-size:12px; }
        .llm-table th { background:#667eea; color:white; padding:9px 8px; text-align:left; position:sticky; top:0; z-index:1; white-space:nowrap; }
        .llm-table td { border:1px solid #ddd; padding:7px; vertical-align:top; }
        .llm-table tr:nth-child(even) { background:#f9f9f9; }
        .cell-pre { white-space:pre-wrap; word-break:break-word; font-family:'Courier New',monospace; font-size:11px; background:#f4f4f4; border:1px solid #ddd; border-radius:4px; padding:5px 7px; max-height:180px; overflow-y:auto; }
        .llm-container { max-height:500px; overflow-y:auto; overflow-x:auto; }
        .llm-stats { background:#e3f2fd; padding:12px 16px; border-radius:8px; margin-bottom:12px; }
        .checkbox-label { display:flex; align-items:center; gap:10px; margin-top:8px; }
        .checkbox-label input { width:auto; }
        .fallback-yes { color:#dc3545; font-weight:bold; }
        .fallback-no  { color:#28a745; }
        .legend { display:flex; gap:12px; margin-bottom:8px; font-size:12px; }
        .legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }
        .dot-llm  { background:#667eea; }
        .dot-rl   { background:#28a745; }
        .dot-cons { background:#ffc107; }
        .persona-table { width:100%; border-collapse:collapse; font-size:12px; margin-top:12px; }
        .persona-table th { background:#764ba2; color:white; padding:8px 10px; text-align:left; }
        .persona-table td { border:1px solid #ddd; padding:7px 10px; }
        .persona-table tr:nth-child(even) { background:#f9f9f9; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Marine Port Optimization — RL &amp; LLM Agents</h1>
        <p>Real-time simulation · Agent Persona Analysis · Multi-Run Reliability</p>
        <div id="ollamaStatus" class="ollama-badge ollama-err">Checking Ollama...</div>
    </div>
    <div class="content">

        <!-- ── SIDEBAR ── -->
        <div class="sidebar">
            <h3>Simulation Controls</h3>
            <div class="section-title">Fleet</div>
            <div class="control-group">
                <label>Total Ships:</label>
                <input type="number" id="total_ships" value="20" min="5" max="100">
            </div>
            <div class="control-group">
                <label>Total Days:</label>
                <input type="number" id="total_days" value="365" min="10" max="730">
            </div>
            <div class="control-group">
                <label>LLM / Adaptive Ships:</label>
                <input type="number" id="adaptive_ships" value="10" min="0" max="100">
                <div class="hint">Remaining ships = pure RL (conservative)</div>
            </div>
            <div class="section-title">Environment</div>
            <div class="control-group">
                <label>Port Capacity (%): <span id="capacityValue">60</span>%</label>
                <input type="range" id="capacity" min="10" max="90" value="60"
                       oninput="document.getElementById('capacityValue').textContent=this.value">
            </div>
            <div class="section-title">LLM Settings</div>
            <div class="checkbox-label">
                <input type="checkbox" id="use_llm">
                <label>Enable Ollama LLM</label>
            </div>
            <div class="control-group" style="margin-top:8px;">
                <label>LLM Timeout (s): <span id="timeoutValue">120</span></label>
                <input type="range" id="llm_timeout" min="30" max="300" value="120" step="10"
                       oninput="document.getElementById('timeoutValue').textContent=this.value">
            </div>
            <div class="section-title">Speed</div>
            <div class="control-group">
                <label>Step delay (ms): <span id="speedValue">50</span></label>
                <input type="range" id="speed" min="0" max="500" value="50"
                       oninput="document.getElementById('speedValue').textContent=this.value">
            </div>
            <button class="btn-start"   onclick="startSimulation()">&#9654; Start Single Run</button>
            <button class="btn-pause"   onclick="pauseSimulation()" disabled>&#9646;&#9646; Pause</button>
            <button class="btn-stop"    onclick="stopSimulation()"  disabled>&#9646; Stop</button>
            <button class="btn-refresh" onclick="refreshPlots()">&#8635; Refresh Plots</button>
            <button class="btn-export"  onclick="exportLLMLogs()">&#8659; Export LLM Logs</button>

            <div class="section-title">Multi-Run Experiment</div>
            <div class="control-group">
                <label>Number of Runs:</label>
                <input type="number" id="num_runs" value="10" min="2" max="50">
                <div class="hint">Each run is a full independent simulation</div>
            </div>
            <button class="btn-multirun" onclick="startMultiRun()">&#9654;&#9654; Start Multi-Run</button>
            <button class="btn-stop" id="btn-stop-multi" onclick="stopMultiRun()" disabled style="background:#c0392b;">&#9646; Stop Multi-Run</button>

            <div id="statusBadge" class="status-badge status-stopped">Stopped</div>

            <div class="metrics">
                <h3 style="margin-bottom:8px; font-size:14px;">Live Metrics</h3>
                <div class="metric-item"><span>Day:</span><span id="day">0/0</span></div>
                <div class="metric-item"><span>Efficiency:</span><span id="efficiency">0%</span></div>
                <div class="metric-item"><span>Adaptive success:</span><span id="adaptive">0%</span></div>
                <div class="metric-item"><span>Conservative success:</span><span id="conservative">0%</span></div>
                <div class="metric-item"><span>Optimal Days:</span><span id="optimal">0</span></div>
                <div class="metric-item"><span>Avg Q-Value:</span><span id="qvalue">0.000</span></div>
                <div class="metric-item"><span>LLM calls:</span><span id="llm_calls">0</span></div>
                <div class="metric-item"><span>LLM success:</span><span id="llm_success">0%</span></div>
            </div>
        </div>

        <!-- ── MAIN ── -->
        <div class="main-content">
            <div class="legend">
                <span><span class="legend-dot dot-llm"></span>LLM / Adaptive (with LLM)</span>
                <span><span class="legend-dot dot-rl"></span>Adaptive RL (no LLM)</span>
                <span><span class="legend-dot dot-cons"></span>Conservative RL</span>
            </div>
            <div class="tab-buttons">
                <button class="tab-button active" onclick="switchTab('overview',event)">&#128200; Overview</button>
                <button class="tab-button"        onclick="switchTab('detailed',event)">&#128202; Detailed</button>
                <button class="tab-button"        onclick="switchTab('persona',event)">&#129489; Agent Personas</button>
                <button class="tab-button"        onclick="switchTab('multirun',event)">&#128257; Multi-Run Analysis</button>
                <button class="tab-button"        onclick="switchTab('llm',event)">&#129302; LLM Interactions</button>
            </div>

            <!-- Overview tab -->
            <div id="overview-tab" class="tab-content active">
                <div class="plot-container">
                    <img id="performance-plot" class="plot-image" src="" alt="Performance plot">
                </div>
            </div>

            <!-- Detailed tab -->
            <div id="detailed-tab" class="tab-content">
                <div class="plot-container">
                    <img id="detailed-plot" class="plot-image" src="" alt="Detailed analysis">
                </div>
            </div>

            <!-- Persona tab -->
            <div id="persona-tab" class="tab-content">
                <div style="display:flex; gap:16px; margin-bottom:12px; flex-wrap:wrap;">
                    <button onclick="refreshPersonas()" style="width:auto; padding:8px 20px; background:#764ba2; color:white; border-radius:8px; border:none; cursor:pointer;">&#8635; Refresh Personas</button>
                    <select id="persona-filter" onchange="filterPersonas()" style="padding:8px 12px; border-radius:8px; border:2px solid #e0e0e0;">
                        <option value="all">All Agents</option>
                        <option value="adaptive">Adaptive / LLM Only</option>
                        <option value="conservative">Conservative / RL Only</option>
                        <option value="llm-true">LLM-enabled Only</option>
                    </select>
                </div>

                <div class="persona-section-label">Adaptive / LLM Agents</div>
                <div id="persona-grid-adaptive" class="persona-grid">
                    <div style="color:#888; padding:20px;">Run simulation first, then refresh personas.</div>
                </div>

                <div class="persona-section-label" style="margin-top:20px;">Conservative / RL Agents</div>
                <div id="persona-grid-conservative" class="persona-grid">
                    <div style="color:#888; padding:20px;">Run simulation first, then refresh personas.</div>
                </div>

                <div class="persona-section-label" style="margin-top:20px;">Persona Type Summary</div>
                <div id="persona-summary" style="color:#888; padding:10px;">No data yet.</div>
            </div>

            <!-- Multi-Run tab -->
            <div id="multirun-tab" class="tab-content">
                <div class="multirun-controls">
                    <h3 style="margin-bottom:10px; font-size:15px;">Multi-Run Experiment Status</h3>
                    <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
                        <div style="flex:1;">
                            <div id="multirun-status-text" style="font-size:13px; color:#555;">Not started</div>
                            <div class="multirun-progress-bar">
                                <div class="multirun-progress-fill" id="multirun-bar" style="width:0%"></div>
                            </div>
                        </div>
                        <button onclick="refreshMultiRun()" style="width:auto; padding:8px 20px; background:#17a2b8; color:white; border-radius:8px; border:none; cursor:pointer;">&#8635; Refresh</button>
                    </div>
                    <div class="progress-log" id="progress-log">Waiting to start...</div>
                </div>

                <div id="agg-stats-section" style="display:none;">
                    <h3 style="font-size:15px; margin-bottom:10px;">Aggregated Statistics (mean ± SD across runs)</h3>
                    <div class="agg-grid" id="agg-grid"></div>
                </div>

                <div id="raw-table-section" style="display:none;">
                    <h3 style="font-size:15px; margin-bottom:10px; margin-top:16px;">Raw Per-Run Data</h3>
                    <div style="overflow-x:auto;">
                        <table class="run-table" id="run-table">
                            <thead>
                                <tr>
                                    <th>Run</th>
                                    <th>Efficiency (%)</th>
                                    <th>Adap Cum Reward</th>
                                    <th>Cons Cum Reward</th>
                                    <th>Adap Success (%)</th>
                                    <th>Cons Success (%)</th>
                                    <th>Congestion (%)</th>
                                    <th>Avg Attend</th>
                                    <th>Avg Q</th>
                                    <th>Strategies</th>
                                    <th>LLM Calls</th>
                                </tr>
                            </thead>
                            <tbody id="run-table-body"></tbody>
                        </table>
                    </div>
                </div>

                <div id="multirun-plot-section" style="display:none; margin-top:20px;">
                    <h3 style="font-size:15px; margin-bottom:10px;">Multi-Run Visualizations</h3>
                    <div class="plot-container">
                        <img id="multirun-plot" class="plot-image" src="" alt="Multi-run analysis">
                    </div>
                </div>

                <div id="persona-multirun-section" style="display:none; margin-top:20px;">
                    <h3 style="font-size:15px; margin-bottom:10px;">Persona Distribution Across Runs</h3>
                    <table class="persona-table" id="persona-metric-table">
                        <thead>
                            <tr>
                                <th>Persona</th><th>Type</th><th>Count</th>
                                <th>Avg Reward</th><th>Reward SD</th>
                                <th>Go Ratio</th><th>Success Rate</th>
                                <th>Strategies</th><th>Trend</th>
                            </tr>
                        </thead>
                        <tbody id="persona-metric-tbody"></tbody>
                    </table>
                </div>
            </div>

            <!-- LLM tab -->
            <div id="llm-tab" class="tab-content">
                <div class="llm-stats">
                    <h3>LLM Performance Statistics</h3>
                    <div id="llm-stats-content">No LLM data yet.</div>
                </div>
                <div class="llm-container">
                    <table class="llm-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th><th>Agent ID</th><th>Type</th>
                                <th>Week</th><th>Day</th><th>Cap %</th>
                                <th>Prompt</th><th>Raw Response</th>
                                <th>Parsed Strategy</th><th>Fallback?</th><th>Initial Q</th>
                            </tr>
                        </thead>
                        <tbody id="llm-table-body">
                            <tr><td colspan="11" style="text-align:center;padding:20px;color:#888;">
                                No LLM interactions yet.
                            </td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
let updateInterval = null;
let llmInterval    = null;
let multiInterval  = null;
let currentTab     = 'overview';
let allPersonas    = [];

// ── Ollama status ──────────────────────────────────────────────────────────────
async function checkOllama() {
    try {
        const r = await fetch('/ollama/status');
        const d = await r.json();
        const el = document.getElementById('ollamaStatus');
        if (d.alive) {
            el.className   = 'ollama-badge ollama-ok';
            el.textContent = 'Ollama online  (model: ' + d.model + ')';
        } else {
            el.className   = 'ollama-badge ollama-err';
            el.textContent = 'Ollama offline — run: ollama serve';
        }
    } catch(e) {}
}
checkOllama();
setInterval(checkOllama, 15000);

// ── Tab switching ──────────────────────────────────────────────────────────────
function switchTab(tab, event) {
    currentTab = tab;
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    if (event && event.target) event.target.classList.add('active');
    ['overview','detailed','persona','multirun','llm'].forEach(t => {
        document.getElementById(t+'-tab').classList.remove('active');
    });
    document.getElementById(tab+'-tab').classList.add('active');
    if (llmInterval) { clearInterval(llmInterval); llmInterval = null; }
    if (multiInterval) { clearInterval(multiInterval); multiInterval = null; }
    if (tab === 'llm') {
        refreshLLMLogs();
        llmInterval = setInterval(refreshLLMLogs, 3000);
    } else if (tab === 'persona') {
        refreshPersonas();
    } else if (tab === 'multirun') {
        refreshMultiRun();
        multiInterval = setInterval(refreshMultiRun, 2000);
    } else {
        refreshPlots();
    }
}

// ── Single-run controls ────────────────────────────────────────────────────────
async function startSimulation() {
    const cfg = {
        total_ships:        parseInt(document.getElementById('total_ships').value),
        total_days:         parseInt(document.getElementById('total_days').value),
        num_adaptive_ships: parseInt(document.getElementById('adaptive_ships').value),
        capacity:           parseInt(document.getElementById('capacity').value),
        delay:              parseInt(document.getElementById('speed').value) / 1000,
        use_llm:            document.getElementById('use_llm').checked,
        llm_timeout:        parseInt(document.getElementById('llm_timeout').value),
    };
    const r = await fetch('/start', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(cfg)
    });
    if (r.ok) {
        document.querySelector('.btn-start').disabled = true;
        document.querySelector('.btn-pause').disabled = false;
        document.querySelector('.btn-stop').disabled  = false;
        setStatus('running');
        if (updateInterval) clearInterval(updateInterval);
        updateInterval = setInterval(() => { updateMetrics(); refreshPlots(); }, 1200);
    }
}

async function pauseSimulation() {
    const r = await fetch('/pause', {method:'POST'});
    if (r.ok) {
        const d   = await r.json();
        const btn = document.querySelector('.btn-pause');
        if (d.paused) { btn.textContent = 'Resume'; setStatus('paused'); }
        else           { btn.textContent = 'Pause';  setStatus('running'); }
    }
}

async function stopSimulation() {
    await fetch('/stop', {method:'POST'});
    if (updateInterval) clearInterval(updateInterval);
    document.querySelector('.btn-start').disabled = false;
    document.querySelector('.btn-pause').disabled = true;
    document.querySelector('.btn-stop').disabled  = true;
    setStatus('stopped');
}

function setStatus(s) {
    const el = document.getElementById('statusBadge');
    el.className   = 'status-badge status-' + s;
    el.textContent = s === 'running' ? 'Running' : s === 'paused' ? 'Paused' : 'Stopped';
}

// ── Plots ──────────────────────────────────────────────────────────────────────
async function refreshPlots() {
    if (currentTab !== 'overview' && currentTab !== 'detailed') return;
    const ep = currentTab === 'overview' ? '/plot/performance' : '/plot/detailed';
    const r  = await fetch(ep);
    const d  = await r.json();
    if (d.image) {
        const id = currentTab === 'overview' ? 'performance-plot' : 'detailed-plot';
        document.getElementById(id).src = 'data:image/png;base64,' + d.image;
    }
}

// ── Metrics ────────────────────────────────────────────────────────────────────
async function updateMetrics() {
    const r = await fetch('/metrics');
    const d = await r.json();
    if (!d) return;
    document.getElementById('day').textContent          = d.current_day+'/'+d.total_days;
    document.getElementById('efficiency').textContent   = (d.efficiency*100).toFixed(1)+'%';
    document.getElementById('adaptive').textContent     = (d.adaptive_success*100).toFixed(1)+'%';
    document.getElementById('conservative').textContent = (d.conservative_success*100).toFixed(1)+'%';
    document.getElementById('optimal').textContent      = d.days_with_1;
    document.getElementById('qvalue').textContent       = d.avg_q_value.toFixed(3);
    document.getElementById('llm_calls').textContent    = d.total_llm_attempts;
    document.getElementById('llm_success').textContent  = (d.llm_success_rate*100).toFixed(1)+'%';
}

// ── Persona tab ────────────────────────────────────────────────────────────────
async function refreshPersonas() {
    const r = await fetch('/personas');
    const d = await r.json();
    if (!d.personas || d.personas.length === 0) return;
    allPersonas = d.personas;
    renderPersonas(allPersonas);
    renderPersonaSummary(d.personas);
}

function renderPersonas(personas) {
    const adaptive     = personas.filter(p => p.agent_type === 'adaptive');
    const conservative = personas.filter(p => p.agent_type === 'conservative');
    document.getElementById('persona-grid-adaptive').innerHTML     = buildPersonaCards(adaptive);
    document.getElementById('persona-grid-conservative').innerHTML = buildPersonaCards(conservative);
}

function filterPersonas() {
    const f = document.getElementById('persona-filter').value;
    let filtered = allPersonas;
    if (f === 'adaptive')     filtered = allPersonas.filter(p => p.agent_type === 'adaptive');
    if (f === 'conservative') filtered = allPersonas.filter(p => p.agent_type === 'conservative');
    if (f === 'llm-true')     filtered = allPersonas.filter(p => p.use_llm);
    renderPersonas(filtered);
}

function buildPersonaCards(personas) {
    if (!personas.length) return '<div style="color:#888;padding:16px;">No agents in this group.</div>';
    return personas.map(p => {
        // Determine agent type for display
        const isLlm = p.use_llm === true;
        const isAdaptive = p.agent_type === 'adaptive';
        
        // Card styling and tag based on actual agent type
        let cardCls, tagCls, tagTxt;
        if (isLlm) {
            cardCls = 'llm';
            tagCls = 'tag-llm';
            tagTxt = 'LLM';
        } else if (isAdaptive) {
            cardCls = 'adaptive-rl';  // New class for pure RL adaptive
            tagCls = 'tag-adaptive-rl';
            tagTxt = 'Adaptive RL';
        } else {
            cardCls = 'rl';
            tagCls = 'tag-rl';
            tagTxt = 'Conservative RL';
        }
        
        const goP     = Math.round(p.go_ratio * 100);
        const stayP   = 100 - goP;
        const desc    = PERSONA_DESCRIPTIONS[p.persona] || '';
        const trendArrow = p.trend > 0.001 ? '&#8593;' : p.trend < -0.001 ? '&#8595;' : '&#8594;';
        const trendColor = p.trend > 0.001 ? '#28a745' : p.trend < -0.001 ? '#dc3545' : '#888';
        return `
        <div class="persona-card ${cardCls}">
            <div class="persona-header">
                <span class="persona-name">Agent #${p.agent_id}</span>
                <span class="persona-tag ${tagCls}">${tagTxt}</span>
            </div>
            <div class="persona-label">${esc(p.persona)}</div>
            <div class="persona-desc">${esc(desc)}</div>
            <div style="margin-bottom:8px;">
                <div style="font-size:10px;color:#888;margin-bottom:2px;">Go/Stay decision ratio</div>
                <div class="decision-bar">
                    <div class="bar-go" style="width:${goP}%"></div>
                    <div class="bar-stay" style="width:${stayP}%"></div>
                </div>
                <div class="bar-label"><span>Go ${goP}%</span><span>Stay ${stayP}%</span></div>
            </div>
            <div class="persona-metrics">
                <div class="pm-item"><div class="pm-label">Avg Reward</div><div class="pm-value">${p.avg_reward}</div></div>
                <div class="pm-item"><div class="pm-label">Reward SD</div><div class="pm-value">${p.reward_std}</div></div>
                <div class="pm-item"><div class="pm-label">Cumulative</div><div class="pm-value">${p.cumulative}</div></div>
                <div class="pm-item"><div class="pm-label">Success Rate</div><div class="pm-value">${(p.success_rate*100).toFixed(1)}%</div></div>
                <div class="pm-item"><div class="pm-label">Strategies</div><div class="pm-value">${p.strat_count}</div></div>
                <div class="pm-item"><div class="pm-label">Q Spread</div><div class="pm-value">${p.q_spread}</div></div>
                <div class="pm-item"><div class="pm-label">Best Q</div><div class="pm-value">${p.best_q}</div></div>
                <div class="pm-item"><div class="pm-label">Trend</div><div class="pm-value" style="color:${trendColor}">${trendArrow} ${p.trend}</div></div>
                ${p.use_llm ? `<div class="pm-item"><div class="pm-label">LLM Calls</div><div class="pm-value">${p.llm_attempts}</div></div>
                <div class="pm-item"><div class="pm-label">LLM Success</div><div class="pm-value">${p.llm_successes}</div></div>` : ''}
            </div>
        </div>`;
    }).join();
}

const PERSONA_DESCRIPTIONS = {
    "Bold Rusher":        "High go-ratio, volatile rewards. Attends frequently regardless of congestion.",
    "Steady Planner":     "Balanced go-ratio, low variance. Consistent, medium-risk strategy.",
    "Cautious Waiter":    "Low go-ratio, avoids congestion well. Stays home more than peers.",
    "Rigid Follower":     "Few strategies explored. Sticks to initial patterns without adapting.",
    "Adaptive Explorer":  "High strategy diversity, improving trend. LLM drives genuine exploration.",
    "Strategic Learner":  "Moderate diversity, positive reward trend. Balanced LLM guidance.",
    "Opportunist":        "High go-ratio spikes. Chases short-term gains, reward-volatile.",
    "Overcautious":       "Low go-ratio, minimal strategy growth. LLM calls rarely change behaviour.",
};

function renderPersonaSummary(personas) {
    const counts = {};
    personas.forEach(p => {
        const key = p.persona + (p.use_llm ? ' [LLM]' : ' [RL]');
        counts[key] = (counts[key] || 0) + 1;
    });
    const html = Object.entries(counts).sort((a,b) => b[1]-a[1]).map(([k,v]) =>
        `<span style="display:inline-block;background:#f0f0f0;border-radius:8px;padding:4px 10px;margin:3px;font-size:12px;"><strong>${v}</strong> ${k}</span>`
    ).join('');
    document.getElementById('persona-summary').innerHTML = html || '<span style="color:#888">No data</span>';
}

// ── Multi-Run ──────────────────────────────────────────────────────────────────
async function startMultiRun() {
    const cfg = {
        total_ships:        parseInt(document.getElementById('total_ships').value),
        total_days:         parseInt(document.getElementById('total_days').value),
        num_adaptive_ships: parseInt(document.getElementById('adaptive_ships').value),
        capacity:           parseInt(document.getElementById('capacity').value),
        delay:              0,
        use_llm:            document.getElementById('use_llm').checked,
        llm_timeout:        parseInt(document.getElementById('llm_timeout').value),
        num_runs:           parseInt(document.getElementById('num_runs').value),
    };
    const r = await fetch('/multirun/start', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(cfg)
    });
    if (r.ok) {
        document.getElementById('btn-stop-multi').disabled = false;
        if (multiInterval) clearInterval(multiInterval);
        multiInterval = setInterval(refreshMultiRun, 2000);
        switchTab('multirun', {target: document.querySelectorAll('.tab-button')[3]});
    }
}

async function stopMultiRun() {
    await fetch('/multirun/stop', {method:'POST'});
    if (multiInterval) { clearInterval(multiInterval); multiInterval = null; }
    document.getElementById('btn-stop-multi').disabled = true;
}

async function refreshMultiRun() {
    const r = await fetch('/multirun/status');
    const d = await r.json();
    if (!d) return;

    const pct = d.total_runs > 0 ? Math.round(d.runs_complete / d.total_runs * 100) : 0;
    document.getElementById('multirun-bar').style.width = pct + '%';
    document.getElementById('multirun-status-text').textContent =
        `Run ${d.current_run}/${d.total_runs} — ${d.runs_complete} complete (${pct}%)${d.is_paused ? ' [PAUSED]' : ''}`;

    const logEl = document.getElementById('progress-log');
    if (d.log && d.log.length) {
        logEl.innerHTML = d.log.map(l => '<div>' + esc(l) + '</div>').join('');
        logEl.scrollTop = logEl.scrollHeight;
    }

    if (d.runs_complete > 0) {
        loadMultiRunData();
    }
    if (!d.is_running && d.runs_complete > 0) {
        document.getElementById('btn-stop-multi').disabled = true;
        if (multiInterval) { clearInterval(multiInterval); multiInterval = null; }
    }
}

async function loadMultiRunData() {
    const r = await fetch('/multirun/data');
    const d = await r.json();
    if (!d) return;

    // Show agg stats
    if (d.agg && Object.keys(d.agg).length) {
        document.getElementById('agg-stats-section').style.display = 'block';
        renderAggStats(d.agg);
    }

    // Show raw table
    if (d.raw && d.raw.length) {
        document.getElementById('raw-table-section').style.display = 'block';
        renderRawTable(d.raw);
    }

    // Show multi-run plot
    if (d.plot_image) {
        document.getElementById('multirun-plot-section').style.display = 'block';
        document.getElementById('multirun-plot').src = 'data:image/png;base64,' + d.plot_image;
    }

    // Show persona cross-run table
    if (d.persona_metrics && d.persona_metrics.length) {
        document.getElementById('persona-multirun-section').style.display = 'block';
        renderPersonaMetricTable(d.persona_metrics);
    }
}

function renderAggStats(agg) {
    const labels = {
        'efficiency':                 'Efficiency',
        'adaptive_cum_reward':        'Adaptive Cum. Reward',
        'conservative_cum_reward':    'Conservative Cum. Reward',
        'adaptive_success_rate':      'Adaptive Success Rate',
        'conservative_success_rate':  'Conservative Success Rate',
        'congestion_rate':            'Congestion Rate',
        'avg_attendance':             'Avg Attendance',
        'avg_q_value':                'Avg Q-Value',
    };
    const scalePct = ['efficiency','adaptive_success_rate','conservative_success_rate','congestion_rate'];
    let html = '';
    for (const [key, label] of Object.entries(labels)) {
        if (!agg[key]) continue;
        const m    = agg[key];
        const mult = scalePct.includes(key) ? 100 : 1;
        const dp   = scalePct.includes(key) ? 1 : key.includes('reward') ? 1 : 3;
        const mean = (m.mean * mult).toFixed(dp);
        const sd   = (m.std  * mult).toFixed(dp);
        const cv   = m.mean !== 0 ? Math.abs(m.std / m.mean) * 100 : 0;
        const cvCls = cv < 10 ? 'cv-good' : cv < 20 ? 'cv-medium' : 'cv-high';
        const unit  = scalePct.includes(key) ? '%' : '';
        html += `<div class="agg-card">
            <div class="agg-metric">${label}</div>
            <div class="agg-mean">${mean}${unit}</div>
            <div class="agg-sd">± ${sd}${unit} SD</div>
            <div class="agg-cv ${cvCls}">CV: ${cv.toFixed(1)}% ${cv<10?'(reliable)':cv<20?'(moderate)':'(high variance)'}</div>
        </div>`;
    }
    document.getElementById('agg-grid').innerHTML = html;
}

function renderRawTable(rows) {
    const tbody = document.getElementById('run-table-body');
    // compute averages
    const fields = ['efficiency','adap_reward','cons_reward','adap_success','cons_success','congestion','avg_attend','avg_q','strategies','llm_calls'];
    const avgs   = {};
    fields.forEach(f => { avgs[f] = rows.reduce((s,r) => s + Number(r[f]||0), 0) / rows.length; });
    const sds    = {};
    fields.forEach(f => { sds[f]  = Math.sqrt(rows.reduce((s,r) => s + Math.pow(Number(r[f]||0)-avgs[f],2), 0) / rows.length); });

    let html = rows.map(r => `<tr>
        <td><strong>${r.run}</strong></td>
        <td>${r.efficiency}%</td><td>${r.adap_reward}</td><td>${r.cons_reward}</td>
        <td>${r.adap_success}%</td><td>${r.cons_success}%</td><td>${r.congestion}%</td>
        <td>${r.avg_attend}</td><td>${r.avg_q}</td><td>${r.strategies}</td><td>${r.llm_calls}</td>
    </tr>`).join('');

    // Average row
    html += `<tr class="avg-row">
        <td>AVG ± SD</td>
        <td>${avgs.efficiency.toFixed(1)} ± ${sds.efficiency.toFixed(1)}%</td>
        <td>${avgs.adap_reward.toFixed(1)} ± ${sds.adap_reward.toFixed(1)}</td>
        <td>${avgs.cons_reward.toFixed(1)} ± ${sds.cons_reward.toFixed(1)}</td>
        <td>${avgs.adap_success.toFixed(1)} ± ${sds.adap_success.toFixed(1)}%</td>
        <td>${avgs.cons_success.toFixed(1)} ± ${sds.cons_success.toFixed(1)}%</td>
        <td>${avgs.congestion.toFixed(1)} ± ${sds.congestion.toFixed(1)}%</td>
        <td>${avgs.avg_attend.toFixed(2)} ± ${sds.avg_attend.toFixed(2)}</td>
        <td>${avgs.avg_q.toFixed(3)} ± ${sds.avg_q.toFixed(3)}</td>
        <td>${avgs.strategies.toFixed(0)} ± ${sds.strategies.toFixed(0)}</td>
        <td>${avgs.llm_calls.toFixed(0)} ± ${sds.llm_calls.toFixed(0)}</td>
    </tr>`;
    tbody.innerHTML = html;
}

function renderPersonaMetricTable(rows) {
    const tbody = document.getElementById('persona-metric-tbody');
    tbody.innerHTML = rows.map(r => `<tr>
        <td><strong>${esc(r.persona)}</strong></td>
        <td><span style="padding:2px 8px;border-radius:8px;font-size:11px;background:${r.type==='LLM'?'#e3ecff':'#e3f5ec'};color:${r.type==='LLM'?'#2c5fcb':'#1a6e3c'}">${r.type}</span></td>
        <td>${r.count}</td>
        <td>${(r.avg_reward_mean||0).toFixed(3)} ± ${(r.avg_reward_std||0).toFixed(3)}</td>
        <td>${(r.reward_std_mean||0).toFixed(3)}</td>
        <td>${((r.go_ratio_mean||0)*100).toFixed(1)}%</td>
        <td>${((r.success_rate_mean||0)*100).toFixed(1)}%</td>
        <td>${(r.strat_count_mean||0).toFixed(1)}</td>
        <td>${(r.trend_mean||0).toFixed(4)}</td>
    </tr>`).join('');
}

// ── LLM logs ──────────────────────────────────────────────────────────────────
async function refreshLLMLogs() {
    try {
        const r = await fetch('/llm/logs');
        const d = await r.json();
        if (d.stats) document.getElementById('llm-stats-content').innerHTML = d.stats;
        const tbody = document.getElementById('llm-table-body');
        if (!d.rows || !d.rows.length) {
            tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;padding:20px;color:#888">No LLM interactions yet.</td></tr>';
            return;
        }
        tbody.innerHTML = '';
        d.rows.forEach(row => {
            const tr = document.createElement('tr');
            const fallbackHtml = row.fallback_used === 'True'
                ? '<span class="fallback-yes">Yes</span>'
                : '<span class="fallback-no">No</span>';
            tr.innerHTML = `
                <td>${esc(row.timestamp)}</td><td>${esc(row.agent_id)}</td><td>${esc(row.agent_type)}</td>
                <td>${esc(row.week)}</td><td>${esc(row.day)}</td><td>${esc(row.capacity_pct)}%</td>
                <td style="max-width:300px"><div class="cell-pre">${esc(row.prompt)}</div></td>
                <td style="max-width:250px"><div class="cell-pre">${esc(row.raw_llm_response)}</div></td>
                <td><div class="cell-pre">${esc(row.parsed_strategy)}</div></td>
                <td>${fallbackHtml}</td><td>${esc(row.initial_q_value)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch(e) { console.error('LLM log error:', e); }
}

function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

async function exportLLMLogs() { window.open('/llm/export','_blank'); }

refreshPlots();
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/ollama/status')
def ollama_status():
    alive = ollama_is_alive()
    return jsonify({'alive': alive, 'model': OLLAMA_MODEL, 'url': OLLAMA_URL})


@app.route('/start', methods=['POST'])
def start_simulation():
    global simulation
    try:
        config     = request.json
        simulation = SimulationEngine()
        if simulation.initialize(config):
            simulation.start_auto_step()
            return jsonify({'status': 'started'})
        return jsonify({'status': 'error'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/pause', methods=['POST'])
def pause_simulation():
    if simulation.is_running:
        simulation.resume() if simulation.is_paused else simulation.pause()
        return jsonify({'paused': simulation.is_paused})
    return jsonify({'paused': False})


@app.route('/stop', methods=['POST'])
def stop_simulation():
    simulation.stop()
    return jsonify({'status': 'stopped'})


@app.route('/metrics')
def get_metrics():
    return jsonify(simulation.get_current_metrics())


@app.route('/plot/performance')
def get_performance_plot():
    results = simulation.get_results()
    img = create_performance_plot(results)
    return jsonify({'image': img})


@app.route('/plot/detailed')
def get_detailed_plot():
    results = simulation.get_results()
    img = create_detailed_analysis_plot(results)
    return jsonify({'image': img})


@app.route('/personas')
def get_personas():
    """Return per-agent persona profiles for the current (or last) single run."""
    if not simulation.ships:
        return jsonify({'personas': []})
    profiles = simulation.get_agent_personas()
    return jsonify({'personas': profiles})


# ── Multi-run routes ──────────────────────────────────────────────────────────

@app.route('/multirun/start', methods=['POST'])
def start_multirun():
    global multi_engine
    try:
        config   = request.json
        n_runs   = int(config.pop('num_runs', 10))
        multi_engine = MultiRunEngine()
        multi_engine.initialize(config, total_runs=n_runs)
        multi_engine.start()
        return jsonify({'status': 'started', 'total_runs': n_runs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/multirun/stop', methods=['POST'])
def stop_multirun():
    multi_engine.stop()
    return jsonify({'status': 'stopped'})


@app.route('/multirun/status')
def multirun_status():
    return jsonify(multi_engine.get_progress())


@app.route('/multirun/data')
def multirun_data():
    """Return raw table, aggregated stats, plot, and persona metrics."""
    raw     = multi_engine.get_raw_table()
    agg     = multi_engine.get_aggregated_stats()
    img     = create_multirun_plot(multi_engine) if multi_engine.run_summaries else None
    persona = multi_engine.get_persona_metric_table()
    return jsonify({
        'raw':            raw,
        'agg':            agg,
        'plot_image':     img,
        'persona_metrics': persona,
    })


# ── LLM log routes ────────────────────────────────────────────────────────────

@app.route('/llm/logs')
def get_llm_logs():
    logs_df = simulation.get_llm_logs()
    if logs_df.empty:
        return jsonify({'rows': [], 'stats': '<p>No LLM interactions recorded yet.</p>'})

    rows = []
    for _, row in logs_df.iterrows():
        iq    = row.get('initial_q_value', None)
        q_str = f"{iq:.4f}" if pd.notna(iq) and isinstance(iq, (int, float)) else 'N/A'
        rows.append({
            'timestamp':        str(row.get('timestamp', '')),
            'agent_id':         str(row.get('agent_id', '')),
            'agent_type':       str(row.get('agent_type', '')),
            'week':             str(row.get('week', '')),
            'day':              str(row.get('day', '')),
            'capacity_pct':     str(row.get('capacity_pct', '')),
            'prompt':           str(row.get('prompt', '')),
            'raw_llm_response': str(row.get('raw_llm_response', '')),
            'parsed_strategy':  str(row.get('parsed_strategy', '')),
            'fallback_used':    str(row.get('fallback_used', '')),
            'initial_q_value':  q_str,
        })

    total     = len(logs_df)
    successes = int(logs_df['success'].sum()) if 'success' in logs_df.columns else total
    rate      = successes / total * 100 if total > 0 else 0
    uniq_s    = logs_df['parsed_strategy'].nunique() if 'parsed_strategy' in logs_df.columns else 0
    fallbacks = int(logs_df['fallback_used'].astype(str).str.lower().eq('true').sum()) if 'fallback_used' in logs_df.columns else 0

    stats_html = f'''
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-size:13px;">
        <div><strong>Total LLM Calls:</strong> {total} &nbsp;|&nbsp; <strong>Success:</strong> {successes} ({rate:.1f}%)</div>
        <div><strong>Unique Strategies:</strong> {uniq_s} &nbsp;|&nbsp; <strong>Fallbacks:</strong> {fallbacks}</div>
        <div><strong>Model:</strong> {html_module.escape(OLLAMA_MODEL)}</div>
    </div>'''

    return jsonify({'rows': rows, 'stats': stats_html})


@app.route('/llm/export')
def export_llm_logs():
    logs_df = simulation.get_llm_logs()
    if logs_df.empty:
        return "No LLM logs to export", 404
    out = io.StringIO()
    logs_df.to_csv(out, index=False)
    out.seek(0)
    return send_file(
        io.BytesIO(out.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='llm_interactions_log.csv'
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def find_free_port(start: int = 5000, attempts: int = 10) -> int:
    for port in range(start, start + attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return start


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == '__main__':
    port = WEB_CONFIG['port']
    if WEB_CONFIG.get('auto_find_port', True):
        port = find_free_port(port)
    print("=" * 60)
    print("MARINE PORT OPTIMIZATION  –  WEB GUI + OLLAMA LLM")
    print("=" * 60)
    print(f"\n  http://{get_local_ip()}:{port}")
    print(f"  Ollama model : {OLLAMA_MODEL}")
    print(f"  Ollama URL   : {OLLAMA_URL}")
    print("\n  Make sure Ollama is running:  ollama serve")
    print(f"  And the model is pulled   :  ollama pull {OLLAMA_MODEL}")
    print("=" * 60)
    app.run(host=WEB_CONFIG['host'], port=port, debug=WEB_CONFIG['debug'])