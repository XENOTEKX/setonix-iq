#!/usr/bin/env python3
"""
Setonix Agent Dashboard Generator

Generates a self-contained HTML dashboard with all data embedded.
Output goes to docs/index.html for GitHub Pages deployment.

Why not a server? Setonix (Pawsey HPC) disables SSH TCP forwarding
on shared login nodes for security, so port forwarding doesn't work.

Usage:
  ./serve.py              # generate docs/index.html
"""

import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(SCRIPT_DIR, 'website')
API_DIR = os.path.join(WEBSITE_DIR, 'api')
DOCS_DIR = os.path.join(SCRIPT_DIR, 'docs')

sys.path.insert(0, API_DIR)
from data import get_all_runs


def generate():
    """Generate self-contained dashboard HTML with embedded data."""
    runs = get_all_runs()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Write JSON for reference
    json_path = os.path.join(API_DIR, 'runs.json')
    with open(json_path, 'w') as f:
        json.dump(runs, f, indent=2)

    # Read the template
    template_path = os.path.join(WEBSITE_DIR, 'index.html')
    with open(template_path) as f:
        template = f.read()

    # Find the <script> block and replace the data loading
    # We'll inject DATA right at the start of the script and replace loadData
    data_json = json.dumps(runs)

    # Build the replacement script block
    old_script_start = '// ============ State ============'
    old_script_end = 'loadData();'

    start_idx = template.index(old_script_start)
    end_idx = template.index(old_script_end) + len(old_script_end)

    new_script = build_script(data_json, now)

    output_html = template[:start_idx] + new_script + template[end_idx:]

    # Remove flamegraph img that references local file
    output_html = output_html.replace(
        '<img src="assets/flamegraph.svg" alt="CPU Flamegraph" style="width:100%;min-width:800px;" onerror="this.parentElement.innerHTML=\'<p style=color:var(--text3)>No flamegraph available yet. Run profiling first.</p>\'">',
        '<p style="color:var(--text3)">Flamegraph not available in static view. Run profiling on Setonix to generate.</p>'
    )

    os.makedirs(DOCS_DIR, exist_ok=True)
    output_path = os.path.join(DOCS_DIR, 'index.html')
    with open(output_path, 'w') as f:
        f.write(output_html)

    # Also write to root for convenience
    root_path = os.path.join(SCRIPT_DIR, 'dashboard.html')
    with open(root_path, 'w') as f:
        f.write(output_html)

    print("Generated: %s" % output_path)
    print("Also: %s" % root_path)
    print("Runs: %d | Generated: %s" % (len(runs), now))
    return output_path


def build_script(data_json, generated_time):
    """Build the complete JavaScript block with embedded data."""
    return '''// ============ State ============
var DATA = %s;
var currentRunIdx = DATA.length - 1;
var charts = {};
var timingChartType = 'bar';

// ============ Navigation ============
document.querySelectorAll('.sidebar nav a').forEach(function(a) {
  a.addEventListener('click', function(e) {
    e.preventDefault();
    var page = a.dataset.page;
    document.querySelectorAll('.sidebar nav a').forEach(function(x) { x.classList.remove('active'); });
    a.classList.add('active');
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.getElementById('page-' + page).classList.add('active');
  });
});

// ============ Data Loading ============
function loadData() {
  if (DATA.length === 0) {
    document.getElementById('overviewSubtitle').textContent = 'No pipeline runs found. Run the pipeline first.';
    return;
  }
  currentRunIdx = DATA.length - 1;
  populateRunSelector();
  renderAll();
  document.getElementById('serverDot').style.background = 'var(--green)';
  document.getElementById('serverStatus').textContent = 'Static (GitHub Pages)';
  document.getElementById('lastUpdate').textContent = 'Generated: %s';
}

function populateRunSelector() {
  var sel = document.getElementById('runSelector');
  sel.innerHTML = DATA.map(function(r, i) {
    return '<option value="' + i + '"' + (i === currentRunIdx ? ' selected' : '') + '>Run ' + r.run_id + '</option>';
  }).join('');
}

function switchRun(idx) {
  currentRunIdx = parseInt(idx);
  renderAll();
}

function refreshData() { loadData(); }

// ============ Rendering ============
function renderAll() {
  var run = DATA[currentRunIdx];
  renderOverview(run);
  renderTests(run);
  renderProfiling(run);
  renderGPU(run);
  renderAllocation();
  renderEnvironment(run);
}

function renderOverview(run) {
  var s = run.summary;
  document.getElementById('overviewSubtitle').textContent =
    'Run ' + run.run_id + ' | ' + (run.env.date || 'N/A') + ' | ' + (run.env.hostname || 'N/A');

  var badge = document.getElementById('statusBadge');
  badge.textContent = s.all_pass ? 'ALL PASS' : s.fail + ' FAILED';
  badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');

  var profile = (run.profile && run.profile.metrics) || {};
  document.getElementById('statsGrid').innerHTML =
    '<div class="stat-card">' +
      '<div class="label">Tests</div>' +
      '<div class="value" style="color:' + (s.all_pass ? 'var(--green)' : 'var(--red)') + '">' + s.pass + '/' + (s.pass + s.fail) + '</div>' +
      '<div class="change">' + (s.all_pass ? 'All passing' : s.fail + ' failed') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Pipeline Time</div>' +
      '<div class="value">' + s.total_time + 's</div>' +
      '<div class="change">' + run.timing.length + ' test commands</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">IPC</div>' +
      '<div class="value">' + (profile.IPC || 'N/A') + '</div>' +
      '<div class="change">Instructions per cycle</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">CPU</div>' +
      '<div class="value">' + ((run.env.cpu || 'N/A').replace(' 64-Core Processor', '')) + '</div>' +
      '<div class="change">' + (run.env.cores || '?') + ' cores allocated</div>' +
    '</div>';

  renderTimingChart(run);
  renderTrendChart();
}

function renderTimingChart(run) {
  if (charts.timing) charts.timing.destroy();
  var ctx = document.getElementById('timingChart').getContext('2d');
  var colors = ['#3b82f6','#8b5cf6','#06b6d4','#22c55e','#eab308','#f97316','#ef4444','#ec4899','#6366f1','#14b8a6'];
  charts.timing = new Chart(ctx, {
    type: timingChartType,
    data: {
      labels: run.timing.map(function(t) { return t.command.substring(0, 45); }),
      datasets: [{
        label: 'Time (s)',
        data: run.timing.map(function(t) { return t.time_s; }),
        backgroundColor: run.timing.map(function(_, i) { return colors[i %% colors.length] + '99'; }),
        borderColor: run.timing.map(function(_, i) { return colors[i %% colors.length]; }),
        borderWidth: 1, borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: timingChartType === 'bar' ? 'y' : undefined,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, titleColor: '#e2e8f0', bodyColor: '#94a3b8', cornerRadius: 8, padding: 12 }
      },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
      }
    }
  });
}

function renderTrendChart() {
  if (charts.trend) charts.trend.destroy();
  var ctx = document.getElementById('trendChart').getContext('2d');
  charts.trend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: DATA.map(function(r) { return r.run_id.substring(0, 10); }),
      datasets: [{
        label: 'Total Time (s)',
        data: DATA.map(function(r) { return r.summary.total_time; }),
        borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)',
        fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#22c55e',
        pointBorderColor: '#1a2332', pointBorderWidth: 2
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, cornerRadius: 8 } },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, beginAtZero: true }
      }
    }
  });
}

function toggleChartType() {
  timingChartType = timingChartType === 'bar' ? 'doughnut' : 'bar';
  renderTimingChart(DATA[currentRunIdx]);
}

function renderTests(run) {
  var s = run.summary;
  var badge = document.getElementById('testsBadge');
  badge.textContent = s.pass + '/' + (s.pass + s.fail) + ' passing';
  badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');

  var tbody = document.querySelector('#testsTable tbody');
  tbody.innerHTML = run.verify.map(function(v) {
    return '<tr data-status="' + v.status + '">' +
      '<td class="status-icon">' + (v.status === 'pass' ? '\\u2705' : '\\u274c') + '</td>' +
      '<td style="font-family:monospace;font-size:0.8rem">' + v.file + '</td>' +
      '<td>' + v.expected + '</td><td>' + v.reported + '</td>' +
      '<td style="color:' + (v.diff < 0.01 ? 'var(--green)' : 'var(--yellow)') + '">' + v.diff + '</td></tr>';
  }).join('');

  var totalTime = run.timing.reduce(function(a, t) { return a + t.time_s; }, 0);
  var ttbody = document.querySelector('#timingTable tbody');
  ttbody.innerHTML = run.timing.map(function(t, i) {
    var pct = ((t.time_s / totalTime) * 100).toFixed(1);
    return '<tr><td>' + (i+1) + '</td>' +
      '<td style="font-family:monospace;font-size:0.8rem">' + t.command + '</td>' +
      '<td>' + t.time_s.toFixed(3) + '</td><td>' + pct + '%%</td>' +
      '<td><div style="width:' + pct + '%%;height:6px;background:var(--accent);border-radius:3px;min-width:2px"></div></td></tr>';
  }).join('');
}

function filterTests(status, btn) {
  document.querySelectorAll('#page-tests .actions .btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  document.querySelectorAll('#testsTable tbody tr').forEach(function(tr) {
    tr.style.display = (status === 'all' || tr.dataset.status === status) ? '' : 'none';
  });
}

function renderProfiling(run) {
  var m = (run.profile && run.profile.metrics) || {};
  var grid = document.getElementById('profileGrid');
  var items = [
    { val: m.IPC || 'N/A', lbl: 'IPC', color: '#3b82f6' },
    { val: m['cache-miss-rate'] ? m['cache-miss-rate'] + '%%' : 'N/A', lbl: 'Cache Miss Rate', color: '#eab308' },
    { val: m['L1-dcache-miss-rate'] ? m['L1-dcache-miss-rate'] + '%%' : 'N/A', lbl: 'L1 D-Cache Miss', color: '#f97316' },
    { val: m['branch-miss-rate'] ? m['branch-miss-rate'] + '%%' : 'N/A', lbl: 'Branch Mispredict', color: '#22c55e' },
    { val: m.instructions ? (m.instructions / 1e9).toFixed(1) + 'B' : 'N/A', lbl: 'Instructions', color: '#8b5cf6' }
  ];
  grid.innerHTML = items.map(function(it) {
    return '<div class="profile-metric"><div class="val" style="color:' + it.color + '">' + it.val + '</div><div class="lbl">' + it.lbl + '</div></div>';
  }).join('');

  if (!m.cycles) return;
  if (charts.counter) charts.counter.destroy();
  var counterCtx = document.getElementById('counterChart').getContext('2d');
  var counterKeys = ['cycles','instructions','cache-references','cache-misses','branch-instructions','branch-misses','L1-dcache-loads','L1-dcache-load-misses'];
  charts.counter = new Chart(counterCtx, {
    type: 'bar',
    data: {
      labels: counterKeys.map(function(k) { return k.replace('L1-dcache-','L1-').replace('-instructions','-instr').replace('-references','-refs'); }),
      datasets: [{ data: counterKeys.map(function(k) { return m[k] || 0; }), backgroundColor: '#3b82f688', borderColor: '#3b82f6', borderWidth: 1, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8', maxRotation: 45, font: { size: 10 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, type: 'logarithmic' }
      }
    }
  });
}

function renderGPU(run) {
  var gpu = run.gpu_info || '';
  document.getElementById('gpuRaw').textContent = gpu || 'No GPU data available';
  // Parse GPU stats from rocm-smi table
  var lines = gpu.split('\\n');
  var temp = 'N/A', power = 'N/A', vram = 'N/A', usage = 'N/A';
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    if (/^0\\s+/.test(line)) {
      var parts = line.trim().split(/\\s+/);
      if (parts.length >= 14) {
        temp = parts[4]; power = parts[5]; vram = parts[13]; usage = parts[14] || '0';
      }
    }
  }
  document.getElementById('gpuGrid').innerHTML =
    '<div class="gpu-stat"><div class="val" style="color:var(--green)">' + temp + '</div><div class="lbl">Temperature</div></div>' +
    '<div class="gpu-stat"><div class="val" style="color:var(--accent)">' + power + '</div><div class="lbl">Power (Avg)</div></div>' +
    '<div class="gpu-stat"><div class="val" style="color:var(--text3)">' + usage + '</div><div class="lbl">GPU Utilization</div></div>' +
    '<div class="gpu-stat"><div class="val" style="color:var(--accent2)">' + vram + '</div><div class="lbl">VRAM Used</div></div>';
}

function renderAllocation() {
  document.getElementById('allocGrid').innerHTML =
    '<div class="card"><div class="card-header"><h2>CPU Allocation (pawsey1351)</h2></div>' +
    '<div style="display:flex;justify-content:space-between;margin-top:8px;"><span>0 kSU used</span><span style="color:var(--text3)">30,000 kSU total</span></div>' +
    '<div class="alloc-bar"><div class="fill" style="width:0%%;background:var(--green)"></div></div>' +
    '<div style="color:var(--green);font-size:0.85rem;margin-top:4px;">30,000 kSU remaining (100%%)</div></div>' +
    '<div class="card"><div class="card-header"><h2>GPU Allocation (pawsey1351-gpu)</h2></div>' +
    '<div style="display:flex;justify-content:space-between;margin-top:8px;"><span>21 kSU used</span><span style="color:var(--text3)">30,000 kSU total</span></div>' +
    '<div class="alloc-bar"><div class="fill" style="width:0.07%%;background:var(--accent);min-width:2px;"></div></div>' +
    '<div style="color:var(--green);font-size:0.85rem;margin-top:4px;">29,979 kSU remaining (99.9%%)</div></div>';
}

function renderEnvironment(run) {
  var tbody = document.querySelector('#envTable tbody');
  var env = run.env || {};
  tbody.innerHTML = Object.keys(env).map(function(k) {
    return '<tr><td style="font-weight:600;width:200px;">' + k + '</td><td style="font-family:monospace;font-size:0.85rem;">' + env[k] + '</td></tr>';
  }).join('');
}

// ============ Init ============
loadData();''' % (data_json, generated_time)


def main():
    generate()


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
