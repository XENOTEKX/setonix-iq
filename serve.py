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
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs')

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

    # Export to logs/ so offline/local renders still have data
    os.makedirs(LOGS_DIR, exist_ok=True)
    logs_json = os.path.join(LOGS_DIR, 'runs.json')
    with open(logs_json, 'w') as f:
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
    end_idx = template.rindex(old_script_end) + len(old_script_end)

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
var expandedRunId = null;

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

// ============ Clipboard Helper ============
function copyText(text, btn) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function() {
      var orig = btn.textContent;
      btn.textContent = 'Copied!'; btn.classList.add('copied');
      setTimeout(function() { btn.textContent = orig; btn.classList.remove('copied'); }, 1500);
    }).catch(function() { fallbackCopy(text, btn); });
  } else { fallbackCopy(text, btn); }
}
function fallbackCopy(text, btn) {
  var ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.left = '-9999px';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); btn.textContent = 'Copied!'; btn.classList.add('copied');
    setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
  } catch(e) {}
  document.body.removeChild(ta);
}

function copyAllCmds(btn) {
  var run = DATA[currentRunIdx];
  var script = '#!/bin/bash\\n# Pipeline commands from Run ' + run.run_id + '\\n# ' + (run.env.date || '') + ' on ' + (run.env.hostname || '') + '\\n\\n' +
    run.timing.map(function(t) { return t.command; }).join('\\n') + '\\n';
  copyText(script, btn);
}

function copySingleCmd(text, btn) { copyText(text, btn); }

// ============ Formatting ============
function fmtTime(s) {
  if (s >= 60) return (s / 60).toFixed(1) + 'm';
  return s.toFixed(2) + 's';
}
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escAttr(s) { return s.replace(/\\\\/g,'\\\\\\\\').replace(/'/g,"\\\\'").replace(/"/g,'&quot;'); }

// ============ Data Loading ============
function loadData() {
  if (DATA.length === 0) {
    document.getElementById('overviewSubtitle').textContent = 'No pipeline runs found. Run the pipeline first.';
    return;
  }
  currentRunIdx = DATA.length - 1;
  populateAllSelectors();
  renderAll();
  renderRunsList();
  document.getElementById('serverDot').style.background = 'var(--green)';
  document.getElementById('serverStatus').textContent = 'Static (GitHub Pages)';
  document.getElementById('lastUpdate').textContent = 'Generated: %s';
}

function populateAllSelectors() {
  var selectors = ['runSelector', 'testRunSelector', 'profRunSelector', 'gpuRunSelector', 'envRunSelector'];
  selectors.forEach(function(id) {
    var sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = DATA.map(function(r, i) {
      return '<option value="' + i + '"' + (i === currentRunIdx ? ' selected' : '') + '>Run ' + r.run_id + '</option>';
    }).join('');
  });
}

function switchRun(idx) {
  currentRunIdx = parseInt(idx);
  populateAllSelectors();
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
  renderQuickCmds(run);
}

// ============ Overview ============
function renderOverview(run) {
  var s = run.summary;
  var profile = (run.profile && run.profile.metrics) || {};

  document.getElementById('overviewSubtitle').textContent =
    'Run ' + run.run_id + ' | ' + (run.env.date || 'N/A') + ' | ' + (run.env.hostname || 'N/A');

  var badge = document.getElementById('statusBadge');
  badge.textContent = s.all_pass ? 'ALL PASS' : s.fail + ' FAILED';
  badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');

  var bestTime = Math.min.apply(null, DATA.map(function(r) { return r.summary.total_time; }));
  var avgTime = (DATA.reduce(function(a, r) { return a + r.summary.total_time; }, 0) / DATA.length).toFixed(1);
  var allPass = DATA.every(function(r) { return r.summary.all_pass; });

  document.getElementById('statsGrid').innerHTML =
    '<div class="stat-card">' +
      '<div class="label">Tests</div>' +
      '<div class="value" style="color:' + (s.all_pass ? 'var(--green)' : 'var(--red)') + '">' + s.pass + '/' + (s.pass + s.fail) + '</div>' +
      '<div class="change">' + (s.all_pass ? 'All passing' : s.fail + ' failed') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Pipeline Time</div>' +
      '<div class="value">' + fmtTime(s.total_time) + '</div>' +
      '<div class="change">' + run.timing.length + ' commands | Best: ' + fmtTime(bestTime) + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">IPC</div>' +
      '<div class="value">' + (profile.IPC || 'N/A') + '</div>' +
      '<div class="change">Instructions per cycle</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Total Runs</div>' +
      '<div class="value">' + DATA.length + '</div>' +
      '<div class="change">' + (allPass ? 'All green' : 'Some failures') + ' | Avg: ' + avgTime + 's</div>' +
    '</div>';

  renderTimingChart(run);
  renderTrendChart();
}

function renderQuickCmds(run) {
  var el = document.getElementById('quickCmds');
  el.innerHTML = run.timing.map(function(t, i) {
    return '<div class="cmd-block">' +
      '<span class="cmd-num">' + (i + 1) + '</span>' +
      '<span class="cmd-text">' + escHtml(t.command) + '</span>' +
      '<span class="cmd-time">' + t.time_s.toFixed(3) + 's</span>' +
      '<button class="cmd-copy" onclick="copySingleCmd(\\'' + escAttr(t.command) + '\\', this)">Copy</button>' +
    '</div>';
  }).join('');
}

// ============ All Runs List ============
function renderRunsList() {
  renderAllRunsStats();
  filterRuns();
}

function renderAllRunsStats() {
  var total = DATA.length;
  var passing = DATA.filter(function(r) { return r.summary.all_pass; }).length;
  var failing = total - passing;
  var bestTime = Math.min.apply(null, DATA.map(function(r) { return r.summary.total_time; }));
  var avgTime = (DATA.reduce(function(a, r) { return a + r.summary.total_time; }, 0) / total).toFixed(1);
  var ipcs = DATA.map(function(r) { return (r.profile && r.profile.metrics && r.profile.metrics.IPC) || 0; });
  var bestIPC = Math.max.apply(null, ipcs);
  var totalCmds = DATA.reduce(function(a, r) { return a + r.timing.length; }, 0);

  document.getElementById('allRunsStats').innerHTML =
    '<div class="stat-card">' +
      '<div class="label">Total Runs</div>' +
      '<div class="value">' + total + '</div>' +
      '<div class="change">' + totalCmds + ' total commands</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Passing</div>' +
      '<div class="value" style="color:var(--green)">' + passing + '</div>' +
      '<div class="change">' + (failing > 0 ? failing + ' with failures' : 'All green') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Best Time</div>' +
      '<div class="value" style="color:var(--accent)">' + fmtTime(bestTime) + '</div>' +
      '<div class="change">Avg: ' + avgTime + 's</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Peak IPC</div>' +
      '<div class="value" style="color:var(--accent2)">' + (bestIPC > 0 ? bestIPC.toFixed(3) : 'N/A') + '</div>' +
      '<div class="change">Instructions per cycle</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Hosts</div>' +
      '<div class="value">' + uniqueHosts().length + '</div>' +
      '<div class="change">' + (uniqueHosts().join(', ') || 'N/A') + '</div>' +
    '</div>';
}

function uniqueHosts() {
  var seen = {};
  var result = [];
  DATA.forEach(function(r) {
    var h = r.env.hostname;
    if (h && !seen[h]) { seen[h] = true; result.push(h); }
  });
  return result;
}

function filterRuns() {
  var query = (document.getElementById('runSearch') ? document.getElementById('runSearch').value : '').toLowerCase();
  var status = document.getElementById('runStatusFilter') ? document.getElementById('runStatusFilter').value : 'all';
  var sort = document.getElementById('runSortBy') ? document.getElementById('runSortBy').value : 'date-desc';

  var runs = DATA.map(function(r, i) { return { run: r, idx: i }; });

  if (query) {
    runs = runs.filter(function(item) {
      var r = item.run;
      var haystack = [
        r.run_id, r.env.hostname || '', r.env.date || '', r.env.cpu || '', r.env.gcc || '', r.env.rocm || ''
      ].concat(r.timing.map(function(t) { return t.command; })).join(' ').toLowerCase();
      return haystack.indexOf(query) !== -1;
    });
  }

  if (status === 'pass') runs = runs.filter(function(item) { return item.run.summary.all_pass; });
  if (status === 'fail') runs = runs.filter(function(item) { return !item.run.summary.all_pass; });

  runs.sort(function(a, b) {
    switch(sort) {
      case 'date-asc': return (a.run.env.date || '').localeCompare(b.run.env.date || '');
      case 'time-asc': return a.run.summary.total_time - b.run.summary.total_time;
      case 'time-desc': return b.run.summary.total_time - a.run.summary.total_time;
      case 'ipc-desc':
        var aIPC = (a.run.profile && a.run.profile.metrics && a.run.profile.metrics.IPC) || 0;
        var bIPC = (b.run.profile && b.run.profile.metrics && b.run.profile.metrics.IPC) || 0;
        return bIPC - aIPC;
      default: return (b.run.env.date || '').localeCompare(a.run.env.date || '');
    }
  });

  document.getElementById('runCount').textContent = 'Showing ' + runs.length + ' of ' + DATA.length + ' runs';

  var listEl = document.getElementById('runsList');
  listEl.innerHTML = runs.map(function(item, rank) {
    var r = item.run;
    var s = r.summary;
    var m = (r.profile && r.profile.metrics) || {};
    var rankClass = rank === 0 ? 'gold' : rank === 1 ? 'silver' : rank === 2 ? 'bronze' : '';
    var isExpanded = expandedRunId === r.run_id;
    var cpuShort = (r.env.cpu || '').replace(' 64-Core Processor', '');

    return '<div class="run-row ' + (isExpanded ? 'active' : '') + '" id="run-' + r.run_id + '">' +
      '<div class="run-row-summary" onclick="toggleRunDetail(\\'' + r.run_id + '\\', ' + item.idx + ')">' +
        '<div class="rank ' + rankClass + '">#' + (rank + 1) + '</div>' +
        '<div class="run-info">' +
          '<div class="run-id">' + r.run_id + '</div>' +
          '<div class="run-meta">' + (r.env.date || 'N/A') + ' &middot; ' + (r.env.hostname || 'N/A') + ' &middot; ' + (r.env.cores || '?') + ' cores &middot; ' + cpuShort + '</div>' +
        '</div>' +
        '<div class="run-time">' + fmtTime(s.total_time) + '</div>' +
        '<div class="run-tests"><span class="badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail') + '">' + s.pass + '/' + (s.pass + s.fail) + '</span></div>' +
        '<div class="run-ipc">' + (m.IPC || '\\u2014') + '</div>' +
        '<div class="run-cores">' + (r.env.cores || '?') + 'C</div>' +
        '<button class="run-detail-btn" onclick="event.stopPropagation(); toggleRunDetail(\\'' + r.run_id + '\\', ' + item.idx + ')">' + (isExpanded ? 'Hide' : 'Details') + '</button>' +
      '</div>' +
      '<div class="run-detail ' + (isExpanded ? 'open' : '') + '" id="detail-' + r.run_id + '">' +
        (isExpanded ? buildRunDetail(r) : '') +
      '</div>' +
    '</div>';
  }).join('');
}

function toggleRunDetail(runId, runIdx) {
  if (expandedRunId === runId) {
    expandedRunId = null;
  } else {
    expandedRunId = runId;
  }
  filterRuns();
}

function buildRunDetail(run) {
  var s = run.summary;
  var m = (run.profile && run.profile.metrics) || {};
  var env = run.env || {};

  var cmdsHtml = run.timing.map(function(t, i) {
    return '<div class="cmd-block">' +
      '<span class="cmd-num">' + (i + 1) + '</span>' +
      '<span class="cmd-text">' + escHtml(t.command) + '</span>' +
      '<span class="cmd-time">' + t.time_s.toFixed(3) + 's</span>' +
      '<button class="cmd-copy" onclick="event.stopPropagation(); copySingleCmd(\\'' + escAttr(t.command) + '\\', this)">Copy</button>' +
    '</div>';
  }).join('');

  var verifyHtml;
  if (run.verify.length > 0) {
    verifyHtml = '<table class="verify-mini">' +
      '<thead><tr><th>Status</th><th>File</th><th>Expected</th><th>Reported</th><th>Diff</th></tr></thead>' +
      '<tbody>' + run.verify.map(function(v) {
        return '<tr>' +
          '<td>' + (v.status === 'pass' ? '\\u2705' : '\\u274c') + '</td>' +
          '<td style="font-family:monospace;font-size:0.8rem">' + v.file + '</td>' +
          '<td>' + v.expected + '</td>' +
          '<td>' + v.reported + '</td>' +
          '<td style="color:' + (v.diff < 0.01 ? 'var(--green)' : 'var(--yellow)') + '">' + v.diff + '</td>' +
        '</tr>';
      }).join('') + '</tbody></table>';
  } else {
    verifyHtml = '<p style="color:var(--text3)">No verification data</p>';
  }

  var envKeys = Object.keys(env);
  var envHtml = envKeys.map(function(k) {
    return '<div class="detail-kv">' +
      '<div class="dk-label">' + k + '</div>' +
      '<div class="dk-value">' + env[k] + '</div>' +
    '</div>';
  }).join('');

  var profileHtml;
  if (m.IPC) {
    profileHtml = '<div class="detail-grid">' +
      '<div class="detail-kv"><div class="dk-label">IPC</div><div class="dk-value" style="color:var(--accent)">' + m.IPC + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Cache Miss Rate</div><div class="dk-value" style="color:var(--yellow)">' + (m['cache-miss-rate'] || 'N/A') + '%%</div></div>' +
      '<div class="detail-kv"><div class="dk-label">L1 D-Cache Miss</div><div class="dk-value" style="color:#f97316">' + (m['L1-dcache-miss-rate'] || 'N/A') + '%%</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Branch Mispredict</div><div class="dk-value" style="color:var(--green)">' + (m['branch-miss-rate'] || 'N/A') + '%%</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Instructions</div><div class="dk-value" style="color:var(--accent2)">' + (m.instructions ? (m.instructions / 1e9).toFixed(1) + 'B' : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Cycles</div><div class="dk-value">' + (m.cycles ? (m.cycles / 1e9).toFixed(2) + 'B' : 'N/A') + '</div></div>' +
    '</div>';
  } else {
    profileHtml = '<p style="color:var(--text3)">No profiling data</p>';
  }

  return '<div class="detail-section">' +
      '<h3>Commands (' + run.timing.length + ')</h3>' +
      '<div style="margin-bottom:8px">' +
        '<button class="copy-all-btn" onclick="event.stopPropagation(); copyRunCmds(\\'' + run.run_id + '\\', this)">' +
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>' +
          'Copy All as Script' +
        '</button>' +
      '</div>' +
      cmdsHtml +
    '</div>' +
    '<div class="detail-section">' +
      '<h3>Verification (' + s.pass + ' pass, ' + s.fail + ' fail)</h3>' +
      verifyHtml +
    '</div>' +
    '<div class="detail-section">' +
      '<h3>Performance Counters</h3>' +
      profileHtml +
    '</div>' +
    '<div class="detail-section">' +
      '<h3>Environment</h3>' +
      '<div class="detail-grid">' + envHtml + '</div>' +
    '</div>';
}

function copyRunCmds(runId, btn) {
  var run = null;
  for (var i = 0; i < DATA.length; i++) {
    if (DATA[i].run_id === runId) { run = DATA[i]; break; }
  }
  if (!run) return;
  var script = '#!/bin/bash\\n# Pipeline commands from Run ' + run.run_id + '\\n# ' + (run.env.date || '') + ' on ' + (run.env.hostname || '') + '\\n\\n' +
    run.timing.map(function(t) { return t.command; }).join('\\n') + '\\n';
  copyText(script, btn);
}

// ============ Charts ============
function renderTimingChart(run) {
  if (charts.timing) charts.timing.destroy();
  var ctx = document.getElementById('timingChart').getContext('2d');
  var colors = ['#3b82f6','#8b5cf6','#06b6d4','#22c55e','#eab308','#f97316','#ef4444','#ec4899','#6366f1','#14b8a6'];
  charts.timing = new Chart(ctx, {
    type: timingChartType,
    data: {
      labels: run.timing.map(function(t) { return t.command.substring(0, 50); }),
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

// ============ Tests ============
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
      '<td style="font-family:monospace;font-size:0.8rem">' + escHtml(t.command) + '</td>' +
      '<td>' + t.time_s.toFixed(3) + '</td><td>' + pct + '%%</td>' +
      '<td><div style="width:' + pct + '%%;height:6px;background:var(--accent);border-radius:3px;min-width:2px"></div></td>' +
      '<td><button class="cmd-copy" onclick="copySingleCmd(\\'' + escAttr(t.command) + '\\', this)">Copy</button></td></tr>';
  }).join('');
}

function filterTests(status, btn) {
  document.querySelectorAll('#page-tests .actions .btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  document.querySelectorAll('#testsTable tbody tr').forEach(function(tr) {
    tr.style.display = (status === 'all' || tr.dataset.status === status) ? '' : 'none';
  });
}

// ============ Profiling ============
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

  if (charts.ipcTrend) charts.ipcTrend.destroy();
  var ipcCtx = document.getElementById('ipcTrendChart').getContext('2d');
  charts.ipcTrend = new Chart(ipcCtx, {
    type: 'line',
    data: {
      labels: DATA.map(function(r) { return r.run_id.substring(0, 10); }),
      datasets: [{
        label: 'IPC',
        data: DATA.map(function(r) { return (r.profile && r.profile.metrics && r.profile.metrics.IPC) || null; }),
        borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)',
        fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#8b5cf6',
        pointBorderColor: '#1a2332', pointBorderWidth: 2, spanGaps: true
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

// ============ GPU ============
function renderGPU(run) {
  var gpu = run.gpu_info || '';
  document.getElementById('gpuRaw').textContent = gpu || 'No GPU data available';
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

// ============ Allocation ============
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

// ============ Environment ============
function renderEnvironment(run) {
  var tbody = document.querySelector('#envTable tbody');
  var env = run.env || {};
  var keys = Object.keys(env);
  tbody.innerHTML = keys.map(function(k) {
    return '<tr><td style="font-weight:600;width:200px;">' + k + '</td><td style="font-family:monospace;font-size:0.85rem;">' + env[k] + '</td></tr>';
  }).join('');
}

// ============ Init ============
loadData();''' % (data_json, generated_time)


def main():
    generate()


if __name__ == '__main__':
    main()
