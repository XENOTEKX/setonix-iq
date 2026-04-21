// web/js/pages/overview.js — v2 (insight-oriented)

import { store } from '../state.js';
import { loadRun } from '../data.js';
import { mountRunPicker } from '../components/run-picker.js';
import { bindCopyButtons } from '../components/copy-button.js';
import * as scaling from '../charts/scaling.js';
import * as efficiency from '../charts/efficiency.js';
import * as ipcScaling from '../charts/ipc-scaling.js';
import * as perfMatrix from '../charts/performance-matrix.js';
import { escHtml, fmtTime, fmtNum } from '../utils.js';

const TMPL = `
  <div class="page-header">
    <div>
      <h1>Pipeline Overview</h1>
      <div class="subtitle" id="ovSubtitle">Insight dashboard for IQ-TREE runs on Setonix</div>
    </div>
    <span class="badge badge-info" id="ovBadge">—</span>
  </div>

  <div class="stats-grid" id="ovStats"></div>

  <div class="run-picker-wrap" style="margin-bottom:22px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <span style="font-size:0.72rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Selected run</span>
      <span style="font-size:0.68rem; color:var(--text-muted);">Search · ↑↓ navigate · ↵ select · Esc close</span>
    </div>
    <div id="ovRunPicker"></div>
  </div>

  <section id="ovBestRuns" class="best-runs" aria-label="Top runs"></section>

  <div style="font-size:0.72rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.12em; font-weight:700; margin:8px 0 10px;">Datasets</div>
  <section id="ovDatasets" class="ds-grid" aria-label="Dataset profiles"></section>

  <div class="card">
    <div class="card-header"><h2>IQ-TREE Configuration · <span style="color:var(--text3); font-weight:500; font-size:0.78rem;" id="ovConfigRunId">—</span></h2>
      <div class="actions"><button class="copy-btn" data-copy="#ovConfigText">Copy config</button></div>
    </div>
    <div class="card-body" id="ovConfig"></div>
    <pre id="ovConfigText" class="sr-only"></pre>
  </div>

  <div class="charts-row">
    <div class="card">
      <div class="card-header"><h2>Thread Scaling</h2><div class="actions"><span class="btn-sm">wall vs threads (log–log)</span></div></div>
      <div class="chart-wrapper"><canvas id="ovScalingCanvas"></canvas></div>
    </div>
    <div class="card">
      <div class="card-header"><h2>Parallel Efficiency</h2><div class="actions"><span class="btn-sm">speedup ÷ threads · ideal = 100%</span></div></div>
      <div class="chart-wrapper"><canvas id="ovEfficiencyCanvas"></canvas></div>
    </div>
  </div>

  <div class="charts-row">
    <div class="card">
      <div class="card-header"><h2>IPC vs Threads</h2><div class="actions"><span class="btn-sm">microarch efficiency per dataset</span></div></div>
      <div class="chart-wrapper"><canvas id="ovIpcCanvas"></canvas></div>
    </div>
    <div class="card">
      <div class="card-header"><h2>Performance Matrix</h2><div class="actions"><span class="btn-sm">wall × threads · bubble = sites</span></div></div>
      <div class="chart-wrapper"><canvas id="ovMatrixCanvas"></canvas></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Commands · <span style="color:var(--text3); font-weight:500; font-size:0.78rem;" id="ovCmdsRunId">—</span></h2>
      <div class="actions"><button class="copy-btn" data-copy="#ovCmdsText">Copy all</button></div>
    </div>
    <div class="card-body" id="ovCmds"></div>
    <pre id="ovCmdsText" class="sr-only"></pre>
  </div>
`;

export async function mount(root) {
  root.innerHTML = TMPL;
  const idx = store.get('runsIndex');

  renderStats(idx);
  renderBestRuns(idx);
  renderDatasets(idx);

  scaling.render(document.getElementById('ovScalingCanvas'), idx);
  efficiency.render(document.getElementById('ovEfficiencyCanvas'), idx);
  ipcScaling.render(document.getElementById('ovIpcCanvas'), idx);
  perfMatrix.render(document.getElementById('ovMatrixCanvas'), idx);

  // Default: fastest run
  const defaultRun = [...idx].sort((a, b) => (a.wall_s ?? 1e12) - (b.wall_s ?? 1e12))[0];
  mountRunPicker(document.getElementById('ovRunPicker'), idx, {
    selectedId: defaultRun?.run_id,
    onChange: (r) => loadRun(r.run_id).then((full) => updateRunSections(full)),
  });
  if (defaultRun) {
    const full = await loadRun(defaultRun.run_id);
    updateRunSections(full);
  }

  bindCopyButtons(root);
}

/* --------------------------- Stats --------------------------- */
function renderStats(idx) {
  const total = idx.length;
  const allPass = idx.every((r) => r.all_pass);
  const bestIPC = idx.reduce((a, b) => (Number(b.IPC || 0) > Number(a?.IPC || 0) ? b : a), null);
  const bestSpeedup = idx.reduce((a, b) => (Number(b.speedup || 0) > Number(a?.speedup || 0) ? b : a), null);

  document.getElementById('ovStats').innerHTML = `
    <div class="stat-card">
      <div class="label">Total runs</div>
      <div class="value">${total}</div>
      <div class="change">${allPass ? 'All tests passing' : 'Some failures'}</div>
    </div>
    <div class="stat-card">
      <div class="label">Datasets</div>
      <div class="value">${new Set(idx.map(r => r.dataset_short).filter(Boolean)).size}</div>
      <div class="change">${[...new Set(idx.map(r => r.threads).filter(Boolean))].length} thread configs</div>
    </div>
    <div class="stat-card">
      <div class="label">Best speedup</div>
      <div class="value">${bestSpeedup?.speedup ? bestSpeedup.speedup.toFixed(2) + '×' : '—'}</div>
      <div class="change">${escHtml(bestSpeedup?.dataset_short || '—')} @ ${bestSpeedup?.threads || '—'}T</div>
    </div>
    <div class="stat-card">
      <div class="label">Peak IPC</div>
      <div class="value">${bestIPC?.IPC ? bestIPC.IPC.toFixed(2) : '—'}</div>
      <div class="change">${escHtml(bestIPC?.dataset_short || '—')} @ ${bestIPC?.threads || '—'}T</div>
    </div>
  `;
}

/* --------------------------- Best runs (clickable) --------------------------- */
function renderBestRuns(idx) {
  if (!idx.length) return;
  const fastest = [...idx].sort((a, b) => (a.wall_s ?? 1e12) - (b.wall_s ?? 1e12))[0];
  const bestSpeedup = [...idx].sort((a, b) => (b.speedup ?? 0) - (a.speedup ?? 0))[0];
  const bestIPC = [...idx].sort((a, b) => (b.IPC ?? 0) - (a.IPC ?? 0))[0];

  const card = (kind, valueHtml, run, subHtml) => `
    <a class="best-run-card" href="#/runs?open=${encodeURIComponent(run.run_id)}" aria-label="Open ${escHtml(run.run_id)} in All Runs">
      <span class="br-cta">Details
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </span>
      <div class="br-kind">${kind}</div>
      <div class="br-value">${valueHtml}</div>
      <div class="br-meta">${escHtml(run.run_id)}</div>
      <div class="br-sub">${subHtml}</div>
    </a>
  `;

  document.getElementById('ovBestRuns').innerHTML = [
    card('Fastest run',   fmtTime(fastest.wall_s),                      fastest,     `${escHtml(fastest.dataset_short || '—')} · ${fastest.threads}T`),
    card('Best speedup',  (bestSpeedup.speedup || 0).toFixed(2) + '×',  bestSpeedup, `${escHtml(bestSpeedup.dataset_short || '—')} · ${bestSpeedup.threads}T (${((bestSpeedup.efficiency || 0) * 100).toFixed(0)}% eff)`),
    card('Peak IPC',      (bestIPC.IPC || 0).toFixed(3),                bestIPC,     `${escHtml(bestIPC.dataset_short || '—')} · ${bestIPC.threads}T`),
  ].join('');
}

/* --------------------------- Dataset profile cards --------------------------- */
function renderDatasets(idx) {
  const byDs = new Map();
  for (const r of idx) {
    const ds = r.dataset_short;
    if (!ds) continue;
    if (!byDs.has(ds)) byDs.set(ds, []);
    byDs.get(ds).push(r);
  }
  if (!byDs.size) {
    document.getElementById('ovDatasets').innerHTML = '<div class="empty">No dataset metadata available.</div>';
    return;
  }
  const cards = [];
  for (const [ds, runs] of [...byDs.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const first = runs[0];
    const best = runs.reduce((a, b) => (b.wall_s < (a?.wall_s ?? Infinity) ? b : a), null);
    const threads = [...new Set(runs.map(r => r.threads).filter(Boolean))].sort((a, b) => a - b);
    cards.push(`
      <div class="ds-card">
        <span class="ds-tag">Dataset</span>
        <h3>${escHtml(ds)}</h3>
        <div class="ds-row"><span>Taxa</span><strong>${first.taxa ?? '—'}</strong></div>
        <div class="ds-row"><span>Sites</span><strong>${first.sites?.toLocaleString?.() ?? '—'}</strong></div>
        <div class="ds-row"><span>File size</span><strong>${first.size_mb != null ? first.size_mb.toFixed(2) + ' MB' : '—'}</strong></div>
        <div class="ds-row"><span>Runs</span><strong>${runs.length}</strong></div>
        <div class="ds-row"><span>Thread configs</span><strong>${threads.join(', ') || '—'}</strong></div>
        <div class="ds-row"><span>Best wall</span><strong class="accent">${best ? fmtTime(best.wall_s) : '—'}</strong></div>
      </div>
    `);
  }
  document.getElementById('ovDatasets').innerHTML = cards.join('');
}

/* --------------------------- Selected-run sections --------------------------- */
function updateRunSections(run) {
  if (!run) return;

  document.getElementById('ovSubtitle').textContent =
    `Run ${run.run_id} · ${run.env?.date || 'n/a'} · ${run.env?.hostname || 'n/a'}`;
  document.getElementById('ovConfigRunId').textContent = run.run_id;
  document.getElementById('ovCmdsRunId').textContent = run.run_id;

  const badge = document.getElementById('ovBadge');
  const s = run.summary || {};
  if (s.fail === 0) {
    badge.className = 'badge badge-pass';
    badge.textContent = s.pass ? `ALL PASS · ${s.pass}` : 'OK';
  } else {
    badge.className = 'badge badge-fail';
    badge.textContent = `${s.fail} FAILED`;
  }
  renderConfig(run);
  renderCmds(run);
}

function renderConfig(run) {
  const p = run.profile || {};
  const m = p.metrics || {};
  const env = run.env || {};
  const h = run.hints || {};

  const items = (pairs) => pairs
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `
      <div class="config-item"><span class="ci-label">${escHtml(k)}</span><span class="ci-value">${escHtml(String(v))}</span></div>
    `).join('');

  const alignment = items([
    ['Dataset', p.dataset || h.dataset],
    ['Threads', p.threads ?? h.threads],
    ['Run type', run.run_type],
  ]);
  const model = items([
    ['Model', h.model],
    ['Wall', fmtTime(run.summary?.total_time)],
    ['Commands', (run.timing || []).length],
    ['Verify pass', run.summary?.pass],
    ['Verify fail', run.summary?.fail],
  ]);
  const sys = items([
    ['Hostname', env.hostname],
    ['CPU', env.cpu],
    ['Cores', env.cores],
    ['GCC', env.gcc],
    ['ROCm', env.rocm],
    ['IPC', m.IPC],
    ['FE-stall %', m['frontend-stall-rate']],
  ]);

  document.getElementById('ovConfig').innerHTML = `
    <div class="config-grid">
      <div class="config-section"><h3>Alignment</h3><div class="config-items">${alignment}</div></div>
      <div class="config-section"><h3>Model &amp; Results</h3><div class="config-items">${model}</div></div>
      <div class="config-section"><h3>System</h3><div class="config-items">${sys}</div></div>
    </div>
    ${run.description ? `<div class="config-cmd" style="margin-top:12px;">${escHtml(run.description)}</div>` : ''}
  `;

  const toText = (label, pairs) =>
    `# ${label}\n` + pairs.filter(([, v]) => v != null && v !== '').map(([k, v]) => `${k}: ${v}`).join('\n');
  document.getElementById('ovConfigText').textContent = [
    `# Run ${run.run_id}`,
    toText('Alignment', [['Dataset', p.dataset || h.dataset], ['Threads', p.threads ?? h.threads], ['Run type', run.run_type]]),
    toText('Model & Results', [['Model', h.model], ['Wall', fmtTime(run.summary?.total_time)], ['Commands', (run.timing || []).length], ['Pass', run.summary?.pass], ['Fail', run.summary?.fail]]),
    toText('System', [['Host', env.hostname], ['CPU', env.cpu], ['Cores', env.cores], ['GCC', env.gcc], ['ROCm', env.rocm], ['IPC', m.IPC]]),
  ].join('\n\n');
}

function renderCmds(run) {
  const timing = run.timing || [];
  document.getElementById('ovCmds').innerHTML = timing.length
    ? timing.map((t, i) => `
      <div class="cmd-block">
        <span class="cmd-num">${i + 1}</span>
        <span class="cmd-text">${escHtml(t.command)}</span>
        <span class="cmd-time">${fmtTime(t.time_s)}</span>
      </div>
    `).join('')
    : '<div class="empty">No commands recorded.</div>';
  document.getElementById('ovCmdsText').textContent =
    '#!/usr/bin/env bash\nset -euo pipefail\n\n' + timing.map((t) => t.command).join('\n');
}
