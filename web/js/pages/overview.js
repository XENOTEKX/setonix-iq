// web/js/pages/overview.js — v2 (insight-oriented)

import { store } from '../state.js';
import { loadRun } from '../data.js';
import { mountRunPicker } from '../components/run-picker.js';
import { bindCopyButtons } from '../components/copy-button.js';
import { attachExpand } from '../components/chart-expand.js';
import * as scaling from '../charts/scaling.js';
import * as efficiency from '../charts/efficiency.js';
import * as ipcScaling from '../charts/ipc-scaling.js';
import * as perfMatrix from '../charts/performance-matrix.js';
import { escHtml, fmtTime, fmtNum } from '../utils.js';

/* --------------------------- Platform helpers --------------------------- */
function platformOf(r) {
  return r?.platform || (r?.pbs_id ? 'gadi' : (r?.slurm_id ? 'setonix' : null));
}
function platformBadge(p, { compact = false } = {}) {
  if (p === 'gadi') return `<span class="badge badge-platform-gadi" title="Gadi · NCI · Intel Sapphire Rapids">${compact ? 'Gadi' : 'Gadi · NCI'}</span>`;
  if (p === 'setonix') return `<span class="badge badge-platform-setonix" title="Setonix · Pawsey · AMD EPYC">${compact ? 'Setonix' : 'Setonix · Pawsey'}</span>`;
  return '';
}
function platformLabel(p) {
  if (p === 'gadi') return 'Gadi';
  if (p === 'setonix') return 'Setonix';
  return '—';
}
// A run is eligible for leaderboards / charts only if it actually produced
// timing data and its verify phase passed. This filters out the stub records
// that harvesters emit for cancelled / failed jobs.
function isValidRun(r) {
  return r && r.all_pass && Number(r.wall_s) > 0;
}

const TMPL = `
  <div class="page-header">
    <div>
      <h1>Pipeline Overview</h1>
      <div class="subtitle" id="ovSubtitle">Insight dashboard for IQ-TREE runs on Setonix (Pawsey) and Gadi (NCI)</div>
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
    <div class="card" id="ovScalingCard">
      <div class="card-header"><h2>Thread Scaling</h2><div class="actions"><span class="btn-sm">wall vs threads (log–log)</span></div></div>
      <div class="chart-wrapper"><canvas id="ovScalingCanvas"></canvas></div>
    </div>
    <div class="card" id="ovEfficiencyCard">
      <div class="card-header"><h2>Parallel Efficiency</h2><div class="actions"><span class="btn-sm">speedup ÷ threads · ideal = 100%</span></div></div>
      <div class="chart-wrapper"><canvas id="ovEfficiencyCanvas"></canvas></div>
    </div>
  </div>

  <div class="charts-row">
    <div class="card" id="ovIpcCard">
      <div class="card-header"><h2>IPC vs Threads</h2><div class="actions"><span class="btn-sm">microarch efficiency per dataset</span></div></div>
      <div class="chart-wrapper"><canvas id="ovIpcCanvas"></canvas></div>
    </div>
    <div class="card" id="ovMatrixCard">
      <div class="card-header"><h2>Performance Matrix</h2><div class="actions"><span class="btn-sm">wall × threads · bubble = sites</span></div></div>
      <div class="chart-wrapper"><canvas id="ovMatrixCanvas"></canvas></div>
    </div>
  </div>

  <div class="card activity-panel">
    <div class="card-header"><h2>Recent Runs</h2>
      <div class="actions"><a href="#/runs" class="btn-sm" style="text-decoration:none;">View all →</a></div>
    </div>
    <div class="card-body" id="ovActivity"></div>
  </div>
`;

export async function mount(root) {
  root.innerHTML = TMPL;
  const idx = store.get('runsIndex');

  renderStats(idx);
  renderBestRuns(idx);
  renderDatasets(idx);
  renderActivity(idx);

  scaling.render(document.getElementById('ovScalingCanvas'), idx);
  efficiency.render(document.getElementById('ovEfficiencyCanvas'), idx);
  ipcScaling.render(document.getElementById('ovIpcCanvas'), idx);
  perfMatrix.render(document.getElementById('ovMatrixCanvas'), idx);

  // Expand buttons for each chart
  const chartSpecs = [
    { id: 'ovScalingCard',    title: 'Thread Scaling',      badge: 'wall vs threads (log–log)',      mod: scaling },
    { id: 'ovEfficiencyCard', title: 'Parallel Efficiency', badge: 'speedup ÷ threads · ideal 100%', mod: efficiency },
    { id: 'ovIpcCard',        title: 'IPC vs Threads',      badge: 'microarch efficiency',           mod: ipcScaling },
    { id: 'ovMatrixCard',     title: 'Performance Matrix',  badge: 'wall × threads · bubble=sites',  mod: perfMatrix },
  ];
  for (const spec of chartSpecs) {
    const card = document.getElementById(spec.id);
    attachExpand(card, {
      title: spec.title,
      badge: spec.badge,
      renderFn: (body) => {
        body.innerHTML = '<div class="chart-wrapper" style="height:100%;"><canvas></canvas></div>';
        spec.mod.render(body.querySelector('canvas'), idx);
      },
    });
  }

  // Default: fastest *valid* run so Gadi stub records don't hijack the picker.
  const defaultRun = [...idx]
    .filter(isValidRun)
    .sort((a, b) => a.wall_s - b.wall_s)[0] || idx[0];
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
  const valid = idx.filter(isValidRun);
  const failedCount = idx.length - valid.length;
  const platforms = new Set(idx.map(platformOf).filter(Boolean));
  const platformLine = [...platforms].map(platformLabel).join(' · ') || 'no platform';

  const bestIPC = valid.reduce((a, b) => (Number(b.IPC || 0) > Number(a?.IPC || 0) ? b : a), null);
  const bestSpeedup = valid.reduce((a, b) => (Number(b.speedup || 0) > Number(a?.speedup || 0) ? b : a), null);
  // Dataset count is scoped to (dataset, platform) so Setonix's xlarge_dna.fa
  // and Gadi's xlarge_mf.fa (different taxa×sites despite similar names) count
  // as distinct profiles.
  const dsKeys = new Set(
    idx.filter(r => r.dataset_short).map(r => `${r.dataset_short}|${platformOf(r) || '?'}`)
  );

  document.getElementById('ovStats').innerHTML = `
    <div class="stat-card">
      <div class="label">Total runs</div>
      <div class="value">${total}</div>
      <div class="change">${failedCount ? `${valid.length} valid · ${failedCount} stub/failed` : 'All tests passing'}</div>
    </div>
    <div class="stat-card">
      <div class="label">Datasets</div>
      <div class="value">${dsKeys.size}</div>
      <div class="change">${platformLine}</div>
    </div>
    <div class="stat-card">
      <div class="label">Best speedup</div>
      <div class="value">${bestSpeedup?.speedup ? bestSpeedup.speedup.toFixed(2) + '×' : '—'}</div>
      <div class="change">${escHtml(bestSpeedup?.dataset_short || '—')} · ${platformLabel(platformOf(bestSpeedup))} @ ${bestSpeedup?.threads || '—'}T</div>
    </div>
    <div class="stat-card">
      <div class="label">Peak IPC</div>
      <div class="value">${bestIPC?.IPC ? bestIPC.IPC.toFixed(2) : '—'}</div>
      <div class="change">${bestIPC ? `${escHtml(bestIPC.dataset_short || '—')} · ${platformLabel(platformOf(bestIPC))} @ ${bestIPC.threads || '—'}T` : 'CPU counters pending'}</div>
    </div>
  `;
}

/* --------------------------- Best runs (clickable) --------------------------- */
function renderBestRuns(idx) {
  if (!idx.length) return;
  const valid = idx.filter(isValidRun);
  if (!valid.length) {
    document.getElementById('ovBestRuns').innerHTML =
      '<div class="empty">No successful runs with timing data yet.</div>';
    return;
  }
  const fastest = [...valid].sort((a, b) => a.wall_s - b.wall_s)[0];
  const withSpeedup = valid.filter(r => r.speedup != null);
  const bestSpeedup = withSpeedup.length
    ? [...withSpeedup].sort((a, b) => (b.speedup ?? 0) - (a.speedup ?? 0))[0]
    : fastest;
  const withIPC = valid.filter(r => r.IPC != null);
  const bestIPC = withIPC.length
    ? [...withIPC].sort((a, b) => (b.IPC ?? 0) - (a.IPC ?? 0))[0]
    : null;

  const card = (kind, valueHtml, run, subHtml) => run ? `
    <a class="best-run-card" href="#/runs?open=${encodeURIComponent(run.run_id)}" aria-label="Open ${escHtml(run.run_id)} in All Runs">
      <span class="br-cta">Details
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </span>
      <div class="br-kind">${kind} ${platformBadge(platformOf(run), { compact: true })}</div>
      <div class="br-value">${valueHtml}</div>
      <div class="br-meta">${escHtml(run.run_id)}</div>
      <div class="br-sub">${subHtml}</div>
    </a>
  ` : `
    <div class="best-run-card best-run-card--empty">
      <div class="br-kind">${kind}</div>
      <div class="br-value" style="color:var(--text3);">—</div>
      <div class="br-sub">CPU counters pending</div>
    </div>
  `;

  document.getElementById('ovBestRuns').innerHTML = [
    card('Fastest run',  fmtTime(fastest.wall_s),                      fastest,     `${escHtml(fastest.dataset_short || '—')} · ${fastest.threads}T`),
    card('Best speedup', (bestSpeedup.speedup || 0).toFixed(2) + '×', bestSpeedup, `${escHtml(bestSpeedup.dataset_short || '—')} · ${bestSpeedup.threads}T (${((bestSpeedup.efficiency || 0) * 100).toFixed(0)}% eff)`),
    card('Peak IPC',     bestIPC ? (bestIPC.IPC || 0).toFixed(3) : '—', bestIPC, bestIPC ? `${escHtml(bestIPC.dataset_short || '—')} · ${bestIPC.threads}T` : ''),
  ].join('');
}

/* --------------------------- Dataset profile cards --------------------------- */
function renderDatasets(idx) {
  // Group by (dataset, platform) — Setonix and Gadi regenerate datasets with
  // different taxa×sites dimensions, so a combined card would be misleading.
  const byKey = new Map();
  for (const r of idx) {
    const ds = r.dataset_short;
    if (!ds) continue;
    const plat = platformOf(r) || 'unknown';
    const key = `${ds}|${plat}`;
    if (!byKey.has(key)) byKey.set(key, { ds, platform: plat, runs: [] });
    byKey.get(key).runs.push(r);
  }
  if (!byKey.size) {
    document.getElementById('ovDatasets').innerHTML = '<div class="empty">No dataset metadata available.</div>';
    return;
  }
  const cards = [];
  const sortedGroups = [...byKey.values()].sort((a, b) =>
    a.ds.localeCompare(b.ds) || a.platform.localeCompare(b.platform));
  for (const { ds, platform, runs } of sortedGroups) {
    // Pick a representative (prefer a run with taxa/sites populated).
    const rep = runs.find(r => r.taxa && r.sites) || runs[0];
    const validRuns = runs.filter(isValidRun);
    const best = validRuns.reduce((a, b) => (b.wall_s < (a?.wall_s ?? Infinity) ? b : a), null);
    const threads = [...new Set(validRuns.map(r => r.threads).filter(Boolean))].sort((a, b) => a - b);
    const stubs = runs.length - validRuns.length;
    const patterns = rep.patterns;
    const sites = rep.sites;
    const compression = (patterns && sites)
      ? `${patterns.toLocaleString()} · ${((patterns / sites) * 100).toFixed(1)}% unique`
      : null;
    cards.push(`
      <div class="ds-card">
        <div class="ds-head">
          <span class="ds-tag">Dataset</span>
          ${platformBadge(platform, { compact: true })}
        </div>
        <h3>${escHtml(ds)}</h3>
        <div class="ds-row"><span>Taxa</span><strong>${rep.taxa ?? '—'}</strong></div>
        <div class="ds-row"><span>Sites</span><strong>${rep.sites?.toLocaleString?.() ?? '—'}</strong></div>
        ${compression ? `<div class="ds-row"><span>Patterns</span><strong>${compression}</strong></div>` : ''}
        <div class="ds-row"><span>File size</span><strong>${rep.size_mb != null ? (rep.size_estimated ? '~' : '') + rep.size_mb.toFixed(2) + ' MB' : '—'}</strong></div>
        <div class="ds-row"><span>Runs</span><strong>${runs.length}${stubs ? ` <span style="color:var(--text3); font-weight:400;">(${stubs} stub)</span>` : ''}</strong></div>
        <div class="ds-row"><span>Thread configs</span><strong>${threads.join(', ') || '—'}</strong></div>
        <div class="ds-row"><span>Best wall</span><strong class="accent">${best ? fmtTime(best.wall_s) : '—'}</strong></div>
        ${rep.size_estimated ? '<div class="ds-note">~ size estimated from alignment dimensions</div>' : ''}
      </div>
    `);
  }
  document.getElementById('ovDatasets').innerHTML = cards.join('');
}

/* --------------------------- Selected-run sections --------------------------- */
function updateRunSections(run) {
  if (!run) return;

  document.getElementById('ovSubtitle').textContent =
    `Run ${run.run_id} · ${platformLabel(platformOf(run))} · ${run.env?.date || 'n/a'} · ${run.env?.hostname || 'n/a'}`;
  document.getElementById('ovConfigRunId').textContent = run.run_id;

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
}

function renderConfig(run) {
  const p = run.profile || {};
  const m = p.metrics || {};
  const env = run.env || {};
  const h = run.hints || {};
  const di = run.dataset_info || {};
  const mf = run.modelfinder || {};

  const items = (pairs) => pairs
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `
      <div class="config-item"><span class="ci-label">${escHtml(k)}</span><span class="ci-value">${escHtml(String(v))}</span></div>
    `).join('');

  const fmtInt = (n) => (typeof n === 'number') ? n.toLocaleString() : n;
  const sizeMb = di.file_size_bytes ? (di.file_size_bytes / 1_000_000).toFixed(2) + ' MB' : null;

  const alignment = items([
    ['Dataset', p.dataset || h.dataset],
    ['Sequence type', di.sequence_type],
    ['Taxa', fmtInt(di.taxa)],
    ['Sites', fmtInt(di.sites)],
    ['Distinct patterns', fmtInt(di.patterns)],
    ['Informative sites', fmtInt(di.informative_sites)],
    ['Constant sites', fmtInt(di.constant_sites)],
    ['File size', sizeMb],
    ['Threads', p.threads ?? h.threads],
    ['Run type', run.run_type],
  ]);
  const model = items([
    ['Model selected', mf.model_selected || h.model],
    ['Best-fit (BIC)', mf.best_model_bic],
    ['Log-likelihood', mf.log_likelihood != null ? mf.log_likelihood.toLocaleString() : null],
    ['Tree length', mf.tree_length],
    ['Gamma α', mf.gamma_alpha],
    ['BIC', mf.bic != null ? mf.bic.toLocaleString() : null],
    ['AIC', mf.aic != null ? mf.aic.toLocaleString() : null],
    ['Wall', fmtTime(run.summary?.total_time)],
    ['Verify pass', run.summary?.pass],
    ['Verify fail', run.summary?.fail],
  ]);
  const sys = items([
    ['Hostname', env.hostname],
    ['CPU', env.cpu],
    ['Cores', env.cores],
    ['GCC', env.gcc],
    ['VTune', env.vtune_version || env.rocm],
    ['IPC', m.IPC],
    ['FE-stall %', m['frontend-stall-rate']],
  ]);

  document.getElementById('ovConfig').innerHTML = `
    <div class="config-grid">
      <div class="config-section"><h3>Alignment</h3><div class="config-items">${alignment}</div></div>
      <div class="config-section"><h3>Model &amp; Results</h3><div class="config-items">${model}</div></div>
      <div class="config-section"><h3>System</h3><div class="config-items">${sys}</div></div>
    </div>
    ${renderModelfinderShare(mf, run.summary)}
    ${renderCandidatesTable(mf.candidates)}
    ${p.perf_cmd ? `
      <details class="config-cmd" style="margin-top:12px;">
        <summary style="cursor:pointer;">How this was measured (perf command)</summary>
        <pre style="margin-top:8px; white-space:pre-wrap; word-break:break-all; font-size:0.72rem; color:var(--text2);">${escHtml(p.perf_cmd)}</pre>
      </details>` : ''}
    ${run.description ? `<div class="config-cmd" style="margin-top:12px;">${escHtml(run.description)}</div>` : ''}
  `;

  const toText = (label, pairs) =>
    `# ${label}\n` + pairs.filter(([, v]) => v != null && v !== '').map(([k, v]) => `${k}: ${v}`).join('\n');
  document.getElementById('ovConfigText').textContent = [
    `# Run ${run.run_id}`,
    toText('Alignment', [
      ['Dataset', p.dataset || h.dataset],
      ['Sequence type', di.sequence_type],
      ['Taxa', di.taxa], ['Sites', di.sites],
      ['Patterns', di.patterns],
      ['Informative sites', di.informative_sites],
      ['Constant sites', di.constant_sites],
      ['Threads', p.threads ?? h.threads],
      ['Run type', run.run_type],
    ]),
    toText('Model & Results', [
      ['Model', mf.model_selected || h.model],
      ['Best-fit (BIC)', mf.best_model_bic],
      ['Log-likelihood', mf.log_likelihood],
      ['Tree length', mf.tree_length],
      ['Gamma α', mf.gamma_alpha],
      ['BIC', mf.bic], ['AIC', mf.aic],
      ['Wall', fmtTime(run.summary?.total_time)],
      ['Pass', run.summary?.pass], ['Fail', run.summary?.fail],
    ]),
    toText('System', [['Host', env.hostname], ['CPU', env.cpu], ['Cores', env.cores], ['GCC', env.gcc], ['VTune', env.vtune_version || env.rocm], ['IPC', m.IPC]]),
    p.perf_cmd ? `# Perf command\n${p.perf_cmd}` : null,
  ].filter(Boolean).join('\n\n');
}

function renderCandidatesTable(candidates) {
  if (!Array.isArray(candidates) || !candidates.length) return '';
  const sorted = [...candidates].sort((a, b) => (a.bic ?? Infinity) - (b.bic ?? Infinity));
  const bestBic = sorted[0]?.bic;
  const rows = sorted.slice(0, 10).map((c, i) => {
    const dBic = (c.bic != null && bestBic != null) ? c.bic - bestBic : null;
    const dBicCell = dBic == null
      ? '—'
      : (i === 0 ? '<span style="color:var(--accent); font-weight:600;">best</span>'
                 : (dBic > 1e4 ? (dBic / 1000).toFixed(1) + 'k' : dBic.toFixed(1)));
    return `
      <tr${i === 0 ? ' class="mf-best"' : ''}>
        <td><code>${escHtml(c.model || '')}</code></td>
        <td class="num">${c.log_likelihood != null ? c.log_likelihood.toLocaleString() : '—'}</td>
        <td class="num">${c.bic != null ? c.bic.toLocaleString() : '—'}</td>
        <td class="num">${dBicCell}</td>
        <td class="num">${c.bic_weight != null ? c.bic_weight.toPrecision(3) : '—'}</td>
        <td class="num">${c.aic != null ? c.aic.toLocaleString() : '—'}</td>
        <td class="num">${c.aic_weight != null ? c.aic_weight.toPrecision(3) : '—'}</td>
      </tr>
    `;
  }).join('');
  return `
    <div style="margin-top:16px;">
      <h3 class="mf-title">Top ModelFinder candidates</h3>
      <div class="mf-scroll">
        <table class="mf-candidates">
          <thead>
            <tr>
              <th>Model</th>
              <th class="num">LogL</th>
              <th class="num">BIC</th>
              <th class="num" title="BIC minus best BIC — &gt;10 = decisively worse">ΔBIC</th>
              <th class="num">w-BIC</th>
              <th class="num">AIC</th>
              <th class="num">w-AIC</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="mf-note">Top ${Math.min(candidates.length, 10)} of ${candidates.length} models sorted by BIC. ΔBIC &gt; 10 = decisively worse than best.</div>
    </div>
  `;
}

function renderModelfinderShare(mf, summary) {
  const mfWall = mf?.wall_time_s;
  const total = summary?.total_time;
  if (!mfWall || !total) return '';
  const pct = Math.max(0, Math.min(100, (mfWall / total) * 100));
  return `
    <div class="mf-share" style="margin-top:14px; padding:10px 12px; background:var(--card-2); border:1px solid var(--border); border-radius:10px;">
      <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap;">
        <div style="font-size:0.74rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">ModelFinder share of wall time</div>
        <div style="font-size:0.82rem; color:var(--text);"><strong style="color:var(--accent);">${pct.toFixed(1)}%</strong> — ${fmtTime(mfWall)} of ${fmtTime(total)}</div>
      </div>
      <div style="margin-top:8px; background:var(--bg-2); border-radius:999px; height:6px; overflow:hidden;">
        <div style="height:100%; width:${pct}%; background:var(--accent-grad); border-radius:999px;"></div>
      </div>
    </div>
  `;
}

function renderActivity(idx) {
  const el = document.getElementById('ovActivity');
  if (!el) return;
  if (!idx.length) { el.innerHTML = '<div class="empty">No runs recorded yet.</div>'; return; }
  const recent = [...idx]
    .sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')))
    .slice(0, 6);
  el.innerHTML = `
    <div class="activity-list">
      ${recent.map((r) => {
        const ds = r.dataset_short || r.dataset || '—';
        const dims = (r.taxa && r.sites) ? ` · ${r.taxa}×${r.sites}` : '';
        const model = r.model || (r.run_type === 'modelfinder' ? 'ModelFinder' : '');
        const plat = platformOf(r);
        return `
          <a class="activity-row" href="#/runs?open=${encodeURIComponent(r.run_id)}">
            <span class="activity-dot ${r.all_pass ? '' : 'fail'}"></span>
            <div class="activity-main">
              <div class="activity-title">${escHtml(r.label || r.run_id)} ${platformBadge(plat, { compact: true })}</div>
              <div class="activity-sub">${escHtml(ds)}${dims}${model ? ' · ' + escHtml(model) : ''} · ${escHtml(r.date || 'no date')}</div>
            </div>
            <span class="activity-threads">T=${r.threads ?? '—'}</span>
            <span class="activity-wall">${fmtTime(r.wall_s)}</span>
          </a>
        `;
      }).join('')}
    </div>
  `;
}
