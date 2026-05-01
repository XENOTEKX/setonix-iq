// web/js/pages/runs.js — leaderboard/list with expandable detail

import { store } from '../state.js';
import { loadRun } from '../data.js';
import { bindCopyButtons } from '../components/copy-button.js';
import { escHtml, fmtTime, debounce } from '../utils.js';
import * as hotspotChart from '../charts/hotspot.js';

const TMPL = `
  <div class="page-header">
    <div>
      <h1>All Runs</h1>
      <div class="subtitle">Click any row for expandable detail</div>
    </div>
  </div>
  <div class="filter-bar">
    <input type="search" id="runsSearch" placeholder="Filter by dataset, model, run id…" aria-label="Filter">
    <select id="runsSort" aria-label="Sort">
      <option value="wall_asc">Wall ↑ (fastest)</option>
      <option value="wall_desc">Wall ↓</option>
      <option value="ipc_desc">IPC ↓</option>
      <option value="threads_asc">Threads ↑</option>
      <option value="date_desc">Most recent</option>
    </select>
    <select id="runsStatus" aria-label="Status filter">
      <option value="">All statuses</option>
      <option value="pass">Pass</option>
      <option value="fail">Fail</option>
      <option value="archived">Archived (pre-audit)</option>
      <option value="active">Active (non-archived)</option>
      <option value="non_canonical">Non-canonical (reference runs)</option>
    </select>
    <span class="count" id="runsCount"></span>
  </div>
  <div class="runs-legend" aria-hidden="true">
    <div class="runs-legend-title">Column key</div>
    <div class="runs-legend-grid">
      <span><b>●</b> Pass / fail status</span>
      <span><b>Label</b> Run id + short description</span>
      <span><b>Dataset</b> File · taxa × sites · size MB · model</span>
      <span><b>Wall</b> Total wall-clock (s / m / h)</span>
      <span><b>IPC</b> Instructions per cycle (microarch efficiency)</span>
      <span><b>T</b> Thread count</span>
      <span><b>P/T</b> Verification tests passed / total</span>
    </div>
  </div>
  <div id="runsList"></div>
`;

export function mount(root, ctx = {}) {
  root.innerHTML = TMPL;
  const list = document.getElementById('runsList');
  const search = document.getElementById('runsSearch');
  const sort = document.getElementById('runsSort');
  const status = document.getElementById('runsStatus');
  const autoOpenId = ctx.query?.open || null;

  function paint() {
    const q = (search.value || '').toLowerCase();
    const st = status.value;
    const runs = store.get('runsIndex').filter((r) => {
      if (st === 'pass' && !r.all_pass) return false;
      if (st === 'fail' && r.all_pass) return false;
      if (st === 'archived' && !r.archived) return false;
      if (st === 'active' && (r.archived || r.non_canonical)) return false;
      if (st === 'non_canonical' && !r.non_canonical) return false;
      if (!q) return true;
      return [r.run_id, r.dataset, r.model, r.label, r.description].some(
        (f) => f && String(f).toLowerCase().includes(q)
      );
    });

    switch (sort.value) {
      case 'wall_desc': runs.sort((a, b) => b.wall_s - a.wall_s); break;
      case 'ipc_desc': runs.sort((a, b) => (b.IPC || 0) - (a.IPC || 0)); break;
      case 'threads_asc': runs.sort((a, b) => (a.threads || 0) - (b.threads || 0)); break;
      case 'date_desc': runs.sort((a, b) => String(b.date || '').localeCompare(String(a.date || ''))); break;
      default: runs.sort((a, b) => a.wall_s - b.wall_s);
    }

    document.getElementById('runsCount').textContent = `${runs.length} runs`;
    list.innerHTML = runs.length
      ? runs.map(renderRow).join('')
      : '<div class="empty">No runs match the filter.</div>';
    list.querySelectorAll('[data-runrow]').forEach(bindRow);
  }

  search.addEventListener('input', debounce(paint, 120));
  sort.addEventListener('change', paint);
  status.addEventListener('change', paint);
  paint();

  // Auto-open a requested run and scroll it into view
  if (autoOpenId) {
    const target = list.querySelector(`[data-runrow="${CSS.escape(autoOpenId)}"]`);
    if (target) {
      const summary = target.querySelector('.run-row-summary');
      summary?.click();
      setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'center' }), 80);
    }
  }
}

function renderRow(r) {
  const statusBadge = r.all_pass
    ? '<span class="badge badge-pass">✓</span>'
    : '<span class="badge badge-fail">✗</span>';
  const platform = r.platform || (r.pbs_id ? 'gadi' : (r.slurm_id ? 'setonix' : null));
  const archivedBadge = r.archived
    ? '<span class="badge badge-archived" title="Pre-audit run — collected under non-canonical conditions. Excluded from comparison charts.">ARCHIVED</span>'
    : r.non_canonical
      ? `<span class="badge badge-non-canonical" title="Non-canonical reference run — shown as reference on charts but not parity-matched with canonical builds.">${escHtml('NON-CANONICAL · ' + (r.non_canonical_label || 'ref'))}</span>`
      : '';
  const platformBadge = platform === 'gadi'
    ? '<span class="badge badge-platform-gadi" title="Run on NCI Gadi (Intel Sapphire Rapids)">Gadi · NCI</span>'
    : platform === 'setonix'
      ? '<span class="badge badge-platform-setonix" title="Run on Pawsey Setonix (AMD EPYC)">Setonix · Pawsey</span>'
      : '';
  const ds = r.dataset_short || r.dataset || 'n/a';
  const modelLabel = r.model || (r.run_type === 'modelfinder' ? 'ModelFinder' : (r.description || r.run_type || '—'));
  const dims = (r.taxa && r.sites) ? `${r.taxa}×${r.sites}` : null;
  const size = r.size_mb ? `${r.size_estimated ? '~' : ''}${r.size_mb} MB` : null;
  const metaBits = [ds, dims, size, modelLabel].filter(Boolean).join(' · ');
  return `
    <div class="run-row" data-runrow="${escHtml(r.run_id)}">
      <div class="run-row-summary" role="button" tabindex="0">
        <div class="run-status">${statusBadge}</div>
        <div>
          <div class="run-id">${escHtml(r.label || r.run_id)} ${platformBadge}${archivedBadge}</div>
          <div class="run-meta">${escHtml(metaBits)}</div>
        </div>
        <div class="run-time" title="Wall time">${fmtTime(r.wall_s)}</div>
        <div class="run-ipc" title="Instructions per cycle">${r.IPC ? r.IPC.toFixed(2) : '—'}</div>
        <div class="run-threads" title="Thread count">T=${r.threads ?? '—'}</div>
        <div class="run-status" title="Tests passed / total">${r.pass}/${r.pass + r.fail || 0}</div>
        <button class="run-expand" aria-expanded="false">Details</button>
      </div>
      <div class="run-detail"></div>
    </div>
  `;
}

function bindRow(row) {
  const summary = row.querySelector('.run-row-summary');
  const detail = row.querySelector('.run-detail');
  const btn = row.querySelector('.run-expand');
  const open = async () => {
    const isOpen = row.classList.toggle('open');
    detail.classList.toggle('open', isOpen);
    btn.setAttribute('aria-expanded', String(isOpen));
    btn.textContent = isOpen ? 'Hide' : 'Details';
    if (isOpen && !detail.dataset.loaded) {
      detail.innerHTML = '<div class="empty">Loading…</div>';
      const run = await loadRun(row.dataset.runrow);
      detail.innerHTML = renderDetail(run);
      detail.dataset.loaded = '1';
      bindCopyButtons(detail);
      const hs = detail.querySelector('.rd-hotspots');
      if (hs) hotspotChart.render(hs, run.profile?.hotspots);
    }
  };
  summary.addEventListener('click', (e) => {
    if (e.target.closest('button')) return; // button has its own handler
    open();
  });
  summary.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      open();
    }
  });
  btn.addEventListener('click', (e) => { e.stopPropagation(); open(); });
}

function renderDetail(run) {
  const timing = run.timing || [];
  const verify = run.verify || [];
  const p = run.profile || {};
  const m = p.metrics || {};
  const env = run.env || {};

  const cacheLevel = m.cache_level || (run.platform === 'gadi' ? 'L3' : run.platform === 'setonix' ? 'L2' : null);
  const cacheMissLabel = cacheLevel ? `${cacheLevel} miss %` : 'Cache miss %';
  // Prefer the platform-specific alias (l2-miss-rate / l3-miss-rate) when
  // present; fall back to the generic cache-miss-rate field for older runs
  // that pre-date the cache-level annotation.
  const cacheMissValue =
       (cacheLevel === 'L2' ? m['l2-miss-rate'] : null)
    ?? (cacheLevel === 'L3' ? m['l3-miss-rate'] : null)
    ?? m['cache-miss-rate'];

  const kv = Object.entries({
    Platform: run.platform === 'gadi' ? 'Gadi (NCI · Intel Sapphire Rapids)'
            : run.platform === 'setonix' ? 'Setonix (Pawsey · AMD EPYC)'
            : (run.pbs_id ? 'Gadi (NCI · Intel Sapphire Rapids)'
               : run.slurm_id ? 'Setonix (Pawsey · AMD EPYC)' : null),
    Dataset: p.dataset || run.hints?.dataset,
    Threads: p.threads ?? run.hints?.threads,
    Model: run.hints?.model,
    IPC: m.IPC,
    'L1d-MPKI': m['L1d-mpki'],
    'L1d miss %': m['L1-dcache-miss-rate'],
    // FE-stall units differ across vendors: AMD Zen3 reports cycles
    // (max 100 %); Intel SPR reports issue-slots (max 600 %, since SPR
    // can issue 6 µops/cycle). Tooltip annotates the unit.
    [`FE stall % (${run.platform === 'gadi' ? 'Intel slots' : 'AMD cycles'})`]: m['frontend-stall-rate'],
    [cacheMissLabel]: cacheMissValue,
    'Build tag': run.build_tag,
    'OpenMP runtime': env.omp_runtime,
    Host: env.hostname,
    CPU: env.cpu,
    Date: env.date,
    Job: run.pbs_id || run.slurm_id,
  }).filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `<div class="kv-item"><div class="k">${escHtml(k)}</div><div class="v">${escHtml(typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v))}</div></div>`)
    .join('');

  const cmds = timing.map((t, i) => `
    <div class="cmd-block">
      <span class="cmd-num">${i + 1}</span>
      <span class="cmd-text">${escHtml(t.command)}</span>
      <span class="cmd-time">${fmtTime(t.time_s)}</span>
    </div>
  `).join('');

  const cmdsText = timing.map((t) => t.command).join('\n');
  const verifyTable = verify.length ? `
    <table class="data-table">
      <thead><tr><th>Status</th><th>File</th><th>Expected</th><th>Reported</th><th>|Δ|</th></tr></thead>
      <tbody>${verify.map((v) => `
        <tr>
          <td><span class="badge badge-${v.status === 'pass' ? 'pass' : 'fail'}">${v.status}</span></td>
          <td>${escHtml(v.file)}</td>
          <td>${v.expected}</td>
          <td>${v.reported}</td>
          <td>${v.diff}</td>
        </tr>`).join('')}</tbody>
    </table>` : '';

  return `
    <div style="margin-bottom:14px;"><div class="kv-grid">${kv}</div></div>
    ${p.hotspots?.length ? '<h3 style="margin:10px 0;font-size:0.78rem;color:var(--text3);text-transform:uppercase;">Hotspots</h3><div class="rd-hotspots"></div>' : ''}
    ${verifyTable ? '<h3 style="margin:14px 0 6px;font-size:0.78rem;color:var(--text3);text-transform:uppercase;">Verification</h3>' + verifyTable : ''}
    ${cmds ? `
      <div style="display:flex;justify-content:space-between;align-items:center;margin:14px 0 6px;">
        <h3 style="font-size:0.78rem;color:var(--text3);text-transform:uppercase;margin:0;">Commands</h3>
        <button class="copy-btn btn-sm" data-copy="${escHtml(cmdsText)}">Copy all</button>
      </div>${cmds}` : ''}
  `;
}
