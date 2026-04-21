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
    </select>
    <span class="count" id="runsCount"></span>
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
  return `
    <div class="run-row" data-runrow="${escHtml(r.run_id)}">
      <div class="run-row-summary" role="button" tabindex="0">
        <div class="run-status">${statusBadge}</div>
        <div>
          <div class="run-id">${escHtml(r.label || r.run_id)}</div>
          <div class="run-meta">${escHtml(r.dataset || 'n/a')} · ${escHtml(r.model || '')} · ${escHtml(r.description || r.run_type || '')}</div>
        </div>
        <div class="run-time">${fmtTime(r.wall_s)}</div>
        <div class="run-ipc">${r.IPC ? r.IPC.toFixed(2) : '—'}</div>
        <div class="run-threads">T=${r.threads ?? '—'}</div>
        <div class="run-status">${r.pass}/${r.pass + r.fail || 0}</div>
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

  const kv = Object.entries({
    Dataset: p.dataset || run.hints?.dataset,
    Threads: p.threads ?? run.hints?.threads,
    Model: run.hints?.model,
    IPC: m.IPC,
    'FE stall %': m['frontend-stall-rate'],
    'Cache miss %': m['cache-miss-rate'],
    Host: env.hostname,
    CPU: env.cpu,
    Date: env.date,
    SLURM: run.slurm_id,
  }).filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `<div class="kv-item"><div class="k">${escHtml(k)}</div><div class="v">${escHtml(String(v))}</div></div>`)
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
