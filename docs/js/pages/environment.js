// web/js/pages/environment.js

import { store } from '../state.js?v=20260430105505';
import { loadRun } from '../data.js?v=20260430105505';
import { mountRunPicker } from '../components/run-picker.js?v=20260430105505';
import { escHtml } from '../utils.js?v=20260430105505';

export async function mount(root) {
  root.innerHTML = `
    <div class="page-header"><div><h1>Environment</h1>
    <div class="subtitle">System &amp; toolchain details per run</div></div></div>
    <div class="run-picker-wrap" style="margin-bottom:22px;">
      <div style="margin-bottom:10px; font-size:0.72rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Selected run</div>
      <div id="envRunPicker"></div>
    </div>
    <div class="card"><div class="card-body" id="envBody"></div></div>
  `;

  const idx = store.get('runsIndex') || [];
  if (!idx.length) {
    document.getElementById('envBody').innerHTML = '<div class="empty">No runs indexed.</div>';
    return;
  }
  const first = idx[0];

  mountRunPicker(document.getElementById('envRunPicker'), idx, {
    selectedId: first.run_id,
    onChange: async (r) => renderEnv(await loadRun(r.run_id)),
  });
  renderEnv(await loadRun(first.run_id));
}

function renderEnv(run) {
  const env = run?.env || {};
  const flat = [];
  for (const [k, v] of Object.entries(env)) {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v)) {
        flat.push([`${k}.${k2}`, v2]);
      }
    } else {
      flat.push([k, v]);
    }
  }
  document.getElementById('envBody').innerHTML = flat.length
    ? `<div class="kv-grid">${flat.map(([k, v]) =>
        `<div class="kv-item"><div class="k">${escHtml(k)}</div><div class="v">${escHtml(String(v))}</div></div>`
      ).join('')}</div>`
    : '<div class="empty">No environment data captured for this run.</div>';
}
