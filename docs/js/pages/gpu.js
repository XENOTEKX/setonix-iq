// web/js/pages/gpu.js — GPU telemetry aggregated across deep profiles (NVIDIA V100 on Gadi gpuvolta / AMD MI250X on Setonix)

import { store } from '../state.js?v=20260430152808';
import { loadProfile } from '../data.js?v=20260430152808';
import { escHtml } from '../utils.js?v=20260430152808';

export async function mount(root) {
  root.innerHTML = `
    <div class="page-header"><div><h1>GPU</h1>
    <div class="subtitle">GPU telemetry from deep profiles (NVIDIA V100 on Gadi <code>gpuvolta</code> / AMD MI250X on Setonix)</div></div></div>
    <div class="card"><div class="card-body" id="gpuBody">Loading…</div></div>
  `;
  const body = document.getElementById('gpuBody');
  const idx = store.get('profilesIndex').filter((p) => p.has_gpu);
  if (!idx.length) {
    body.innerHTML = '<div class="empty">No GPU profiles available yet. Run <code>./start.sh deepprofile</code> on Gadi (or Setonix) to populate.</div>';
    return;
  }
  const profiles = await Promise.all(idx.map((p) => loadProfile(p.profile_id)));
  body.innerHTML = profiles.map((p) => {
    const gpu = p.gpu || {};
    const keys = Object.keys(gpu).slice(0, 20);
    const kv = keys.map((k) => `<div class="kv-item"><div class="k">${escHtml(k)}</div><div class="v">${escHtml(JSON.stringify(gpu[k]).slice(0, 60))}</div></div>`).join('');
    return `
      <h3 style="font-size:0.82rem;margin:0 0 8px;">${escHtml(p.profile_id)}</h3>
      <div class="kv-grid" style="margin-bottom:14px;">${kv || '<div class="empty">No GPU keys</div>'}</div>
    `;
  }).join('');
}
