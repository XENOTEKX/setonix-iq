// web/js/components/run-selector.js

import { store } from '../state.js';
import { loadRun } from '../data.js';
import { escHtml, fmtTime } from '../utils.js';

export function render(container, onChange) {
  const runs = store.get('runsIndex');
  const selectedId = store.get('selectedRunId') || (runs[0] && runs[0].run_id);

  const opts = runs
    .map((r) => {
      const label = `${r.label || r.run_id} — ${r.dataset || '?'} / T=${r.threads ?? '?'} / ${fmtTime(r.wall_s)}`;
      return `<option value="${escHtml(r.run_id)}" ${r.run_id === selectedId ? 'selected' : ''}>${escHtml(label)}</option>`;
    })
    .join('');

  container.innerHTML = `
    <div class="run-selector-top">
      <label for="runSelect">Run</label>
      <select id="runSelect" aria-label="Select run">${opts}</select>
      <div class="run-selector-info" id="runSelectorInfo"></div>
    </div>
  `;

  const select = container.querySelector('#runSelect');
  select.addEventListener('change', async (e) => {
    const id = e.target.value;
    store.set({ selectedRunId: id });
    const run = await loadRun(id);
    onChange?.(run);
    updateInfo(container, id);
  });

  updateInfo(container, selectedId);

  // Initial load
  if (selectedId) {
    loadRun(selectedId).then((run) => onChange?.(run));
  }
}

function updateInfo(container, id) {
  const runs = store.get('runsIndex');
  const r = runs.find((x) => x.run_id === id);
  if (!r) return;
  const info = container.querySelector('#runSelectorInfo');
  if (!info) return;
  info.innerHTML = `
    <span>Dataset <span class="rsi-value">${escHtml(r.dataset || 'n/a')}</span></span>
    <span>Threads <span class="rsi-value">${escHtml(String(r.threads ?? '—'))}</span></span>
    <span>IPC <span class="rsi-accent">${r.IPC ? r.IPC.toFixed(3) : '—'}</span></span>
    <span>Wall <span class="rsi-accent">${fmtTime(r.wall_s)}</span></span>
    <span>Status <span class="rsi-value">${r.all_pass ? '✓' : '✗'} ${r.pass}/${r.pass + r.fail}</span></span>
  `;
}
