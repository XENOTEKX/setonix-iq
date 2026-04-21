// web/js/pages/environment.js

import * as runSelector from '../components/run-selector.js';
import { escHtml } from '../utils.js';

export function mount(root) {
  root.innerHTML = `
    <div class="page-header"><div><h1>Environment</h1>
    <div class="subtitle">System &amp; toolchain details per run</div></div></div>
    <div class="run-selector-bar" id="envRunSelector"></div>
    <div class="card"><div class="card-body" id="envBody"></div></div>
  `;
  runSelector.render(document.getElementById('envRunSelector'), (run) => {
    const env = run.env || {};
    const entries = Object.entries(env);
    document.getElementById('envBody').innerHTML = entries.length
      ? `<div class="kv-grid">${entries.map(([k, v]) =>
          `<div class="kv-item"><div class="k">${escHtml(k)}</div><div class="v">${escHtml(String(v))}</div></div>`
        ).join('')}</div>`
      : '<div class="empty">No environment data captured for this run.</div>';
  });
}
