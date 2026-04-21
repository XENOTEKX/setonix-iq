// web/js/charts/hotspot.js — per-run hotspot bar list (top N functions)

import { escHtml, fmtPercent, shortFn } from '../utils.js';

export function render(container, hotspots, { limit = 10 } = {}) {
  if (!hotspots || !hotspots.length) {
    container.innerHTML = '<div class="empty">No hotspot data for this run.</div>';
    return;
  }
  const top = hotspots.slice(0, limit);
  const max = Math.max(...top.map((h) => h.percent));
  container.innerHTML = top
    .map((h) => {
      const pct = (h.percent / max) * 100;
      return `
        <div class="hotspot-row">
          <div class="hotspot-pct">${fmtPercent(h.percent, 2)}</div>
          <div class="hotspot-bar-wrap"><div class="hotspot-bar" style="width:${pct}%"></div></div>
          <div class="hotspot-func" title="${escHtml(h.function)}">${escHtml(shortFn(h.function))}</div>
          <div class="hotspot-module">${escHtml(h.module || '—')}</div>
        </div>
      `;
    })
    .join('');
}
