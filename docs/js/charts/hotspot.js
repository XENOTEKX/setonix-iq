// web/js/charts/hotspot.js — per-run hotspot bar list (top N functions)

import { escHtml, fmtPercent, shortFn } from '../utils.js?v=20260425232033';

export function render(container, hotspots, { limit = 10 } = {}) {
  if (!hotspots || !hotspots.length) {
    container.innerHTML = '<div class="empty" style="padding:32px 20px; text-align:center;">No hotspot data for this run.</div>';
    return;
  }
  const top = hotspots.slice(0, limit);
  const max = Math.max(...top.map((h) => h.percent));
  container.innerHTML = `<div class="hotspot-list">${top
    .map((h) => {
      const pct = (h.percent / max) * 100;
      return `
        <div class="hotspot-row">
          <div class="hotspot-pct">${fmtPercent(h.percent, 2)}</div>
          <div class="hotspot-bar-wrap"><div class="hotspot-bar" style="width:${pct}%"></div></div>
          <div class="hotspot-func" title="${escHtml(h.function)}">${escHtml(shortFn(h.function))}</div>
          <div class="hotspot-module" title="${escHtml(h.module || '')}">${escHtml(h.module || '—')}</div>
        </div>
      `;
    })
    .join('')}</div>`;
}
