// web/js/charts/callstack.js — callstack visualisation from folded_stacks

import { escHtml } from '../utils.js';

function colourFrame(frame) {
  if (!frame || frame === '[unknown]') return 'cs-unk';
  if (frame.includes('computeLikelihood') || frame.includes('computePartialLikelihood')) return 'cs-lh';
  if (frame.includes('PhyloTree') || frame.includes('PhyloNode')) return 'cs-phylo';
  if (frame.includes('GOMP_') || frame.includes('omp')) return 'cs-omp';
  return '';
}

export function render(container, foldedStacks, { limit = 25 } = {}) {
  if (!foldedStacks || !foldedStacks.length) {
    container.innerHTML = '<div class="empty">No callstack data in this run.</div>';
    return;
  }
  const total = foldedStacks.reduce((a, s) => a + s.count, 0) || 1;
  const top = foldedStacks.slice(0, limit);
  const max = top[0].count;

  const rows = top.map((s) => {
    const pct = ((s.count / total) * 100).toFixed(2);
    const barPct = (s.count / max) * 100;
    const frames = (s.stack || '').split(';').map((f, i, arr) => {
      const cls = colourFrame(f) + (i === arr.length - 1 ? ' cs-leaf' : '');
      return `<span class="${cls}">${escHtml(f || '[empty]')}</span>`;
    }).join('<span class="cs-sep">→</span>');

    return `
      <div class="callstack-row">
        <div class="callstack-count">${s.count.toLocaleString()}</div>
        <div class="callstack-bar-wrap"><div class="callstack-bar-fill" style="width:${barPct}%"></div></div>
        <div class="callstack-frames"><strong>${pct}%</strong>&nbsp; ${frames}</div>
      </div>
    `;
  }).join('');

  container.innerHTML = `
    <div class="card-body" style="padding:0 0 10px;">
      <div style="padding:10px 16px;font-size:0.72rem;color:#5a6880;">
        Top ${top.length} of ${foldedStacks.length.toLocaleString()} unique stacks
        &middot; total samples ${total.toLocaleString()}
      </div>
      ${rows}
    </div>`;
}
