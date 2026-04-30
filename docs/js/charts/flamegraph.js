// web/js/charts/flamegraph.js — minimal flamegraph from folded_stacks

import { escHtml, hashColour, shortFn } from '../utils.js?v=20260430153241';

export function render(container, foldedStacks) {
  if (!foldedStacks || !foldedStacks.length) {
    container.innerHTML = '<div class="empty">No stacks to render.</div>';
    return;
  }

  // Build hierarchical tree
  const root = { name: 'root', value: 0, children: {} };
  for (const s of foldedStacks) {
    const frames = (s.stack || '').split(';').filter(Boolean);
    let node = root;
    node.value += s.count;
    for (const f of frames) {
      if (!node.children[f]) node.children[f] = { name: f, value: 0, children: {} };
      node = node.children[f];
      node.value += s.count;
    }
  }

  const totalSamples = root.value || 1;
  const maxDepth = 20;
  const rowsByDepth = [];

  function walk(node, depth, xPct, widthPct) {
    if (depth >= maxDepth || widthPct < 0.15) return;
    if (depth > 0) {
      if (!rowsByDepth[depth - 1]) rowsByDepth[depth - 1] = [];
      rowsByDepth[depth - 1].push({
        name: node.name,
        value: node.value,
        xPct,
        widthPct,
      });
    }
    const kids = Object.values(node.children).sort((a, b) => b.value - a.value);
    let childX = xPct;
    for (const k of kids) {
      const childWidth = (k.value / node.value) * widthPct;
      walk(k, depth + 1, childX, childWidth);
      childX += childWidth;
    }
  }
  walk(root, 0, 0, 100);

  container.innerHTML = `
    <div class="flame-banner">
      <strong>Flamegraph</strong> — X axis = share of samples; Y axis = stack depth.
      Click frames in your profiler for full detail.
    </div>
    <div class="flamegraph-container" style="position:relative;height:${rowsByDepth.length * 17}px;">
      ${rowsByDepth.map((row, depth) => `
        <div class="flame-row" style="position:absolute;top:${depth * 17}px;left:0;right:0;height:16px;">
          ${row.map((f) => {
            const pct = ((f.value / totalSamples) * 100).toFixed(2);
            return `
              <div class="flame-frame"
                   style="position:absolute;left:${f.xPct}%;width:${f.widthPct}%;background:${hashColour(f.name, 1)};"
                   title="${escHtml(f.name)} — ${pct}% (${f.value.toLocaleString()} samples)">
                ${escHtml(shortFn(f.name))}
              </div>`;
          }).join('')}
        </div>`).join('')}
    </div>`;
}
