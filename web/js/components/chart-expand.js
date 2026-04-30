// web/js/components/chart-expand.js
// Adds a fullscreen expand button to a .card container. When clicked, opens
// a modal with a fresh, larger render of the chart. When runsIndex is provided,
// dataset and thread filter chips are rendered above the chart.

const EXPAND_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

export function attachExpand(card, { title, badge, renderFn, runsIndex }) {
  if (!card || card.querySelector('.chart-expand-btn')) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'chart-expand-btn';
  btn.setAttribute('aria-label', `Expand ${title}`);
  btn.title = `Expand ${title}`;
  btn.innerHTML = EXPAND_SVG;

  const actions = card.querySelector('.card-header .actions');
  if (actions) actions.appendChild(btn);
  else card.appendChild(btn);

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    openModal({ title, badge, renderFn, runsIndex });
  });
}

/* ── Filter helpers ──────────────────────────────────────────── */

function buildFilterOptions(runsIndex) {
  const datasets = new Set();
  const threads = new Set();
  for (const r of runsIndex) {
    if (r.archived) continue;
    if (r.dataset_short) datasets.add(r.dataset_short);
    if (r.threads != null) threads.add(Number(r.threads));
  }
  return {
    datasets: [...datasets].sort(),
    threads: [...threads].sort((a, b) => a - b),
  };
}

function applyFilters(runsIndex, activeDatasets, activeThreads) {
  return runsIndex.filter(r => {
    if (r.dataset_short && !activeDatasets.has(r.dataset_short)) return false;
    if (r.threads != null && !activeThreads.has(Number(r.threads))) return false;
    return true;
  });
}

function renderFilterBar(container, opts, activeDatasets, activeThreads, rerender) {
  const makeGroup = (label, items, activeSet) => {
    const group = document.createElement('div');
    group.className = 'cm-filter-group';
    const lbl = document.createElement('span');
    lbl.className = 'cm-filter-label';
    lbl.textContent = label;
    group.appendChild(lbl);
    const chips = document.createElement('div');
    chips.className = 'cm-filter-chips';
    for (const item of items) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'cm-chip' + (activeSet.has(item) ? ' cm-chip--on' : '');
      chip.textContent = String(item);
      chip.addEventListener('click', () => {
        if (activeSet.has(item)) {
          if (activeSet.size > 1) { activeSet.delete(item); chip.classList.remove('cm-chip--on'); }
        } else {
          activeSet.add(item); chip.classList.add('cm-chip--on');
        }
        rerender();
      });
      chips.appendChild(chip);
    }
    group.appendChild(chips);
    return group;
  };

  if (opts.datasets.length > 1) container.appendChild(makeGroup('Dataset', opts.datasets, activeDatasets));
  if (opts.threads.length > 1)  container.appendChild(makeGroup('Threads', opts.threads, activeThreads));
}

/* ── Modal ───────────────────────────────────────────────────── */

function openModal({ title, badge, renderFn, runsIndex }) {
  const hasIndex = Array.isArray(runsIndex) && runsIndex.length > 0;
  const filterOpts = hasIndex ? buildFilterOptions(runsIndex) : null;
  const showFilters = filterOpts && (filterOpts.datasets.length > 1 || filterOpts.threads.length > 1);

  const modal = document.createElement('div');
  modal.className = 'chart-modal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.innerHTML = `
    <div class="chart-modal-inner">
      <div class="chart-modal-header">
        <div style="display:flex; align-items:center; gap:12px;">
          <h2>${escape(title || 'Chart')}</h2>
          ${badge ? `<span class="badge">${escape(badge)}</span>` : ''}
        </div>
        <button class="chart-modal-close" aria-label="Close">✕</button>
      </div>
      ${showFilters ? '<div class="chart-modal-filters"></div>' : ''}
      <div class="chart-modal-body"></div>
    </div>
  `;
  document.body.appendChild(modal);

  const close = () => { modal.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  modal.querySelector('.chart-modal-close').addEventListener('click', close);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
  document.addEventListener('keydown', onKey);

  const body = modal.querySelector('.chart-modal-body');
  const activeDatasets = new Set(filterOpts?.datasets || []);
  const activeThreads  = new Set(filterOpts?.threads  || []);

  const rerender = () => {
    const filtered = hasIndex ? applyFilters(runsIndex, activeDatasets, activeThreads) : runsIndex;
    body.innerHTML = '';
    requestAnimationFrame(() => {
      try { renderFn(body, filtered); }
      catch (err) {
        body.innerHTML = `<div class="empty">Failed to render: ${escape(String(err))}</div>`;
      }
    });
  };

  if (showFilters) {
    renderFilterBar(modal.querySelector('.chart-modal-filters'), filterOpts, activeDatasets, activeThreads, rerender);
  }
  rerender();
}

function escape(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
