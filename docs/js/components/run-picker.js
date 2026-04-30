// web/js/components/run-picker.js
// Searchable combobox for selecting a run. Replaces the plain <select>.
//
// Usage:
//   mountRunPicker(container, runs, { selectedId, onChange })
//
// Renders a trigger button that, when clicked, opens a panel with a search
// input and a list of all runs showing key stats. Click-out and Escape close.

function fmtNum(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  const v = Number(n);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return v.toFixed(digits);
}
function fmtWall(s) {
  if (!s) return '—';
  if (s < 60) return `${s.toFixed(1)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(2)}h`;
}

export function mountRunPicker(container, runs, { selectedId, onChange } = {}) {
  const sorted = [...runs].sort((a, b) => {
    // group by platform, then dataset, then threads ascending
    const pa = (a.platform || 'zzz');
    const pb = (b.platform || 'zzz');
    if (pa !== pb) return pa.localeCompare(pb);
    const da = a.dataset_short || a.dataset || '';
    const db = b.dataset_short || b.dataset || '';
    if (da !== db) return da.localeCompare(db);
    return (a.threads || 0) - (b.threads || 0);
  });
  let current = sorted.find(r => r.run_id === selectedId) || sorted[0];
  let query = '';
  let activeIdx = 0;
  let open = false;
  // Caret position to restore after each render() (rebuilding innerHTML
  // destroys the live input element, so we must re-focus + restore caret
  // on the freshly-rendered input or typing drops every other keystroke).
  let savedCaret = 0;
  // Filter state. null = "All"; values are matched against r.platform,
  // r.dataset_short, r.threads respectively.
  let filterPlatform = null;
  let filterDataset = null;
  let filterThreads = null;

  // Build unique sets of filter values from the data.
  const uniqPlatforms = [...new Set(sorted.map(r => r.platform).filter(Boolean))].sort();
  const uniqDatasets = [...new Set(sorted.map(r => r.dataset_short).filter(Boolean))].sort();
  const uniqThreads  = [...new Set(sorted.map(r => r.threads).filter(t => t != null))]
    .sort((a, b) => a - b);

  const platLabel = (p) => {
    if (!p) return p;
    return p.charAt(0).toUpperCase() + p.slice(1);
  };

  const root = document.createElement('div');
  root.className = 'run-picker';
  container.replaceChildren(root);

  const render = () => {
    const q = query.trim().toLowerCase();
    const filtered = sorted.filter(r => {
      if (filterPlatform && r.platform !== filterPlatform) return false;
      if (filterDataset && r.dataset_short !== filterDataset) return false;
      if (filterThreads != null && r.threads !== filterThreads) return false;
      if (!q) return true;
      return (
        (r.run_id || '').toLowerCase().includes(q) ||
        (r.dataset_short || '').toLowerCase().includes(q) ||
        (r.label || '').toLowerCase().includes(q) ||
        (r.description || '').toLowerCase().includes(q) ||
        (r.platform || '').toLowerCase().includes(q) ||
        String(r.threads || '').includes(q)
      );
    });

    if (activeIdx >= filtered.length) activeIdx = Math.max(0, filtered.length - 1);

    const chip = (label, isActive, onClickAttr) =>
      `<button type="button" class="rp-chip${isActive ? ' active' : ''}" ${onClickAttr}>${label}</button>`;

    const platChips = [chip('All', filterPlatform === null, 'data-filter="platform" data-value=""')]
      .concat(uniqPlatforms.map(p =>
        chip(platLabel(p), filterPlatform === p, `data-filter="platform" data-value="${p}"`)))
      .join('');
    const dsChips = [chip('All', filterDataset === null, 'data-filter="dataset" data-value=""')]
      .concat(uniqDatasets.map(d =>
        chip(d, filterDataset === d, `data-filter="dataset" data-value="${d}"`)))
      .join('');
    const thChips = [chip('All', filterThreads === null, 'data-filter="threads" data-value=""')]
      .concat(uniqThreads.map(t =>
        chip(`${t}T`, filterThreads === t, `data-filter="threads" data-value="${t}"`)))
      .join('');

    root.classList.toggle('open', open);
    root.innerHTML = `
      <button class="run-picker-trigger" type="button" aria-haspopup="listbox" aria-expanded="${open}">
        <span class="rp-badge">RUN</span>
        <div class="rp-current">
          <span class="rp-label">${current?.run_id || '—'}</span>
          <div class="rp-stats">
            <div class="rp-stat"><span class="rp-stat-k">Dataset</span><span class="rp-stat-v">${current?.dataset_short || '—'}</span></div>
            <div class="rp-stat"><span class="rp-stat-k">Threads</span><span class="rp-stat-v">${current?.threads || '—'}</span></div>
            <div class="rp-stat"><span class="rp-stat-k">Wall</span><span class="rp-stat-v">${fmtWall(current?.wall_s)}</span></div>
            <div class="rp-stat"><span class="rp-stat-k">IPC</span><span class="rp-stat-v accent">${fmtNum(current?.IPC)}</span></div>
          </div>
        </div>
        <svg class="rp-chevron" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </button>
      <div class="run-picker-panel" role="listbox">
        <div class="rp-search">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--text3);"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" placeholder="Search by id, dataset, threads…" value="${query.replace(/"/g, '&quot;')}" autocomplete="off" />
          <span class="rp-hint">${filtered.length} of ${sorted.length}</span>
        </div>
        <div class="rp-filters">
          <div class="rp-filter-row"><span class="rp-filter-k">Platform</span><div class="rp-chip-row">${platChips}</div></div>
          <div class="rp-filter-row"><span class="rp-filter-k">Dataset</span><div class="rp-chip-row">${dsChips}</div></div>
          <div class="rp-filter-row"><span class="rp-filter-k">Threads</span><div class="rp-chip-row">${thChips}</div></div>
        </div>
        <div class="rp-list">
          ${filtered.length === 0
            ? '<div class="rp-empty">No runs match your search</div>'
            : filtered.map((r, i) => `
              <div class="rp-option ${r.run_id === current?.run_id ? 'selected' : ''} ${i === activeIdx ? 'active' : ''} ${r.all_pass ? '' : 'fail'}" data-idx="${i}" role="option">
                <span class="rp-dot" aria-hidden="true"></span>
                <div class="rp-opt-main">
                  <span class="rp-opt-label">${r.run_id}</span>
                  <span class="rp-opt-sub">${r.dataset_short || '—'} · ${r.model || '—'}</span>
                </div>
                <span class="rp-opt-num wall">${fmtWall(r.wall_s)}</span>
                <span class="rp-opt-num threads">${r.threads || '—'}T</span>
                <span class="rp-opt-num ipc">${fmtNum(r.IPC)}</span>
              </div>
            `).join('')}
        </div>
      </div>
    `;

    const trigger = root.querySelector('.run-picker-trigger');
    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      open = !open;
      render();
      if (open) setTimeout(() => root.querySelector('.rp-search input')?.focus(), 20);
    });

    const panel = root.querySelector('.run-picker-panel');
    panel?.addEventListener('click', (e) => e.stopPropagation());

    root.querySelectorAll('.rp-chip').forEach(el => {
      el.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const dim = el.dataset.filter;
        const raw = el.dataset.value;
        const val = raw === '' ? null : raw;
        if (dim === 'platform') filterPlatform = val;
        else if (dim === 'dataset') filterDataset = val;
        else if (dim === 'threads') filterThreads = (val == null ? null : Number(val));
        activeIdx = 0;
        render();
      });
    });

    const input = root.querySelector('.rp-search input');
    if (input) {
      // If the panel is open, restore focus + caret to the search input.
      // The previous input element is destroyed each time render() rebuilds
      // root.innerHTML, so without this the user loses focus after every
      // keystroke ("can only type one character at a time"). We restore the
      // caret to the end (or to the saved offset, when available) so typing
      // continues seamlessly.
      if (open && document.activeElement !== input) {
        const caret = (typeof savedCaret === 'number') ? savedCaret : query.length;
        input.focus();
        try { input.setSelectionRange(caret, caret); } catch (_) { /* readonly types */ }
      }
      input.addEventListener('input', (e) => {
        query = e.target.value;
        savedCaret = e.target.selectionStart ?? query.length;
        activeIdx = 0;
        render();
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); activeIdx = Math.min(activeIdx + 1, filtered.length - 1); render(); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); activeIdx = Math.max(activeIdx - 1, 0); render(); }
        else if (e.key === 'Enter') { e.preventDefault(); pick(filtered[activeIdx]); }
        else if (e.key === 'Escape') { open = false; render(); }
      });
    }

    root.querySelectorAll('.rp-option').forEach(el => {
      el.addEventListener('click', () => {
        const i = Number(el.dataset.idx);
        pick(filtered[i]);
      });
      el.addEventListener('mouseenter', () => {
        activeIdx = Number(el.dataset.idx);
        root.querySelectorAll('.rp-option').forEach(x => x.classList.remove('active'));
        el.classList.add('active');
      });
    });
  };

  const pick = (r) => {
    if (!r) return;
    current = r;
    open = false;
    render();
    onChange?.(r);
  };

  const onDocClick = (e) => {
    if (open && !root.contains(e.target)) { open = false; render(); }
  };
  document.addEventListener('click', onDocClick);

  render();

  return {
    destroy: () => document.removeEventListener('click', onDocClick),
    setSelected: (id) => { const r = sorted.find(x => x.run_id === id); if (r) { current = r; render(); } },
  };
}
