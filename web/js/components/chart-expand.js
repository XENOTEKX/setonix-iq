// web/js/components/chart-expand.js
// Adds a fullscreen expand button to a .card container. When clicked, opens
// a modal with a fresh, larger render of the chart by re-invoking the given
// renderFn with a larger canvas target.

const EXPAND_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

export function attachExpand(card, { title, badge, renderFn }) {
  if (!card || card.querySelector('.chart-expand-btn')) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'chart-expand-btn';
  btn.setAttribute('aria-label', `Expand ${title}`);
  btn.innerHTML = EXPAND_SVG;
  card.appendChild(btn);

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    openModal({ title, badge, renderFn });
  });
}

function openModal({ title, badge, renderFn }) {
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
      <div class="chart-modal-body"></div>
    </div>
  `;
  document.body.appendChild(modal);

  const close = () => {
    modal.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (e) => { if (e.key === 'Escape') close(); };

  modal.querySelector('.chart-modal-close').addEventListener('click', close);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) close();
  });
  document.addEventListener('keydown', onKey);

  const body = modal.querySelector('.chart-modal-body');
  // Allow layout to settle before rendering
  requestAnimationFrame(() => {
    try { renderFn(body); } catch (err) {
      body.innerHTML = `<div class="empty">Failed to render: ${escape(String(err))}</div>`;
    }
  });
}

function escape(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
