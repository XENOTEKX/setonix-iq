// web/js/pages/allocation.js

export function mount(root) {
  root.innerHTML = `
    <div class="page-header"><div><h1>Allocation</h1>
    <div class="subtitle">Pawsey SU balance</div></div></div>
    <div class="card"><div class="card-body">
      <div class="empty">Live allocation data is fetched on Setonix via <code>pawseyAccountBalance</code>.
      Run <code>./start.sh status</code> on the login node.</div>
    </div></div>
  `;
}
