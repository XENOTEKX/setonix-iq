// web/js/pages/tests.js — aggregated verification across all runs

import { store } from '../state.js?v=20260430105505';
import { loadRun } from '../data.js?v=20260430105505';
import { escHtml, fmtNum } from '../utils.js?v=20260430105505';

export async function mount(root) {
  root.innerHTML = `
    <div class="page-header"><div><h1>Tests &amp; Verification</h1>
    <div class="subtitle">Per-dataset likelihood verification across all runs</div></div></div>
    <div class="card"><div id="testsTable"></div></div>
  `;

  const idx = store.get('runsIndex').filter((r) => r.pass + r.fail > 0);
  if (idx.length === 0) {
    document.getElementById('testsTable').innerHTML =
      '<div class="empty">No runs have verification data yet.</div>';
    return;
  }

  const runs = await Promise.all(idx.map((r) => loadRun(r.run_id)));
  const flat = [];
  for (const r of runs) {
    for (const v of r.verify || []) {
      flat.push({ run: r.run_id, ...v });
    }
  }
  if (flat.length === 0) {
    document.getElementById('testsTable').innerHTML =
      '<div class="empty">No verification records.</div>';
    return;
  }
  document.getElementById('testsTable').innerHTML = `
    <div class="table-scroll">
      <table class="data-table">
        <thead><tr><th>Run</th><th>File</th><th>Status</th><th>Expected</th><th>Reported</th><th>|Δ|</th></tr></thead>
        <tbody>${flat.map((v) => `
          <tr>
            <td style="font-family:var(--font-mono);font-size:0.72rem;">${escHtml(v.run)}</td>
            <td>${escHtml(v.file)}</td>
            <td><span class="badge badge-${v.status === 'pass' ? 'pass' : 'fail'}">${v.status}</span></td>
            <td>${fmtNum(v.expected, 4)}</td>
            <td>${fmtNum(v.reported, 4)}</td>
            <td>${fmtNum(v.diff, 6)}</td>
          </tr>`).join('')}</tbody>
      </table>
    </div>`;
}
