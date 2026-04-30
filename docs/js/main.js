// web/js/main.js — entry point

import { store } from './state.js?v=20260430151648';
import { loadManifest, loadIndexes } from './data.js?v=20260430151648';
import * as router from './router.js?v=20260430151648';

import * as overview from './pages/overview.js?v=20260430151648';
import * as runs from './pages/runs.js?v=20260430151648';
import * as tests from './pages/tests.js?v=20260430151648';
import * as profiling from './pages/profiling.js?v=20260430151648';
import * as gpu from './pages/gpu.js?v=20260430151648';
import * as environment from './pages/environment.js?v=20260430151648';

const PAGE_MOUNT = {
  overview: overview.mount,
  runs: runs.mount,
  tests: tests.mount,
  profiling: profiling.mount,
  gpu: gpu.mount,
  environment: environment.mount,
};

const MAIN = () => document.getElementById('main');

function registerRoutes() {
  for (const [name, mount] of Object.entries(PAGE_MOUNT)) {
    router.register(name, async (ctx) => {
      store.set({ currentPage: name });
      try {
        await mount(MAIN(), ctx);
      } catch (err) {
        console.error(err);
        MAIN().innerHTML = `
          <div class="card"><div class="card-body">
            <h2>Something went wrong rendering this page.</h2>
            <pre style="color:var(--red);margin-top:10px;">${err.message}</pre>
            <p style="margin-top:10px;color:var(--text3);font-size:0.78rem;">Open DevTools for a full stack trace.</p>
          </div></div>`;
      }
    });
  }
}

async function boot() {
  try {
    await Promise.all([loadManifest(), loadIndexes()]);
  } catch (err) {
    MAIN().innerHTML = `
      <div class="card"><div class="card-body">
        <h2>Failed to load dashboard data</h2>
        <p style="color:var(--text3);margin-top:10px;">Make sure <code>data/runs.index.json</code> exists.
        Run <code>python3 tools/build.py</code> locally or push to trigger the build workflow.</p>
        <pre style="color:var(--red);margin-top:10px;">${err.message}</pre>
      </div></div>`;
    return;
  }

  // Seed footer status
  const manifest = store.get('manifest');
  const footer = document.getElementById('buildInfo');
  if (footer && manifest) {
    footer.textContent = `${manifest.runs} runs · ${manifest.profiles} profiles · built ${manifest.generated_at?.slice(0, 16) || '—'}`;
  }

  registerRoutes();
  setupSidebarToggle();
  setupMobileDrawer();
  router.start();
}

function setupSidebarToggle() {
  const layout = document.getElementById('layout');
  const btn = document.getElementById('sidebarToggle');
  if (!layout || !btn) return;
  const KEY = 'sidebar.collapsed';
  const saved = localStorage.getItem(KEY) === '1';
  if (saved) layout.classList.add('sidebar-collapsed');
  const apply = () => {
    const collapsed = layout.classList.toggle('sidebar-collapsed');
    localStorage.setItem(KEY, collapsed ? '1' : '0');
    btn.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
  };
  btn.addEventListener('click', apply);
  // ⌘B / Ctrl-B shortcut
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'b' || e.key === 'B')) {
      e.preventDefault();
      apply();
    }
  });
}

function setupMobileDrawer() {
  const layout = document.getElementById('layout');
  const menuBtn = document.getElementById('menuToggle');
  const backdrop = document.getElementById('sidebarBackdrop');
  const sidebar = document.getElementById('sidebar');
  if (!layout || !menuBtn) return;

  const open = () => {
    layout.classList.add('sidebar-open');
    document.body.classList.add('drawer-open');
    menuBtn.setAttribute('aria-expanded', 'true');
  };
  const close = () => {
    layout.classList.remove('sidebar-open');
    document.body.classList.remove('drawer-open');
    menuBtn.setAttribute('aria-expanded', 'false');
  };
  const toggle = () => {
    if (layout.classList.contains('sidebar-open')) close();
    else open();
  };

  menuBtn.addEventListener('click', toggle);
  backdrop?.addEventListener('click', close);

  // Close when tapping a nav link inside the drawer
  sidebar?.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-page]');
    if (a) close();
  });

  // Close on Esc
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && layout.classList.contains('sidebar-open')) close();
  });

  // Close on hashchange (route change) and on resize up from mobile
  window.addEventListener('hashchange', close);
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768) close();
  });
}

boot();
