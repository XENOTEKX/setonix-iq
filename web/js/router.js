// web/js/router.js — hash-based routing

const routes = new Map();

export function register(name, handler) {
  routes.set(name, handler);
}

export function go(name, replace = false) {
  const hash = `#/${name}`;
  if (replace) history.replaceState(null, '', hash);
  else location.hash = hash;
}

function current() {
  const m = location.hash.match(/^#\/?([^?]*)/);
  return (m && m[1]) || 'overview';
}

function currentQuery() {
  const i = location.hash.indexOf('?');
  if (i < 0) return {};
  const out = {};
  for (const pair of location.hash.slice(i + 1).split('&')) {
    if (!pair) continue;
    const [k, v = ''] = pair.split('=');
    out[decodeURIComponent(k)] = decodeURIComponent(v);
  }
  return out;
}

async function dispatch() {
  const name = current();
  const query = currentQuery();
  const links = document.querySelectorAll('[data-page]');
  links.forEach((a) => {
    if (a.dataset.page === name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  const handler = routes.get(name) || routes.get('overview');
  if (handler) await handler({ query });
}

export function start() {
  window.addEventListener('hashchange', dispatch);
  document.addEventListener('click', (e) => {
    const a = e.target.closest('[data-page]');
    if (a) {
      e.preventDefault();
      go(a.dataset.page);
    }
  });
  dispatch();
}
