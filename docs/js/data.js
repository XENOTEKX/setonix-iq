// web/js/data.js — fetch + cache

import { store } from './state.js?v=20260430105505';

const BASE = 'data';

async function getJson(path) {
  const res = await fetch(`${BASE}/${path}`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
  return res.json();
}

export async function loadManifest() {
  const m = await getJson('manifest.json').catch(() => null);
  store.set({ manifest: m });
  return m;
}

export async function loadIndexes() {
  const [runs, profiles] = await Promise.all([
    getJson('runs.index.json').catch(() => []),
    getJson('profiles.index.json').catch(() => []),
  ]);
  store.set({ runsIndex: runs, profilesIndex: profiles });
  return { runs, profiles };
}

export async function loadRun(id) {
  if (!id) return null;
  const cache = store.get('runs');
  if (cache.has(id)) return cache.get(id);
  const run = await getJson(`runs/${encodeURIComponent(id)}.json`);
  cache.set(id, run);
  return run;
}

export async function loadProfile(id) {
  if (!id) return null;
  const cache = store.get('profiles');
  if (cache.has(id)) return cache.get(id);
  const p = await getJson(`profiles/${encodeURIComponent(id)}.json`);
  cache.set(id, p);
  return p;
}
