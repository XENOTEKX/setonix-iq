// web/js/data.js — fetch + cache

import { store } from './state.js?v=20260430160708';

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

/**
 * loadRunProfile — lazily fetches the heavy-blob companion file
 * (docs/data/runs/<id>.profile.json) that build.py splits out from the main
 * run JSON. Contains folded_stacks and memory_timeseries.
 *
 * Merges the blobs into the cached run object so subsequent callers of
 * loadRun() see them without a second fetch.
 */
export async function loadRunProfile(id) {
  if (!id) return null;
  const run = await loadRun(id);
  if (!run) return null;
  // Already merged (or was never split — development build)
  if (run.profile && run.profile.folded_stacks !== undefined) return run;

  const profCache = store.get('runProfiles') || new Map();
  if (!store.get('runProfiles')) store.set({ runProfiles: profCache });
  if (profCache.has(id)) return profCache.get(id);

  const blobs = await getJson(`runs/${encodeURIComponent(id)}.profile.json`).catch(() => null);
  if (blobs && run.profile) {
    Object.assign(run.profile, blobs);
  }
  profCache.set(id, run);
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
