// web/js/state.js — tiny reactive store

function createStore(initial) {
  const state = { ...initial };
  const listeners = new Set();

  return {
    get(key) {
      return key == null ? state : state[key];
    },
    set(patch) {
      Object.assign(state, patch);
      listeners.forEach((fn) => fn(state));
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

export const store = createStore({
  manifest: null,
  runsIndex: [],        // summaries, always loaded
  profilesIndex: [],    // summaries, always loaded
  runs: new Map(),      // id → full run (lazy)
  profiles: new Map(),  // id → full profile (lazy)
  selectedRunId: null,
  currentPage: 'overview',
});
