import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

/**
 * Fetches `path` and optionally polls. Keeps the previous data visible while refreshing.
 * Returns { data, error, loading, reload }.
 */
export function useApi(path, { interval = 0, enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: enabled });
  const [tick, setTick] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  useEffect(() => {
    if (!enabled || !path) return undefined;
    const controller = new AbortController();
    let timer;

    const load = async (silent) => {
      if (!silent) setState((s) => ({ ...s, loading: true }));
      try {
        const data = await api.get(path, { signal: controller.signal });
        if (alive.current) setState({ data, error: null, loading: false });
      } catch (error) {
        if (error.name === "AbortError") return;
        if (alive.current) setState((s) => ({ data: s.data, error, loading: false }));
      }
    };

    load(false);
    if (interval > 0) timer = setInterval(() => load(true), interval);
    return () => { controller.abort(); clearInterval(timer); };
  }, [path, interval, enabled, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

/** Polls a background job until it finishes. */
export function useJob(jobId, onFinish) {
  const { data: job, reload } = useApi(jobId ? `/api/jobs/${jobId}` : null, {
    interval: jobId ? 1500 : 0,
    enabled: Boolean(jobId),
  });
  const finished = useRef(null);
  useEffect(() => {
    if (job && job.status !== "running" && finished.current !== job.id) {
      finished.current = job.id;
      onFinish?.(job);
    }
  }, [job, onFinish]);
  return { job: jobId ? job : null, reload };
}

export function useLocalStorage(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw !== null ? JSON.parse(raw) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
  }, [key, value]);
  return [value, setValue];
}

/** Element width via ResizeObserver, for charts that draw at real pixel size. */
export function useWidth(initial = 600) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => setWidth(Math.max(120, Math.round(el.getBoundingClientRect().width)));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}
