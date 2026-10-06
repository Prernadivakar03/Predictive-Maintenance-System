import { createContext, useContext, useEffect, useState } from "react";
import Icon from "./components/Icon.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import { ToastProvider } from "./components/Toast.jsx";
import { Badge } from "./components/ui.jsx";
import { useApi } from "./lib/hooks.js";
import { href, useRoute } from "./lib/router.js";
import Overview from "./pages/Overview.jsx";
import Predict from "./pages/Predict.jsx";
import Model from "./pages/Model.jsx";
import Experiments from "./pages/Experiments.jsx";
import Pipeline from "./pages/Pipeline.jsx";
import Monitoring from "./pages/Monitoring.jsx";
import Retraining from "./pages/Retraining.jsx";
import DataPage from "./pages/Data.jsx";
import System from "./pages/System.jsx";

const ROUTES = [
  { path: "/", label: "Overview", icon: "overview", Page: Overview },
  { path: "/predict", label: "Predict", icon: "predict", Page: Predict },
  { path: "/model", label: "Model", icon: "model", Page: Model },
  { path: "/experiments", label: "Experiments", icon: "experiments", Page: Experiments },
  { path: "/pipeline", label: "Pipeline", icon: "pipeline", Page: Pipeline },
  { path: "/monitoring", label: "Monitoring", icon: "monitoring", Page: Monitoring },
  { path: "/retraining", label: "Retraining", icon: "retraining", Page: Retraining },
  { path: "/data", label: "Data", icon: "data", Page: DataPage },
  { path: "/system", label: "System", icon: "system", Page: System },
];

const OverviewContext = createContext(null);
export const useOverview = () => useContext(OverviewContext);

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("pm-theme"); } catch { return null; }
  });
  const effective = theme || (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const toggle = () => {
    const next = effective === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("pm-theme", next); } catch { /* ignore */ }
  };
  return [effective, toggle];
}

function Brand() {
  return (
    <div className="brand">
      <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="6" fill="#0c1721" stroke="#263646" />
        <path d="M16 5.5l8.6 5v10l-8.6 5-8.6-5v-10z" fill="none" stroke="#3fcbb0" strokeWidth="2.2" strokeLinejoin="round" />
        <path d="M9.5 16h3.2l1.6-3.6 2.6 7.2 1.8-3.6h3.8" fill="none" stroke="#f2b134" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div>
        <div className="brand-name">Predictive Maintenance</div>
        <div className="brand-sub">MLOps console</div>
      </div>
    </div>
  );
}

export default function App() {
  const route = useRoute();
  const [open, setOpen] = useState(false);
  const [theme, toggleTheme] = useTheme();
  const overview = useApi("/api/overview", { interval: 20000 });
  const health = useApi("/health", { interval: 15000 });

  useEffect(() => setOpen(false), [route]);

  const active = ROUTES.find((r) => r.path === route) || ROUTES[0];
  const Page = active.Page;
  const needsRetrain = overview.data?.monitoring?.retraining_required;

  let apiBadge = <Badge tone="plain">Checking</Badge>;
  if (health.data) apiBadge = <Badge tone="ok">API healthy</Badge>;
  else if (health.error) apiBadge = <Badge tone="bad">{health.error.status === 0 ? "API offline" : "Model unavailable"}</Badge>;

  return (
    <ToastProvider>
      <OverviewContext.Provider value={overview}>
        <div className="app">
          <aside className={`rail ${open ? "open" : ""}`} aria-label="Primary">
            <Brand />
            <nav className="nav">
              {ROUTES.map((r) => (
                <a key={r.path} href={href(r.path)} aria-current={r.path === active.path ? "page" : undefined}>
                  <Icon name={r.icon} /> {r.label}
                  {r.path === "/retraining" && needsRetrain && <span className="nav-flag" title="Retraining recommended" />}
                </a>
              ))}
            </nav>
            <div className="rail-foot">
              <a href="/docs" target="_blank" rel="noreferrer">API documentation</a>
              <span>v{overview.data?.service?.api_version || "1.0.0"}</span>
            </div>
          </aside>

          <div className="main">
            <header className="topbar">
              <button className="btn icon-btn menu-btn" onClick={() => setOpen((o) => !o)} aria-label="Toggle navigation"><Icon name="menu" /></button>
              <strong style={{ fontFamily: "var(--font-display)", fontSize: 17 }}>{active.label}</strong>
              <span className="spacer" />
              {apiBadge}
              {health.data && <span className="hide-sm"><Badge tone="plain">Model v{health.data.model_version}</Badge></span>}
              <button className="btn icon-btn" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
                <Icon name={theme === "dark" ? "sun" : "moon"} />
              </button>
            </header>
            <main className="page" key={active.path}>
              <ErrorBoundary key={active.path}><Page /></ErrorBoundary>
            </main>
          </div>
        </div>
      </OverviewContext.Provider>
    </ToastProvider>
  );
}
