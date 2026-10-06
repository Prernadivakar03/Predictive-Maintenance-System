import { useState } from "react";
import { LogView } from "../components/JobPanel.jsx";
import { Async, Badge, Button, KV, PageHead, Panel, Segmented } from "../components/ui.jsx";
import { useToast } from "../components/Toast.jsx";
import { api } from "../lib/api.js";
import { dash, duration, pct } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";

export default function System() {
  const toast = useToast();
  const system = useApi("/api/system");
  const [level, setLevel] = useState("ALL");
  const [auto, setAuto] = useState(true);
  const logs = useApi("/api/logs?lines=300", { interval: auto ? 5000 : 0 });
  const [busy, setBusy] = useState(false);
  const health = useApi("/api/health", { interval: 10000 });

  const reload = async () => {
    setBusy(true);
    try { const r = await api.post("/api/model/reload"); toast(`Reloaded ${r.model_name} v${r.model_version}.`, "ok"); health.reload(); }
    catch (e) { toast(e.message, "bad"); } finally { setBusy(false); }
  };

  const shown = (logs.data?.lines || []).filter((l) => level === "ALL" || (level === "WARN+" ? ["WARNING", "ERROR", "CRITICAL"].includes(l.level) : l.level === level))
    .map((l) => ({ ts: l.ts.replace(" ", "T"), level: l.level, message: `[${l.level}] ${l.logger.split(":")[0]}: ${l.message}` }));

  return (
    <>
      <PageHead title="System" actions={<Button icon="refresh" busy={busy} onClick={reload}>Reload model</Button>}>Service health, environment, configuration and the application log.</PageHead>
      <div className="grid cols-2">
        <Panel title="Service" right={health.data ? <Badge tone="ok">Healthy</Badge> : health.error ? <Badge tone="bad">{health.error.status === 0 ? "Offline" : "Unhealthy"}</Badge> : null}>
          <KV items={[
            ["Model", health.data ? `${health.data.model_name} v${health.data.model_version}` : health.error?.message || dash],
            ["Environment", system.data?.environment || dash],
            ["Uptime", system.data ? duration(system.data.uptime_s) : dash],
            ["Python", system.data?.python], ["Platform", system.data?.platform],
          ]} />
        </Panel>
        <Panel title="Links">
          <Async state={system}>{(s) => (
            <div className="stack">
              <a href="/docs" target="_blank" rel="noreferrer">Interactive API documentation (Swagger)</a>
              <a href="/redoc" target="_blank" rel="noreferrer">API reference (ReDoc)</a>
              <a href={s.links.mlflow} target="_blank" rel="noreferrer">MLflow tracking UI</a>
              <a href={s.links.airflow} target="_blank" rel="noreferrer">Airflow scheduler UI</a>
              <span className="muted small">MLflow and Airflow are separate services; start them as described in the README.</span>
            </div>
          )}</Async>
        </Panel>
      </div>
      <Async state={system}>
        {(s) => (
          <div className="grid cols-3">
            <Panel title="Packages">
              <KV items={Object.entries(s.packages).map(([k, v]) => [k, v ? v : <span className="faint" key={k}>not installed</span>])} />
            </Panel>
            <Panel title="Files and stores">
              <div className="stack">{s.paths.map((p) => (
                <div key={p.label}><div className="row between"><strong>{p.label}</strong><Badge tone={p.exists ? "ok" : "warn"}>{p.exists ? "Found" : "Missing"}</Badge></div><div className="mono faint small" style={{ wordBreak: "break-all" }}>{p.path}</div></div>
              ))}</div>
            </Panel>
            <Panel title="Active thresholds" sub="config/config.yaml">
              <KV items={[
                ["Min recall", pct(s.thresholds.min_acceptable_recall, 0)], ["Min F1", s.thresholds.min_acceptable_f1],
                ["Drift p-value", s.thresholds.p_value_alpha], ["Max drifted features", pct(s.thresholds.max_drifted_feature_ratio, 0)],
                ["Prediction shift limit", s.thresholds.wasserstein_prob_drift_limit],
                ["Gate: min recall", pct(s.promotion_criteria.min_recall_threshold, 0)], ["Gate: precision drop", `${s.promotion_criteria.max_precision_drop_pct} pts`],
              ]} />
            </Panel>
          </div>
        )}
      </Async>
      <Panel title="Application log" sub={logs.data?.path}
        right={<><Segmented label="Log level" value={level} onChange={setLevel} options={[{ value: "ALL", label: "All" }, { value: "INFO", label: "Info" }, { value: "WARN+", label: "Warnings and errors" }]} />
          <label className="check"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Auto-refresh</label></>} flush>
        <LogView lines={shown} follow={auto} />
      </Panel>
    </>
  );
}
