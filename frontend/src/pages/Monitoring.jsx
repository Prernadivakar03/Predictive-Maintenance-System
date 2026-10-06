import { useRef, useState } from "react";
import { Async, Badge, Banner, Button, Empty, KV, PageHead, Panel, Stats } from "../components/ui.jsx";
import { Bullet, ConfusionMatrix, HBars, LineChart } from "../components/Charts.jsx";
import { useToast } from "../components/Toast.jsx";
import { useOverview } from "../App.jsx";
import { api } from "../lib/api.js";
import { parseCsv } from "../lib/csv.js";
import { dash, fixed, num, pct, relTime } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";
import { href } from "../lib/router.js";

const SIMULATED = ["sensor_drift", "thermal_drift", "wear_shift", "combined"];

function Controls({ status, onResult }) {
  const toast = useToast();
  const [scenario, setScenario] = useState("baseline");
  const [severity, setSeverity] = useState(0.6);
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState(null);
  const fileRef = useRef(null);
  const meta = status.scenarios.find((s) => s.id === scenario);

  const pickFile = async (file) => {
    try {
      const rows = parseCsv(await file.text());
      setUpload({ name: file.name, rows });
    } catch (e) { toast(e.message, "bad"); }
  };

  const run = async () => {
    setBusy(true);
    try {
      const result = await api.post("/api/monitoring/run", { scenario, severity, rows: scenario === "upload" ? upload?.rows : undefined });
      onResult(result);
      toast("Monitoring check complete.", "ok");
    } catch (e) { toast(e.message, "bad"); } finally { setBusy(false); }
  };

  return (
    <Panel title="Run a check" sub="Compare a batch of sensor data with the training baseline">
      <div className="grid cols-3" style={{ alignItems: "end" }}>
        <div className="field">
          <label htmlFor="scenario">Data to check</label>
          <select id="scenario" className="select" value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {status.scenarios.map((s) => (
              <option key={s.id} value={s.id} disabled={s.id === "live" && status.live_samples < 30}>
                {s.label}{s.id === "live" ? ` (${status.live_samples} logged)` : ""}
              </option>
            ))}
          </select>
        </div>
        {SIMULATED.includes(scenario) && (
          <div className="field">
            <label htmlFor="severity">Severity: {Math.round(severity * 100)}%</label>
            <input id="severity" type="range" min="0" max="1" step="0.05" value={severity} onChange={(e) => setSeverity(Number(e.target.value))} />
          </div>
        )}
        {scenario === "upload" && (
          <div className="field">
            <label>Sensor rows (CSV)</label>
            <Button icon="upload" onClick={() => fileRef.current?.click()}>{upload ? `${upload.name} (${upload.rows.length} rows)` : "Choose CSV file"}</Button>
            <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => e.target.files[0] && pickFile(e.target.files[0])} />
          </div>
        )}
        <div><Button variant="primary" icon="play" busy={busy} disabled={scenario === "upload" && !upload} onClick={run}>Run check</Button></div>
      </div>
      <p className="muted small" style={{ marginTop: 12 }}>{meta?.description}{scenario === "live" && status.live_samples < 30 ? " At least 30 logged predictions are needed." : ""}</p>
    </Panel>
  );
}

function Result({ s }) {
  const { summary, report, thresholds, context } = s;
  const dd = report.data_drift, pd = report.prediction_drift, perf = report.performance_degradation;
  const features = Object.entries(dd.feature_details).map(([name, v]) => ({ name, ...v })).sort((a, b) => b.ks_statistic - a.ks_statistic);
  const maxKs = Math.max(...features.map((f) => f.ks_statistic), 0.05) * 1.15;
  const m = perf.metrics;

  return (
    <>
      <Banner tone={summary.retraining_required ? "warn" : "ok"}
        title={summary.retraining_required ? "Retraining is recommended" : "No action needed"}
        action={summary.retraining_required && <a className="btn sm" href={href("/retraining")}>Go to retraining</a>}>
        Checked {context?.label || "validation set"}{context?.rows ? `, ${context.rows.toLocaleString()} rows` : ""}{context?.severity != null ? ` at ${Math.round(context.severity * 100)}% severity` : ""} {relTime(summary.timestamp)}.
      </Banner>

      <Panel flush>
        <Stats items={[
          { label: "Feature drift", value: summary.data_drift_detected ? "Alert" : "Normal", tone: summary.data_drift_detected ? "bad" : "ok", note: `${dd.drifted_features_count} of ${dd.total_features_count} features shifted (limit ${pct(thresholds.max_drifted_feature_ratio, 0)})` },
          { label: "Prediction shift", value: summary.prediction_drift_detected ? "Alert" : "Normal", tone: summary.prediction_drift_detected ? "bad" : "ok", note: `Distance ${fixed(pd.wasserstein_distance, 4)} (limit ${thresholds.wasserstein_prob_drift_limit})` },
          { label: "Performance", value: perf.ground_truth_available ? (perf.performance_degraded ? "Degraded" : "Normal") : "No labels", tone: perf.ground_truth_available ? (perf.performance_degraded ? "bad" : "ok") : undefined, note: perf.ground_truth_available ? `Recall ${pct(m.recall)}, F1 ${fixed(m.f1_score, 2)}` : "Ground truth not in this batch" },
        ]} />
      </Panel>

      <div className="grid cols-32">
        <Panel title="Feature drift" sub="Kolmogorov-Smirnov test against the training data" flush>
          <div className="table-wrap" style={{ maxHeight: 460, overflow: "auto" }}>
            <table className="table">
              <thead><tr><th>Feature</th><th style={{ width: "34%" }}>KS statistic</th><th className="num">p-value</th><th>Status</th></tr></thead>
              <tbody>
                {features.map((f) => (
                  <tr key={f.name}>
                    <td>{f.name}</td>
                    <td><div className="row" style={{ flexWrap: "nowrap" }}><div style={{ flex: 1 }}><Bullet value={f.ks_statistic} max={maxKs} tone={f.is_drifted ? "bad" : "ok"} label={`KS ${f.ks_statistic}`} /></div><span style={{ minWidth: 46, textAlign: "right" }}>{fixed(f.ks_statistic, 3)}</span></div></td>
                    <td className="num">{f.p_value < 0.0001 ? "<0.0001" : fixed(f.p_value, 4)}</td>
                    <td><Badge tone={f.is_drifted ? "bad" : "ok"}>{f.is_drifted ? "Drifted" : "Stable"}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <div className="stack">
          <Panel title="Prediction shift" sub="Average failure probability">
            <div className="stack">
              <HBars labelWidth={90} max={Math.max(pd.ref_mean_probability, pd.curr_mean_probability, 0.05) * 1.3} format={(v) => pct(v)}
                items={[{ label: "Training", value: pd.ref_mean_probability, color: "var(--s-lr)" }, { label: "This batch", value: pd.curr_mean_probability, color: pd.prediction_drift_detected ? "var(--bad)" : "var(--accent)" }]} />
              <div>
                <div className="row between small"><span className="muted">Wasserstein distance</span><span>{fixed(pd.wasserstein_distance, 4)} / {thresholds.wasserstein_prob_drift_limit}</span></div>
                <Bullet value={pd.wasserstein_distance} threshold={thresholds.wasserstein_prob_drift_limit} max={Math.max(thresholds.wasserstein_prob_drift_limit * 2, pd.wasserstein_distance * 1.1)} tone={pd.prediction_drift_detected ? "bad" : "ok"} label="Wasserstein distance versus limit" />
              </div>
            </div>
          </Panel>
          <Panel title="Model performance" sub={perf.ground_truth_available ? "Against ground-truth labels" : undefined}>
            {perf.ground_truth_available ? (
              <div className="stack">
                {[["Recall", m.recall, thresholds.min_acceptable_recall], ["F1 score", m.f1_score, thresholds.min_acceptable_f1]].map(([k, v, th]) => (
                  <div key={k}><div className="row between small"><span className="muted">{k}</span><span>{fixed(v, 3)} (floor {th})</span></div><Bullet value={v} threshold={th} tone={v < th ? "bad" : "ok"} label={`${k} versus floor`} /></div>
                ))}
                <ConfusionMatrix cm={m.confusion_matrix} />
              </div>
            ) : <Empty title="No labels in this batch">Drift can be measured without labels, but accuracy needs a 'Machine failure' column once outcomes are known.</Empty>}
          </Panel>
        </div>
      </div>
    </>
  );
}

export default function Monitoring() {
  const state = useApi("/api/monitoring/status");
  const overview = useOverview();
  const [override, setOverride] = useState(null);
  const s = override || state.data;
  const view = { ...state, data: s };

  return (
    <>
      <PageHead title="Monitoring">Detect when incoming sensor data or model behaviour moves away from what the model was trained on.</PageHead>
      <Async state={view}>
        {(status) => {
          const hist = status.history;
          const th = status.thresholds;
          const labels = hist.map((_, i) => `#${i + 1}`);
          return (
            <>
              <Controls status={status} onResult={(r) => { setOverride(r); overview?.reload(); }} />
              {status.available ? <Result s={status} /> : <Panel><Empty title="No monitoring report yet">Run a check above. Start with the validation set (expect no drift), then try a simulated drift.</Empty></Panel>}
              {hist.length > 1 && (
                <div className="grid cols-2">
                  <Panel title="Prediction shift over time" sub="Distance per check, oldest first">
                    <LineChart height={200} labels={labels} series={[{ name: "Wasserstein", color: "var(--accent)", values: hist.map((h) => h.wasserstein_distance) }]} refLines={[{ y: th.wasserstein_prob_drift_limit, label: "limit" }]} format={(v) => v.toFixed(2)} />
                  </Panel>
                  <Panel title="Recall over time" sub="Checks with labels only">
                    <LineChart height={200} labels={labels} domain={[0, 1]} series={[{ name: "Recall", color: "var(--ok)", values: hist.map((h) => h.current_recall) }]} refLines={[{ y: th.min_acceptable_recall, label: "floor" }]} format={(v) => v.toFixed(2)} />
                  </Panel>
                </div>
              )}
              <Panel title="Alert thresholds" sub="Edit in config/config.yaml">
                <KV items={[
                  ["Significance level (KS test)", th.p_value_alpha],
                  ["Drifted feature limit", `${pct(th.max_drifted_feature_ratio, 0)} of features`],
                  ["Prediction shift limit", `Wasserstein > ${th.wasserstein_prob_drift_limit}`],
                  ["Minimum recall", pct(th.min_acceptable_recall, 0)],
                  ["Minimum F1", th.min_acceptable_f1],
                ]} />
              </Panel>
            </>
          );
        }}
      </Async>
    </>
  );
}
