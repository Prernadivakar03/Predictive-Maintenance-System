import { useOverview } from "../App.jsx";
import { Async, Badge, Button, Empty, KV, Panel, PageHead, StatusBadge, Stats } from "../components/ui.jsx";
import { ConfusionMatrix } from "../components/Charts.jsx";
import Icon from "../components/Icon.jsx";
import { useApi } from "../lib/hooks.js";
import { dash, dateTime, fixed, num, pct, relTime, shortId, timeOnly } from "../lib/format.js";
import { href } from "../lib/router.js";

function Lamp({ to, label, value, note, tone, running }) {
  return (
    <a className={`lamp ${tone} ${running ? "run" : ""}`} href={href(to)}>
      <span className="lamp-k">{label}</span>
      <span className="lamp-v">{value}</span>
      <span className="lamp-n" title={note}>{note}</span>
    </a>
  );
}

function lamps(d) {
  const mon = d.monitoring;
  const drift = mon && (mon.data_drift_detected || mon.prediction_drift_detected || mon.performance_degraded);
  const lastJob = d.pipeline.last_job;
  const running = d.pipeline.running;
  const decision = d.retraining.decision;
  return [
    d.data
      ? { to: "/data", label: "Data", value: d.data.quality_valid ? "Valid" : "Issues found", tone: d.data.quality_valid ? "ok" : "bad", note: `${num(d.data.rows, 0)} rows, ${num(d.data.failure_rate_pct)}% failures` }
      : { to: "/data", label: "Data", value: "No report", tone: "warn", note: "Run the pipeline to validate data" },
    d.service.model_loaded
      ? { to: "/model", label: "Model", value: `v${d.model?.version} serving`, tone: "ok", note: d.model?.algorithm || "Production model" }
      : { to: "/model", label: "Model", value: "Not loaded", tone: "bad", note: d.service.error || "Train a model first" },
    !mon
      ? { to: "/monitoring", label: "Drift", value: "Not checked", tone: "warn", note: "Run monitoring to compare against training data" }
      : { to: "/monitoring", label: "Drift", value: drift ? (mon.performance_degraded ? "Degraded" : "Drift detected") : "No drift", tone: drift ? "bad" : "ok", note: `${mon.scenario?.label || "Validation set"}, ${relTime(mon.timestamp)}` },
    running
      ? { to: "/pipeline", label: "Pipeline", value: "Running", tone: "", running: true, note: running.stage }
      : lastJob
        ? { to: "/pipeline", label: "Pipeline", value: lastJob.status === "failed" ? "Last run failed" : "Idle", tone: lastJob.status === "failed" ? "bad" : "ok", note: `Last run ${relTime(lastJob.finished_at || lastJob.started_at)}` }
        : { to: "/pipeline", label: "Pipeline", value: "Idle", tone: "ok", note: "Weekly schedule, Sundays 00:00" },
    decision?.retrain_required
      ? { to: "/retraining", label: "Retraining", value: "Recommended", tone: "warn", note: decision.trigger_reason }
      : { to: "/retraining", label: "Retraining", value: "Not needed", tone: "ok", note: d.retraining.count ? `${d.retraining.count} past run(s)` : "No retraining yet" },
  ];
}

function HourlyBars({ hourly }) {
  const max = Math.max(...hourly.map((h) => h.total), 1);
  return (
    <div aria-label="Predictions per hour, last 24 hours" role="img" style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 64 }}>
      {hourly.map((h) => (
        <div key={h.hour} title={`${dateTime(h.hour)}: ${h.total} predictions, ${h.failures} warnings`} style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "flex-end", height: "100%" }}>
          <div style={{ height: `${(h.failures / max) * 100}%`, background: "var(--bad)", minHeight: h.failures ? 2 : 0 }} />
          <div style={{ height: `${((h.total - h.failures) / max) * 100}%`, background: "var(--accent)", opacity: 0.55, minHeight: h.total - h.failures ? 2 : 0 }} />
          {h.total === 0 && <div style={{ height: 2, background: "var(--line)" }} />}
        </div>
      ))}
    </div>
  );
}

function Activity({ d }) {
  const items = [];
  d.jobs.forEach((j) => items.push({ at: j.started_at, icon: j.kind === "pipeline" ? "pipeline" : "retraining", title: `${j.kind === "pipeline" ? "Pipeline run" : "Retraining"}`, text: j.error || (j.status === "running" ? j.stage : j.status === "succeeded" ? "Completed" : j.status), badge: <StatusBadge status={j.status} /> }));
  const last = d.retraining.last;
  if (last) items.push({ at: last.timestamp, icon: "retraining", title: `Model v${last.old_model_version} to v${last.new_model_version ?? dash}`, text: last.trigger_reason, badge: <StatusBadge status={last.promotion_status} /> });
  if (d.monitoring) items.push({ at: d.monitoring.timestamp, icon: "monitoring", title: "Monitoring check", text: d.monitoring.scenario?.label || "Validation set", badge: <Badge tone={d.monitoring.retraining_required ? "warn" : "ok"}>{d.monitoring.retraining_required ? "Retrain advised" : "Healthy"}</Badge> });
  items.sort((a, b) => new Date(b.at) - new Date(a.at));
  if (!items.length) return <Empty title="No activity yet">Run the pipeline or a monitoring check and it will show up here.</Empty>;
  return (
    <div className="feed">
      {items.slice(0, 7).map((i, idx) => (
        <div className="feed-item" key={idx}>
          <Icon name={i.icon} />
          <div style={{ minWidth: 0 }}>
            <div><strong>{i.title}</strong> <span className="faint small">{relTime(i.at)}</span></div>
            <div className="muted small" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{i.text}</div>
          </div>
          {i.badge}
        </div>
      ))}
    </div>
  );
}

export default function Overview() {
  const overview = useOverview();
  const recent = useApi("/api/predictions/recent?limit=6", { interval: 10000 });

  return (
    <>
      <PageHead title="Overview" actions={<Button icon="refresh" size="sm" onClick={overview.reload}>Refresh</Button>}>
        Is the production model healthy, and is anything waiting for attention?
      </PageHead>
      <Async state={overview}>
        {(d) => {
          const t = d.model?.test_metrics;
          const thresholds = d.thresholds || {};
          return (
            <>
              <div className="annunciator">{lamps(d).map((l) => <Lamp key={l.label} {...l} />)}</div>

              {!d.service.model_loaded && (
                <div className="banner bad" role="alert"><Icon name="alert" /><div><strong>No model is being served.</strong> <span className="muted">{d.service.error} Run the pipeline from the Pipeline page.</span></div></div>
              )}

              <div className="grid cols-21">
                <Panel title="Production model" sub={d.model ? `${d.model.algorithm}, version ${d.model.version}` : undefined}
                  right={d.model && <Badge tone="ok">@Production</Badge>} flush>
                  {t ? (
                    <>
                      <Stats items={[
                        { label: "Recall", value: pct(t.recall), note: `Floor ${pct(thresholds.min_acceptable_recall, 0)}`, tone: t.recall >= (thresholds.min_acceptable_recall ?? 0) ? "ok" : "bad" },
                        { label: "Precision", value: pct(t.precision) },
                        { label: "F1 score", value: fixed(t.f1_score), note: `Floor ${thresholds.min_acceptable_f1 ?? dash}` },
                        { label: "ROC-AUC", value: fixed(t.roc_auc) },
                        { label: "PR-AUC", value: fixed(t.pr_auc) },
                      ]} />
                      <div className="panel-b" style={{ borderTop: "1px solid var(--line-soft)" }}>
                        <KV items={[
                          ["Registered model", d.model.name],
                          ["MLflow run", <span className="mono" key="r">{shortId(d.model.run_id)}</span>],
                          ["Trained", `${dateTime(d.model.trained_at)} (${relTime(d.model.trained_at)})`],
                          ["Metrics are from", "the held-out test set"],
                        ]} />
                      </div>
                    </>
                  ) : <Empty title="No evaluation metrics">Run the pipeline to train and evaluate the model.</Empty>}
                </Panel>
                <Panel title="Test set outcomes" sub="What the model did on unseen data"><ConfusionMatrix cm={t?.confusion_matrix} /></Panel>
              </div>

              <div className="grid cols-2">
                <Panel title="Monitoring" right={<a className="btn sm" href={href("/monitoring")}>Open monitoring</a>} flush>
                  {d.monitoring ? (
                    <div className="stats">
                      {[
                        ["Feature drift", d.monitoring.data_drift_detected, `${d.monitoring.drifted_features_count} features shifted`],
                        ["Prediction shift", d.monitoring.prediction_drift_detected, `Wasserstein ${fixed(d.monitoring.wasserstein_distance, 4)}`],
                        ["Performance", d.monitoring.performance_degraded, d.monitoring.current_recall != null ? `Recall ${pct(d.monitoring.current_recall)}` : "No labels"],
                      ].map(([k, bad, n]) => (
                        <div className={`stat ${bad ? "bad" : "ok"}`} key={k}><div className="k">{k}</div><div className="v" style={{ fontSize: 22 }}>{bad ? "Alert" : "Normal"}</div><div className="n">{n}</div></div>
                      ))}
                    </div>
                  ) : <Empty title="Monitoring has not run" action={<a className="btn primary" href={href("/monitoring")}>Run a check</a>}>Compare recent sensor data with the training baseline.</Empty>}
                </Panel>
                <Panel title="Recent activity" flush><Activity d={d} /></Panel>
              </div>

              <Panel title="Live predictions" sub="Requests served by /predict"
                right={<a className="btn sm" href={href("/predict")}>Make a prediction</a>} flush>
                {d.predictions.total ? (
                  <div className="grid cols-12" style={{ gap: 0 }}>
                    <div style={{ borderRight: "1px solid var(--line-soft)" }}>
                      <Stats items={[
                        { label: "Total served", value: num(d.predictions.total, 0) },
                        { label: "Failure warnings", value: num(d.predictions.failures, 0), note: `${num(d.predictions.failure_rate_pct)}% of requests`, tone: d.predictions.failures ? "warn" : undefined },
                      ]} />
                      <div className="panel-b" style={{ borderTop: "1px solid var(--line-soft)" }}>
                        <div className="small muted" style={{ marginBottom: 8 }}>Last 24 hours</div>
                        <HourlyBars hourly={d.predictions.hourly} />
                      </div>
                    </div>
                    <div className="table-wrap">
                      <table className="table">
                        <thead><tr><th>Time</th><th>Type</th><th className="num">Torque</th><th className="num">Wear</th><th className="num">Probability</th><th>Result</th></tr></thead>
                        <tbody>
                          {(recent.data || []).map((r, i) => (
                            <tr key={i} className={r.prediction ? "row-bad" : ""}>
                              <td>{timeOnly(r.timestamp)}</td><td>{r.Type}</td><td className="num">{num(r["Torque [Nm]"], 1)}</td><td className="num">{num(r["Tool wear [min]"], 0)}</td>
                              <td className="num">{pct(r.failure_probability)}</td><td><Badge tone={r.prediction ? "bad" : "ok"}>{r.prediction ? "Failure warning" : "Normal"}</Badge></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : <Empty title="No predictions yet" action={<a className="btn primary" href={href("/predict")}>Open the predict console</a>}>Every prediction served by the API is logged here and can be used for live drift monitoring.</Empty>}
              </Panel>
            </>
          );
        }}
      </Async>
    </>
  );
}
