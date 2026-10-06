import { useEffect, useState } from "react";
import JobPanel from "../components/JobPanel.jsx";
import { Async, Badge, Banner, Button, ConfirmModal, Empty, KV, PageHead, Panel, StatusBadge } from "../components/ui.jsx";
import { useToast } from "../components/Toast.jsx";
import { api } from "../lib/api.js";
import { dash, dateTime, fixed, pct, relTime } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";
import { href } from "../lib/router.js";

function Rules({ s }) {
  const m = s.monitoring_summary, th = s.thresholds;
  const row = (label, limit, current, fired) => ({ label, limit, current, fired });
  const rows = m ? [
    row("Feature drift", `More than ${pct(th.max_drifted_feature_ratio, 0)} of features`, `${m.drifted_features_count} features shifted`, m.data_drift_detected),
    row("Prediction shift", `Wasserstein above ${th.wasserstein_prob_drift_limit}`, fixed(m.wasserstein_distance, 4), m.prediction_drift_detected),
    row("Performance", `Recall below ${pct(th.min_acceptable_recall, 0)} or F1 below ${th.min_acceptable_f1}`, m.current_recall != null ? `Recall ${pct(m.current_recall)}` : "No labels in last batch", m.performance_degraded),
  ] : [];
  if (!m) return <Empty title="No monitoring data" action={<a className="btn primary" href={href("/monitoring")}>Run a monitoring check</a>}>Automated retraining decides from the latest monitoring report.</Empty>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Trigger</th><th>Fires when</th><th>Latest value</th><th>State</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.label}><td><strong>{r.label}</strong></td><td className="muted">{r.limit}</td><td>{r.current}</td><td><Badge tone={r.fired ? "warn" : "ok"}>{r.fired ? "Fired" : "Clear"}</Badge></td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function Metric({ label, now, before }) {
  const delta = before != null ? now - before : null;
  return (
    <div><div className="faint small">{label}</div>
      <div><strong>{pct(now)}</strong>{delta !== null && <span className="small" style={{ marginLeft: 6, color: delta < -0.0005 ? "var(--bad)" : delta > 0.0005 ? "var(--ok)" : "var(--ink-3)" }}>{Math.abs(delta) < 0.0005 ? "no change" : `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts`}</span>}</div>
    </div>
  );
}

function History({ items }) {
  if (!items.length) return <Empty title="No retraining yet">When a trigger fires (or you retrain manually) each attempt is recorded here with the promotion decision.</Empty>;
  return (
    <div className="timeline">
      {items.map((h, i) => {
        const tone = h.promotion_status === "PROMOTED" ? "ok" : h.promotion_status === "SKIPPED" ? "" : "bad";
        const t = h.test_metrics, p = h.previous_test_metrics;
        return (
          <div className="tl-item" key={i}>
            <span className={`tl-dot ${tone}`} />
            <div className="stack" style={{ gap: 8 }}>
              <div className="row">
                <strong>{h.new_model_version ? `Version ${h.old_model_version} to ${h.new_model_version}` : `Version ${h.old_model_version} kept`}</strong>
                <StatusBadge status={h.promotion_status} />
                <span className="faint small" title={dateTime(h.timestamp)}>{relTime(h.timestamp)}</span>
              </div>
              <div className="muted small">Trigger: {h.trigger_reason}{h.selected_algorithm ? ` · ${h.selected_algorithm}` : ""}</div>
              {t && <div className="row" style={{ gap: 28 }}><Metric label="Recall" now={t.recall} before={p?.recall} /><Metric label="Precision" now={t.precision} before={p?.precision} /><Metric label="F1 score" now={t.f1_score} before={p?.f1_score} /></div>}
              {h.gate_checks && (
                <div className="small" style={{ display: "grid", gap: 3 }}>
                  {h.gate_checks.map((c) => <div key={c.name}><Badge tone={c.passed ? "ok" : "bad"} plain>{c.passed ? "Pass" : "Fail"}</Badge> <strong>{c.name}</strong> <span className="muted">{c.detail}</span></div>)}
                </div>
              )}
              {!h.gate_checks && <div className="muted small">{h.promotion_reason}</div>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Retraining() {
  const toast = useToast();
  const [polling, setPolling] = useState(false);
  const state = useApi("/api/retraining/status", { interval: polling ? 3000 : 20000 });
  const [jobId, setJobId] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(null);

  const running = state.data?.running;
  useEffect(() => setPolling(Boolean(running)), [running]);
  useEffect(() => { if (running && !jobId) setJobId(running.id); }, [running, jobId]);

  const start = async (mode) => {
    setBusy(mode);
    try {
      const job = await api.post("/api/retraining/trigger", { mode });
      setJobId(job.id); setConfirm(false); state.reload();
      toast(mode === "force" ? "Retraining started." : "Automated check started.", "ok");
    } catch (e) { toast(e.message, "bad"); setConfirm(false); } finally { setBusy(null); }
  };

  return (
    <>
      <PageHead title="Retraining"
        actions={<><Button icon="refresh" busy={busy === "auto" || Boolean(running)} onClick={() => start("auto")}>Run automated check</Button><Button variant="danger" icon="bolt" disabled={Boolean(running)} onClick={() => setConfirm(true)}>Retrain now</Button></>}>
        A new model replaces production only if it passes the safety gate. If it does not, the previous model keeps serving.
      </PageHead>
      <Async state={state}>
        {(s) => (
          <>
            {s.decision ? (
              <Banner tone={s.decision.retrain_required ? "warn" : "ok"} title={s.decision.retrain_required ? "Retraining is recommended" : "Production model is healthy"}>
                {s.decision.trigger_reason}
              </Banner>
            ) : <Banner icon="info" title="No monitoring report yet">Run a check on the Monitoring page so the decision engine has something to evaluate.</Banner>}

            {jobId && <JobPanel jobId={jobId} title="Retraining job" />}

            <div className="grid cols-32">
              <Panel title="Automated triggers" sub="Evaluated from the latest monitoring report" flush><Rules s={s} /></Panel>
              <Panel title="Promotion safety gate">
                <div className="stack">
                  <KV items={[
                    ["Minimum recall", pct(s.criteria.min_recall_threshold, 0)],
                    ["Recall vs production", `at least ${s.criteria.min_recall_improvement_pct >= 0 ? "+" : ""}${s.criteria.min_recall_improvement_pct} pts`],
                    ["Precision drop", `at most ${s.criteria.max_precision_drop_pct} pts`],
                  ]} />
                  <p className="muted small">Candidates are compared with the current production model on the same held-out test set. A rejected candidate is rolled back in both the MLflow registry and the files the API serves. Thresholds live in config/config.yaml.</p>
                </div>
              </Panel>
            </div>

            <Panel title="Retraining history"><History items={s.history} /></Panel>
          </>
        )}
      </Async>
      {confirm && (
        <ConfirmModal danger title="Retrain the model now?" confirmLabel="Retrain" busy={busy === "force"} onConfirm={() => start("force")} onClose={() => setConfirm(false)}>
          This ignores the monitoring triggers and trains new candidate models immediately. The result is only promoted if it passes the safety gate.
        </ConfirmModal>
      )}
    </>
  );
}
