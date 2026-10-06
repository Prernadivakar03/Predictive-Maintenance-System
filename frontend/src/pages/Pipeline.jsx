import { useEffect, useState } from "react";
import JobPanel from "../components/JobPanel.jsx";
import { Async, Badge, Banner, Button, ConfirmModal, Empty, KV, PageHead, Panel, StatusBadge } from "../components/ui.jsx";
import { useToast } from "../components/Toast.jsx";
import { api } from "../lib/api.js";
import { dash, dateTime, duration, relTime } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";

function stageNumber(stage) {
  const m = /STAGES?\s+(\d)/i.exec(stage || "");
  return m ? Number(m[1]) : 0;
}

function nodeState(stage, index, running) {
  if (running) {
    const n = stageNumber(running.stage);
    const current = n >= 4 ? [3, 4] : [n - 1];
    if (current.includes(index)) return "run";
    if (index < Math.min(...current)) return "ok";
    return "missing";
  }
  return stage?.status || "missing";
}

function Dag({ dag, stages, running }) {
  const byId = Object.fromEntries(stages.map((s) => [s.id, s]));
  return (
    <div className="dag">
      {dag.tasks.map((t, i) => {
        const s = byId[t.id];
        const state = nodeState(s, i, running);
        return (
          <div key={t.id} className={`dag-node ${state}`}>
            <span className="t">{t.label}</span>
            <span className="d">{t.description}</span>
            <div>{state === "run" ? <Badge tone="info">Running</Badge> : state === "ok" ? <Badge tone="ok">Complete</Badge> : state === "failed" ? <Badge tone="bad">Failed</Badge> : <Badge>Not run</Badge>}</div>
            <span className="small muted">{s?.detail}</span>
            <span className="id">{s?.updated_at ? `Updated ${relTime(s.updated_at)}` : t.id}</span>
          </div>
        );
      })}
    </div>
  );
}

function QualityReport({ q }) {
  if (!q) return <Empty title="No validation report">The data validation stage writes this report.</Empty>;
  const checks = [
    ["Rows present", q.total_rows > 0, `${q.total_rows?.toLocaleString()} rows, ${q.total_columns} columns`],
    ["Schema complete", !(q.issues || []).some((i) => /missing expected|Target column/i.test(i)), "All expected columns found"],
    ["No missing values", Object.keys(q.missing_values || {}).length === 0, Object.keys(q.missing_values || {}).length ? `${Object.keys(q.missing_values).length} column(s) with gaps` : "Zero null cells"],
    ["Values in range", Object.keys(q.invalid_value_counts || {}).length === 0, Object.keys(q.invalid_value_counts || {}).length ? `Out of range: ${Object.keys(q.invalid_value_counts).join(", ")}` : "All sensors within physical limits"],
    ["No duplicate rows", (q.duplicate_rows || 0) === 0, `${q.duplicate_rows || 0} duplicates`],
  ];
  return (
    <div className="stack">
      <div className="row between"><span className="muted small">Checked {dateTime(q.timestamp)}</span><Badge tone={q.is_valid ? "ok" : "bad"}>{q.is_valid ? "Passed" : "Failed"}</Badge></div>
      {checks.map(([label, ok, detail]) => (
        <div className="row between" key={label}><span><Badge tone={ok ? "ok" : "bad"} plain>{ok ? "Pass" : "Fail"}</Badge> {label}</span><span className="muted small">{detail}</span></div>
      ))}
      {(q.issues || []).map((i) => <Banner key={i} tone="bad">{i}</Banner>)}
    </div>
  );
}

export default function Pipeline() {
  const toast = useToast();
  const [polling, setPolling] = useState(false);
  const status = useApi("/api/pipeline/status", { interval: polling ? 2000 : 15000 });
  const data = useApi("/api/data/summary");
  const [selected, setSelected] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const [force, setForce] = useState(false);
  const [busy, setBusy] = useState(false);

  const running = status.data?.running;
  useEffect(() => setPolling(Boolean(running)), [running]);
  useEffect(() => { if (running && !selected) setSelected(running.id); }, [running, selected]);
  const activeId = selected || status.data?.last_pipeline_job?.id || null;

  const start = async () => {
    setBusy(true);
    try {
      const job = await api.post("/api/pipeline/run", { force });
      setSelected(job.id); setConfirm(false); status.reload();
      toast("Pipeline started.", "ok");
    } catch (e) { toast(e.message, "bad"); setConfirm(false); } finally { setBusy(false); }
  };

  return (
    <>
      <PageHead title="Pipeline" actions={<Button variant="primary" icon="play" busy={Boolean(running)} onClick={() => setConfirm(true)}>{running ? "Pipeline running" : "Run pipeline"}</Button>}>
        From raw sensor data to a registered production model. Runs on demand here and on a weekly schedule in Airflow.
      </PageHead>
      <Async state={status}>
        {(s) => (
          <>
            <Panel title="Workflow" sub={s.dag.dag_id}
              right={<><Badge tone="plain">{s.dag.schedule_text}</Badge><a className="btn sm" href={s.links.airflow} target="_blank" rel="noreferrer">Open Airflow</a></>}>
              <div className="stack">
                <Dag dag={s.dag} stages={s.stages} running={running} />
                <div className="small muted">Each task retries up to {s.dag.retries} times, {s.dag.retry_delay_minutes} min apart. A validation failure halts the run before any model is trained. Airflow needs to be running separately for the weekly schedule; see the README.</div>
              </div>
            </Panel>

            <div className="grid cols-32">
              <div className="stack">
                {activeId ? <JobPanel jobId={activeId} title="Pipeline run" /> : <Panel title="Pipeline runs"><Empty title="No runs from this console yet" action={<Button variant="primary" icon="play" onClick={() => setConfirm(true)}>Run the pipeline</Button>}>Starting a run trains all three candidate models, logs them to MLflow and promotes the best one.</Empty></Panel>}
              </div>
              <Panel title="Data quality" sub="Latest validation report"><QualityReport q={data.data?.quality} /></Panel>
            </div>

            <Panel title="Job history" flush>
              {s.jobs.length ? (
                <div className="table-wrap">
                  <table className="table">
                    <thead><tr><th>Job</th><th>Type</th><th>Started</th><th className="num">Duration</th><th>Status</th><th>Detail</th></tr></thead>
                    <tbody>
                      {s.jobs.map((j) => (
                        <tr key={j.id} className={j.id === activeId ? "hl" : ""}>
                          <td><button className="btn sm ghost mono" onClick={() => setSelected(j.id)} aria-label={`View job ${j.id}`}>{j.id}</button></td>
                          <td>{j.kind === "pipeline" ? "Pipeline" : "Retraining"}</td><td>{dateTime(j.started_at)}</td>
                          <td className="num">{duration(j.duration_s)}</td><td><StatusBadge status={j.status} /></td>
                          <td className="muted" style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={j.error || j.stage}>{j.error || j.stage || dash}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <Empty title="No jobs yet">Runs started from this console are listed here and kept across restarts.</Empty>}
            </Panel>
          </>
        )}
      </Async>
      {confirm && (
        <ConfirmModal title="Run the pipeline?" confirmLabel="Start run" busy={busy} onConfirm={start} onClose={() => setConfirm(false)}>
          <div className="stack">
            <span>This ingests and validates the data, trains Logistic Regression, Random Forest and XGBoost, then registers the best model as the new production version. It usually takes a minute or two.</span>
            <label className="check"><input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /> Re-run every stage even if the data is unchanged</label>
          </div>
        </ConfirmModal>
      )}
    </>
  );
}
