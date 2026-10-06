import { useEffect, useRef } from "react";
import { useOverview } from "../App.jsx";
import { useJob } from "../lib/hooks.js";
import { dash, duration, pct, timeOnly } from "../lib/format.js";
import { Banner, KV, Panel, StatusBadge } from "./ui.jsx";
import { useToast } from "./Toast.jsx";

export function LogView({ lines, follow }) {
  const ref = useRef(null);
  useEffect(() => { if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines, follow]);
  return (
    <div className="log" ref={ref} role="log" aria-label="Job log">
      {lines.length ? lines.map((l, i) => (
        <div key={i} className={l.level}><span className="t">{timeOnly(l.ts)}</span><span className="lv">{l.message}</span></div>
      )) : <div className="t">Waiting for output…</div>}
    </div>
  );
}

function ResultSummary({ job }) {
  const r = job.result;
  if (!r) return null;
  if (job.kind === "pipeline" && r.artifacts) {
    const a = r.artifacts;
    return <KV items={[["Selected model", a.best_model_name], ["Registered version", a.registered_model_version && `v${a.registered_model_version} @Production`], ["Test recall", pct(a.test_recall)], ["Test F1", a.test_f1?.toFixed(3)], ["Stages", (r.stages_executed || []).length]]} />;
  }
  if (job.kind === "retraining") {
    if (r.status === "SKIPPED") return <Banner icon="info">Retraining skipped. {r.reason}</Banner>;
    return (
      <div className="stack">
        <Banner tone={r.promotion_status === "PROMOTED" ? "ok" : "bad"} title={r.promotion_status === "PROMOTED" ? `Version ${r.new_model_version} promoted to production` : "Candidate rejected, previous model restored"}>{r.promotion_reason}</Banner>
        <KV items={[["Trigger", r.trigger_reason], ["Algorithm", r.selected_algorithm || dash], ["Version", `v${r.old_model_version} to ${r.new_model_version ? `v${r.new_model_version}` : dash}`]]} />
      </div>
    );
  }
  return null;
}

/** Live view of a background job: status, current stage, result summary and streamed log. */
export default function JobPanel({ jobId, title = "Job" }) {
  const toast = useToast();
  const overview = useOverview();
  const { job } = useJob(jobId, (j) => {
    toast(j.status === "succeeded" ? `${title} finished.` : `${title} failed: ${j.error}`, j.status === "succeeded" ? "ok" : "bad");
    overview?.reload();
  });
  if (!job) return null;
  const running = job.status === "running";
  return (
    <Panel title={`${title} ${job.id}`} right={<StatusBadge status={job.status} />} flush>
      <div className="panel-b stack">
        <div className="row between">
          <span className="muted">{running ? `In progress: ${job.stage}` : job.stage}</span>
          <span className="faint small">{job.duration_s != null ? `Took ${duration(job.duration_s)}` : "Running…"}</span>
        </div>
        {job.error && <Banner tone="bad" title="The job failed">{job.error}</Banner>}
        {!running && !job.error && <ResultSummary job={job} />}
      </div>
      <LogView lines={job.log || []} follow={running} />
    </Panel>
  );
}
