import { useState } from "react";
import Figures from "../components/Figures.jsx";
import { ConfusionMatrix, GroupedBars, HBars, Legend } from "../components/Charts.jsx";
import { Async, Badge, Empty, KV, Panel, PageHead, Stats, Tabs } from "../components/ui.jsx";
import { algoColor, dash, dateTime, fixed, METRIC_LABELS, num, pct, relTime, shortId } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";

const COMPARE = ["recall", "precision", "f1_score", "roc_auc", "pr_auc"];

export default function Model() {
  const model = useApi("/api/model");
  const versions = useApi("/api/registry/versions?limit=30");
  const data = useApi("/api/data/summary");
  const [candidate, setCandidate] = useState(null);

  return (
    <>
      <PageHead title="Model">The model serving predictions, how it compares with the alternatives, and its version history.</PageHead>
      <Async state={model}>
        {(m) => {
          const t = m.test_metrics;
          const names = Object.keys(m.candidates);
          const sel = candidate && names.includes(candidate) ? candidate : m.algorithm || names[0];
          const selMetrics = m.candidates[sel];
          return (
            <>
              <Panel title={m.algorithm || "Production model"} sub={`${m.estimator || ""} · version ${m.serving_version || dash}`} right={<Badge tone="ok">@Production</Badge>} flush>
                {t && <Stats items={[
                  { label: "Recall", value: pct(t.recall), note: m.targets.min_recall ? `Floor ${pct(m.targets.min_recall, 0)}` : undefined, tone: m.targets.min_recall && t.recall < m.targets.min_recall ? "bad" : "ok" },
                  { label: "Precision", value: pct(t.precision) },
                  { label: "F1 score", value: fixed(t.f1_score) },
                  { label: "Accuracy", value: pct(t.accuracy) },
                  { label: "ROC-AUC", value: fixed(t.roc_auc) },
                  { label: "PR-AUC", value: fixed(t.pr_auc) },
                ]} />}
                <div className="panel-b" style={{ borderTop: "1px solid var(--line-soft)" }}>
                  <KV items={[
                    ["Trained", `${dateTime(m.trained_at)} (${relTime(m.trained_at)})`],
                    ["MLflow run", <span className="mono" key="r">{shortId(m.selected_run_id)}</span>],
                    ["Selection rule", "Highest validation recall, then F1 and PR-AUC"],
                    ["Why recall first", "A missed failure costs far more than an extra inspection"],
                  ]} />
                </div>
              </Panel>

              <div className="grid cols-32">
                <Panel title="Candidate comparison" sub="Validation set">
                  <div className="stack">
                    <Legend items={names.map((n) => ({ name: n + (n === m.algorithm ? " (selected)" : ""), color: algoColor(n) }))} />
                    <GroupedBars categories={COMPARE.map((k) => METRIC_LABELS[k])}
                      series={names.map((n) => ({ name: n, color: algoColor(n), values: COMPARE.map((k) => m.candidates[n][k]) }))} />
                  </div>
                </Panel>
                <Panel title="Confusion matrix" sub="Validation set" right={<Badge tone="plain">{sel}</Badge>}>
                  <div className="stack">
                    <Tabs value={sel} onChange={setCandidate} options={names.map((n) => ({ value: n, label: n }))} />
                    <ConfusionMatrix cm={selMetrics?.confusion_matrix} />
                  </div>
                </Panel>
              </div>

              <div className="grid cols-2">
                <Panel title="What drives the prediction" sub="Feature importance">
                  {m.feature_importance.length ? (
                    <HBars labelWidth={170} items={m.feature_importance.map((f) => ({ label: f.feature, value: f.importance }))} format={(v) => pct(v)} />
                  ) : <Empty title="Not available">This model type does not expose feature importance.</Empty>}
                </Panel>
                <Panel title="Hyperparameters">
                  {Object.keys(m.params).length ? <KV items={Object.entries(m.params).map(([k, v]) => [k, String(v)])} /> : <Empty title="No parameters recorded">Run the pipeline to write model metadata.</Empty>}
                </Panel>
              </div>
            </>
          );
        }}
      </Async>

      <Panel title="Model registry" sub={versions.data ? `${versions.data.total} versions of ${versions.data.model_name}` : undefined} flush>
        {versions.error && !versions.data ? <Empty title="Registry unavailable">{versions.error.message}</Empty> : !versions.data ? <div className="panel-b"><div className="skeleton" style={{ height: 120 }} /></div> : (
          <div className="table-wrap" style={{ maxHeight: 420, overflow: "auto" }}>
            <table className="table">
              <thead><tr><th>Version</th><th>Alias</th><th>Algorithm</th><th className="num">Val recall</th><th className="num">Val F1</th><th>Created</th><th>Run</th></tr></thead>
              <tbody>
                {versions.data.versions.map((v) => (
                  <tr key={v.version} className={v.aliases.includes("Production") ? "hl" : ""}>
                    <td><strong>v{v.version}</strong></td>
                    <td>{v.aliases.length ? v.aliases.map((a) => <Badge key={a} tone="ok">{a}</Badge>) : dash}</td>
                    <td>{v.algorithm || dash}</td>
                    <td className="num">{pct(v.metrics.val_recall)}</td><td className="num">{fixed(v.metrics.val_f1_score)}</td>
                    <td title={dateTime(v.created_at)}>{relTime(v.created_at)}</td><td className="mono">{shortId(v.run_id)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Evaluation plots">
        <Figures figures={(data.data?.figures || []).filter((f) => f.name.startsWith("cm_") || f.name.startsWith("model_comparison"))} />
      </Panel>
    </>
  );
}
