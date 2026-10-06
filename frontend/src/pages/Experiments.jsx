import { useMemo, useState } from "react";
import { Async, Badge, Button, Empty, Panel, PageHead, Segmented, StatusBadge, Stats } from "../components/ui.jsx";
import { Legend, LineChart } from "../components/Charts.jsx";
import { algoColor, dash, dateTime, duration, fixed, num, pct, shortId } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";

const COLUMNS = [
  { key: "started_at", label: "Started" },
  { key: "name", label: "Run" },
  { key: "algorithm", label: "Algorithm" },
  { key: "val_recall", label: "Val recall", num: true, metric: true },
  { key: "val_precision", label: "Val precision", num: true, metric: true },
  { key: "val_f1_score", label: "Val F1", num: true, metric: true },
  { key: "val_roc_auc", label: "Val ROC-AUC", num: true, metric: true },
  { key: "test_recall", label: "Test recall", num: true, metric: true },
  { key: "duration_s", label: "Duration", num: true },
  { key: "status", label: "Status" },
];

export default function Experiments() {
  const runsState = useApi("/api/experiments/runs?limit=300");
  const system = useApi("/api/system");
  const [algo, setAlgo] = useState("all");
  const [sort, setSort] = useState({ key: "started_at", dir: "desc" });
  const [metric, setMetric] = useState("val_recall");

  const view = useMemo(() => {
    const runs = runsState.data?.runs || [];
    const filtered = algo === "all" ? runs : runs.filter((r) => r.algorithm === algo);
    const value = (r, k) => (COLUMNS.find((c) => c.key === k)?.metric ? r.metrics[k] : r[k]);
    return [...filtered].sort((a, b) => {
      const av = value(a, sort.key), bv = value(b, sort.key);
      if (av === bv) return 0;
      if (av === undefined || av === null) return 1;
      if (bv === undefined || bv === null) return -1;
      return (av > bv ? 1 : -1) * (sort.dir === "asc" ? 1 : -1);
    });
  }, [runsState.data, algo, sort]);

  const toggleSort = (key) => setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" }));

  return (
    <>
      <PageHead title="Experiments" actions={system.data && <a className="btn" href={system.data.links.mlflow} target="_blank" rel="noreferrer">Open MLflow UI</a>}>
        Every training run is tracked in MLflow with its parameters, metrics and artifacts.
      </PageHead>
      <Async state={runsState}>
        {(d) => {
          if (!d.runs.length) return <Panel><Empty title="No runs yet">Run the pipeline to log the first experiment.</Empty></Panel>;
          const algos = [...new Set(d.runs.map((r) => r.algorithm).filter(Boolean))];
          const scored = d.runs.filter((r) => r.metrics.val_recall !== undefined);
          const best = [...scored].sort((a, b) => b.metrics.val_recall - a.metrics.val_recall || b.metrics.val_f1_score - a.metrics.val_f1_score)[0];
          const chrono = (name) => d.runs.filter((r) => r.algorithm === name && r.metrics[metric] !== undefined).reverse().slice(-30);
          const length = Math.max(...algos.map((a) => chrono(a).length), 0);
          return (
            <>
              <Panel flush>
                <Stats items={[
                  { label: "Tracked runs", value: num(d.total, 0), note: d.experiment?.name },
                  { label: "Pipeline executions", value: num(Math.round(d.total / Math.max(algos.length, 1)), 0), note: `${algos.length} candidates each` },
                  { label: "Best validation recall", value: pct(best?.metrics.val_recall), note: best?.algorithm },
                  { label: "Latest run", value: dateTime(d.runs[0].started_at), note: d.runs[0].name },
                ]} />
              </Panel>

              <Panel title="Metric history" sub="Oldest to newest, latest 30 runs per algorithm"
                right={<Segmented label="Metric" value={metric} onChange={setMetric} options={[{ value: "val_recall", label: "Recall" }, { value: "val_f1_score", label: "F1" }, { value: "val_roc_auc", label: "ROC-AUC" }]} />}>
                <div className="stack">
                  <Legend items={algos.map((a) => ({ name: a, color: algoColor(a) }))} />
                  <LineChart domain={[0, 1]} labels={Array.from({ length }, (_, i) => `#${i + 1}`)}
                    series={algos.map((a) => ({ name: a, color: algoColor(a), values: chrono(a).map((r) => r.metrics[metric]) }))} />
                </div>
              </Panel>

              <Panel title="Runs" sub={`${view.length} shown`} right={<Segmented label="Algorithm" value={algo} onChange={setAlgo} options={[{ value: "all", label: "All" }, ...algos.map((a) => ({ value: a, label: a }))]} />} flush>
                <div className="table-wrap" style={{ maxHeight: 520, overflow: "auto" }}>
                  <table className="table">
                    <thead><tr>{COLUMNS.map((c) => (
                      <th key={c.key} className={`sortable ${c.num ? "num" : ""}`} onClick={() => toggleSort(c.key)} aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}>
                        {c.label}{sort.key === c.key ? (sort.dir === "asc" ? " \u25B2" : " \u25BC") : ""}
                      </th>))}<th>ID</th></tr></thead>
                    <tbody>
                      {view.slice(0, 150).map((r) => (
                        <tr key={r.run_id} className={r.stage === "Production" ? "hl" : ""}>
                          <td>{dateTime(r.started_at)}</td>
                          <td>{r.name}{r.stage === "Production" && <> <Badge tone="ok">Production</Badge></>}</td>
                          <td><span style={{ color: algoColor(r.algorithm), fontWeight: 600 }}>{r.algorithm || dash}</span></td>
                          <td className="num">{pct(r.metrics.val_recall)}</td><td className="num">{pct(r.metrics.val_precision)}</td>
                          <td className="num">{fixed(r.metrics.val_f1_score)}</td><td className="num">{fixed(r.metrics.val_roc_auc)}</td>
                          <td className="num">{r.metrics.test_recall !== undefined ? pct(r.metrics.test_recall) : dash}</td>
                          <td className="num">{duration(r.duration_s)}</td><td><StatusBadge status={r.status} /></td><td className="mono">{shortId(r.run_id)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          );
        }}
      </Async>
    </>
  );
}
