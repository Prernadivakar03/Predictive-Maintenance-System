import { useMemo, useRef, useState } from "react";
import { Badge, Banner, Button, Empty, KV, Panel, PageHead, Segmented } from "../components/ui.jsx";
import { Gauge } from "../components/Charts.jsx";
import Icon from "../components/Icon.jsx";
import { useToast } from "../components/Toast.jsx";
import { api } from "../lib/api.js";
import { download, parseCsv, toCsv } from "../lib/csv.js";
import { PRESETS, SENSORS, derive, failureChecks, probabilityBand, toPayload } from "../lib/domain.js";
import { num, pct, timeOnly } from "../lib/format.js";
import { useApi } from "../lib/hooks.js";
import { href } from "../lib/router.js";

const clampRange = (sensor, stats) => {
  const s = stats?.find((x) => x.column === sensor.api);
  if (!s) return sensor.fallback;
  const pad = (s.max - s.min) * 0.06;
  const lo = Math.floor((s.min - pad) * 10) / 10, hi = Math.ceil((s.max + pad) * 10) / 10;
  return [Math.max(0, lo), hi];
};

const norm = (k) => String(k).toLowerCase().replace(/\[.*?\]/g, "").replace(/[^a-z]/g, "");
const ALIASES = { type: "Type", airtemperature: "Air temperature [K]", processtemperature: "Process temperature [K]", rotationalspeed: "Rotational speed [rpm]", torque: "Torque [Nm]", toolwear: "Tool wear [min]" };
function rowToPayload(row, index) {
  const out = {};
  Object.entries(row).forEach(([k, v]) => { const key = ALIASES[norm(k)]; if (key) out[key] = v; });
  const missing = Object.values(ALIASES).filter((k) => out[k] === undefined || out[k] === "");
  if (missing.length) throw new Error(`Row ${index + 2}: missing ${missing.join(", ")}.`);
  const payload = { Type: String(out.Type).trim().toUpperCase() };
  SENSORS.forEach((s) => {
    const n = Number(out[s.api]);
    if (Number.isNaN(n)) throw new Error(`Row ${index + 2}: '${s.api}' is not a number.`);
    payload[s.api] = n;
  });
  return payload;
}

function BatchScoring({ onScored }) {
  const toast = useToast();
  const fileRef = useRef(null);
  const [over, setOver] = useState(false);
  const [rows, setRows] = useState(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const load = async (file) => {
    setError(null); setResults(null);
    try {
      const parsed = parseCsv(await file.text());
      const payloads = parsed.map(rowToPayload);
      if (payloads.length > 5000) throw new Error("Batch limit is 5,000 rows per upload.");
      setRows(payloads); setName(file.name);
    } catch (e) { setRows(null); setError(e.message); }
  };

  const score = async () => {
    setBusy(true); setError(null);
    try {
      const out = [];
      for (let i = 0; i < rows.length; i += 500) {
        const res = await api.post("/predict", { inputs: rows.slice(i, i + 500) });
        res.predictions.forEach((p, j) => out.push({ ...rows[i + j], prediction: p.prediction, status: p.status, failure_probability: p.failure_probability }));
      }
      setResults(out);
      onScored?.();
      toast(`Scored ${out.length} rows. ${out.filter((r) => r.prediction).length} failure warnings.`, "ok");
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  const columns = ["Type", ...SENSORS.map((s) => s.api), "prediction", "status", "failure_probability"];
  const template = toCsv([{ Type: "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0 }], columns.slice(0, 6));

  return (
    <Panel title="Batch scoring" sub="Score many machines from a CSV file"
      right={<Button size="sm" icon="download" onClick={() => download("sensor_template.csv", template)}>Template</Button>}>
      <div className="stack">
        <div className={`dropzone ${over ? "over" : ""}`} role="button" tabIndex={0}
          onClick={() => fileRef.current?.click()} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); e.dataTransfer.files[0] && load(e.dataTransfer.files[0]); }}>
          <Icon name="upload" size={22} />
          <div><strong>{name || "Drop a CSV file here or click to browse"}</strong></div>
          <div className="small">{rows ? `${rows.length.toLocaleString()} rows ready` : "Columns: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]"}</div>
          <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => e.target.files[0] && load(e.target.files[0])} />
        </div>
        {error && <Banner tone="bad">{error}</Banner>}
        <div className="row">
          <Button variant="primary" icon="play" busy={busy} disabled={!rows} onClick={score}>Score {rows ? rows.length.toLocaleString() : ""} rows</Button>
          {results && <Button icon="download" onClick={() => download("predictions.csv", toCsv(results, columns))}>Download results</Button>}
          {results && <Badge tone={results.some((r) => r.prediction) ? "warn" : "ok"}>{results.filter((r) => r.prediction).length} of {results.length} flagged</Badge>}
        </div>
        {results && (
          <div className="table-wrap" style={{ maxHeight: 320, overflow: "auto", border: "1px solid var(--line-soft)", borderRadius: 4 }}>
            <table className="table">
              <thead><tr><th>#</th><th>Type</th>{SENSORS.map((s) => <th key={s.key} className="num">{s.label}</th>)}<th className="num">Probability</th><th>Result</th></tr></thead>
              <tbody>
                {[...results].map((r, i) => ({ r, i })).sort((a, b) => b.r.failure_probability - a.r.failure_probability).slice(0, 200).map(({ r, i }) => (
                  <tr key={i} className={r.prediction ? "row-bad" : ""}>
                    <td>{i + 1}</td><td>{r.Type}</td>{SENSORS.map((s) => <td key={s.key} className="num">{num(r[s.api], 1)}</td>)}
                    <td className="num">{pct(r.failure_probability)}</td><td><Badge tone={r.prediction ? "bad" : "ok"}>{r.prediction ? "Failure warning" : "Normal"}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {results && results.length > 200 && <div className="small faint">Showing the 200 highest-risk rows. Download the CSV for all {results.length.toLocaleString()}.</div>}
      </div>
    </Panel>
  );
}

export default function Predict() {
  const toast = useToast();
  const data = useApi("/api/data/summary");
  const recent = useApi("/api/predictions/recent?limit=10", { interval: 15000 });
  const [v, setV] = useState({ type: "M", air: 298.1, process: 308.6, rpm: 1551, torque: 42.8, wear: 0 });
  const [preset, setPreset] = useState("normal");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const ranges = useMemo(() => Object.fromEntries(SENSORS.map((s) => [s.key, clampRange(s, data.data?.sensors)])), [data.data]);
  const set = (key, value) => { setV((p) => ({ ...p, [key]: value })); setPreset(null); };
  const checks = failureChecks(v);
  const d = derive(v);

  const submit = async (e) => {
    e?.preventDefault();
    setBusy(true); setError(null);
    const started = performance.now();
    try {
      const res = await api.post("/predict", { inputs: [toPayload(v)] });
      setResult({ ...res.predictions[0], latency: Math.round(performance.now() - started) });
      recent.reload();
    } catch (err) { setError(err.message); toast(err.message, "bad"); } finally { setBusy(false); }
  };

  const band = result ? probabilityBand(result.failure_probability) : null;

  return (
    <>
      <PageHead title="Predict">Enter live sensor readings from a machine and get a failure prediction from the production model.</PageHead>
      <div className="grid cols-32">
        <Panel title="Sensor readings" right={<Segmented label="Product quality type" value={v.type} onChange={(t) => set("type", t)} options={[{ value: "L", label: "Low" }, { value: "M", label: "Medium" }, { value: "H", label: "High" }]} />}>
          <form className="stack" onSubmit={submit} noValidate>
            <div className="presets" role="group" aria-label="Example scenarios">
              {PRESETS.map((p) => (
                <button type="button" key={p.id} className="btn sm" aria-pressed={preset === p.id} style={preset === p.id ? { background: "var(--accent-bg)", borderColor: "var(--accent)" } : undefined}
                  onClick={() => { setV({ type: p.type, air: p.air, process: p.process, rpm: p.rpm, torque: p.torque, wear: p.wear }); setPreset(p.id); }}>{p.label}</button>
              ))}
            </div>
            {SENSORS.map((s) => {
              const [lo, hi] = ranges[s.key];
              return (
                <div className="slider-row" key={s.key}>
                  <div className="top">
                    <label htmlFor={`in-${s.key}`} className="label">{s.label}</label>
                    <div className="input-unit">
                      <input id={`in-${s.key}`} className="input" type="number" step="any" inputMode="decimal" value={v[s.key]} onChange={(e) => set(s.key, e.target.value)} />
                      <span>{s.unit}</span>
                    </div>
                  </div>
                  <input type="range" aria-label={`${s.label} slider`} min={lo} max={hi} step={s.step} value={Number(v[s.key]) || lo} onChange={(e) => set(s.key, Number(e.target.value))} />
                  <div className="scale"><span>{lo}</span><span>{hi} {s.unit}</span></div>
                </div>
              );
            })}
            <div className="row">
              <Button variant="primary" icon="bolt" busy={busy} type="submit">Predict failure risk</Button>
              <span className="small faint">Slider ranges follow the training data.</span>
            </div>
            {error && <Banner tone="bad">{error}</Banner>}
          </form>
        </Panel>

        <div className="stack">
          <Panel title="Failure probability">
            <div className={`verdict ${band || ""}`}>
              <Gauge value={result?.failure_probability} idle={!result} label="Failure probability dial" />
              {result ? (
                <>
                  <div className="big">{pct(result.failure_probability)}</div>
                  <Badge tone={band}>{result.prediction ? "Failure warning: schedule an inspection" : "Normal operation"}</Badge>
                  <div className="small muted">Model v{result.model_version} · answered in {result.latency} ms · {timeOnly(result.timestamp)}</div>
                </>
              ) : <div className="muted" style={{ padding: "6px 0 2px" }}>Run a prediction to move the needle.</div>}
            </div>
          </Panel>
          <Panel title="Physical failure checks" sub="From the AI4I 2020 failure-mode rules">
            <div className="checks">
              {checks.map((c) => (
                <div key={c.id} className={`check-row ${c.level}`}>
                  <span><strong>{c.label}</strong> <span className="faint small">{c.id}</span></span>
                  <Badge tone={c.level}>{c.level === "bad" ? "Rule met" : c.level === "warn" ? "Close" : "Clear"}</Badge>
                  <span className="small muted" style={{ gridColumn: "1 / -1" }}>{c.detail}</span>
                  <span className="meter"><i style={{ width: `${Math.round(c.value * 100)}%` }} /></span>
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="Derived features">
            <KV items={[["Temperature gap", `${d.tempDiff.toFixed(1)} K`], ["Mechanical power", `${d.powerKw.toFixed(2)} kW`], ["Wear \u00d7 torque", Math.round(d.wearTorque).toLocaleString()]]} />
          </Panel>
        </div>
      </div>

      <BatchScoring onScored={recent.reload} />

      <Panel title="Latest predictions" sub="Logged by the API" right={<a className="btn sm" href={href("/monitoring")}>Use for drift monitoring</a>} flush>
        {recent.data?.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Time</th><th>Type</th>{SENSORS.map((s) => <th key={s.key} className="num">{s.label}</th>)}<th className="num">Probability</th><th>Result</th></tr></thead>
              <tbody>
                {recent.data.map((r, i) => (
                  <tr key={i} className={r.prediction ? "row-bad" : ""}>
                    <td>{timeOnly(r.timestamp)}</td><td>{r.Type}</td>{SENSORS.map((s) => <td key={s.key} className="num">{num(r[s.api], 1)}</td>)}
                    <td className="num">{pct(r.failure_probability)}</td><td><Badge tone={r.prediction ? "bad" : "ok"}>{r.prediction ? "Failure warning" : "Normal"}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty title="Nothing logged yet">Predictions you make here appear in this table.</Empty>}
      </Panel>
    </>
  );
}
