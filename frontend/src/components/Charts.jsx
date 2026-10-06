import { useWidth } from "../lib/hooks.js";
import { dash } from "../lib/format.js";

const tickFmt = (v, digits = 2) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : Number(v.toFixed(digits)).toString());

/* ------------------------------------------------------------------ semicircle dial */
function polar(cx, cy, r, v) {
  const theta = Math.PI * (1 - v);
  return [cx + r * Math.cos(theta), cy - r * Math.sin(theta)];
}
function arc(cx, cy, r, v0, v1) {
  const [x0, y0] = polar(cx, cy, r, v0);
  const [x1, y1] = polar(cx, cy, r, v1);
  return `M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`;
}

export function Gauge({ value, label, size = 280, zones = [[0, 0.25, "var(--ok)"], [0.25, 0.5, "var(--warn-fill)"], [0.5, 1, "var(--bad)"]], idle }) {
  const w = size, h = Math.round(size * 0.62), cx = w / 2, cy = h - 18, r = w / 2 - 22;
  const v = idle || value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="chart" style={{ width: w, maxWidth: "100%", margin: "0 auto" }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label || "Gauge"}>
        {zones.map(([a, b, color]) => (
          <path key={a} d={arc(cx, cy, r, a + 0.004, b - 0.004)} stroke={color} strokeWidth="16" fill="none" opacity={idle ? 0.25 : 0.9} />
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const [x, y] = polar(cx, cy, r + 17, t);
          return <text key={t} x={x} y={y + 4} textAnchor="middle">{Math.round(t * 100)}</text>;
        })}
        <g style={{ transform: `rotate(${v * 180 - 90}deg)`, transformOrigin: `${cx}px ${cy}px`, transition: "transform .7s cubic-bezier(.2,.9,.3,1.1)" }}>
          <line x1={cx} y1={cy} x2={cx} y2={cy - r + 8} stroke="var(--ink)" strokeWidth="3" strokeLinecap="round" />
        </g>
        <circle cx={cx} cy={cy} r="7" fill="var(--ink)" />
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ horizontal bars (CSS) */
export function HBars({ items, max, format = (v) => v, labelWidth = 150 }) {
  const top = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <div className="hbars" style={{ "--lw": `${labelWidth}px` }}>
      {items.map((i) => (
        <div className="hbar" key={i.label} title={i.title}>
          <span className="lbl">{i.label}</span>
          <span className="track"><span className="fill" style={{ width: `${Math.max(0, Math.min(100, (i.value / top) * 100))}%`, background: i.color }} /></span>
          <span className="val">{format(i.value)}</span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ grouped vertical bars */
export function GroupedBars({ categories, series, height = 260, domain = [0, 1], format = (v) => v.toFixed(2) }) {
  const [ref, width] = useWidth();
  const m = { l: 38, r: 8, t: 18, b: 28 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const [lo, hi] = domain;
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const groupW = iw / categories.length;
  const barW = Math.max(6, Math.min(34, (groupW - 14) / series.length));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * (hi - lo));
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Grouped bar chart">
        {ticks.map((t) => (
          <g key={t}><line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke="var(--line-soft)" /><text x={m.l - 6} y={y(t) + 4} textAnchor="end">{tickFmt(t)}</text></g>
        ))}
        {categories.map((c, ci) => {
          const gx = m.l + ci * groupW + (groupW - barW * series.length) / 2;
          return (
            <g key={c}>
              {series.map((s, si) => {
                const v = s.values[ci];
                if (v === null || v === undefined) return null;
                const x = gx + si * barW;
                return (
                  <g key={s.name}>
                    <rect x={x + 1} y={y(v)} width={Math.max(1, barW - 2)} height={Math.max(0, y(lo) - y(v))} fill={s.color} rx="2"><title>{`${s.name} \u2013 ${c}: ${format(v)}`}</title></rect>
                    {barW > 24 && <text x={x + barW / 2} y={y(v) - 5} textAnchor="middle" style={{ fontSize: 10.5 }}>{format(v)}</text>}
                  </g>
                );
              })}
              <text x={m.l + ci * groupW + groupW / 2} y={height - 8} textAnchor="middle">{c}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ line chart (category x-axis) */
export function LineChart({ series, labels, height = 240, domain, refLines = [], format = (v) => tickFmt(v) }) {
  const [ref, width] = useWidth();
  const m = { l: 44, r: 12, t: 12, b: 28 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const all = series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
  if (!all.length) return <div className="chart muted small" ref={ref}>No data points yet.</div>;
  let lo = domain?.[0] ?? Math.min(...all, ...refLines.map((r) => r.y));
  let hi = domain?.[1] ?? Math.max(...all, ...refLines.map((r) => r.y));
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const pad = domain ? 0 : (hi - lo) * 0.1;
  lo -= pad; hi += pad;
  if (!domain && Math.min(...all) >= 0) lo = Math.max(lo, 0);
  const n = Math.max(...series.map((s) => s.values.length));
  const x = (i) => m.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * (hi - lo));
  const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / 70))));
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Line chart">
        {ticks.map((t) => (
          <g key={t}><line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke="var(--line-soft)" /><text x={m.l - 6} y={y(t) + 4} textAnchor="end">{format(t)}</text></g>
        ))}
        {refLines.map((r) => (
          <g key={r.label}><line x1={m.l} x2={width - m.r} y1={y(r.y)} y2={y(r.y)} stroke={r.color || "var(--bad)"} strokeDasharray="5 4" /><text x={width - m.r} y={y(r.y) - 5} textAnchor="end" style={{ fill: r.color || "var(--bad)" }}>{r.label}</text></g>
        ))}
        {labels && labels.map((l, i) => (i % every === 0 ? <text key={i} x={x(i)} y={height - 8} textAnchor="middle">{l}</text> : null))}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (v === null || v === undefined ? null : [x(i), y(v), v, i])).filter(Boolean);
          return (
            <g key={s.name}>
              <polyline points={pts.map((p) => `${p[0]},${p[1]}`).join(" ")} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeDasharray={s.dashed ? "4 3" : undefined} />
              {pts.length <= 40 && pts.map((p) => <circle key={p[3]} cx={p[0]} cy={p[1]} r="3" fill={s.color}><title>{`${s.name}${labels ? ` \u2013 ${labels[p[3]]}` : ""}: ${format(p[2])}`}</title></circle>)}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ stacked histogram */
export function Histogram({ hist, height = 92, unit }) {
  const [ref, width] = useWidth(240);
  const { edges, normal, failure } = hist;
  const totals = normal.map((v, i) => v + failure[i]);
  const top = Math.max(...totals, 1);
  const bw = width / totals.length;
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height + 16} role="img" aria-label="Distribution">
        {totals.map((t, i) => {
          const hN = (normal[i] / top) * height, hF = (failure[i] / top) * height;
          return (
            <g key={i}>
              <rect x={i * bw + 0.5} y={height - hN - hF} width={Math.max(1, bw - 1)} height={hF} fill="var(--bad)"><title>{`${tickFmt(edges[i], 1)}\u2013${tickFmt(edges[i + 1], 1)}${unit ? ` ${unit}` : ""}: ${t} rows, ${failure[i]} failures`}</title></rect>
              <rect x={i * bw + 0.5} y={height - hN} width={Math.max(1, bw - 1)} height={hN} fill="var(--accent)" opacity="0.55"><title>{`${t} rows`}</title></rect>
            </g>
          );
        })}
        <line x1="0" x2={width} y1={height} y2={height} stroke="var(--line)" />
        <text x="0" y={height + 13}>{tickFmt(edges[0], 1)}</text>
        <text x={width} y={height + 13} textAnchor="end">{tickFmt(edges[edges.length - 1], 1)}</text>
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ confusion matrix */
export function ConfusionMatrix({ cm }) {
  if (!cm) return <div className="muted small">{dash}</div>;
  const { true_negatives: tn, false_positives: fp, false_negatives: fn, true_positives: tp } = cm;
  const f = (n) => n.toLocaleString();
  return (
    <div className="cm" role="table" aria-label="Confusion matrix">
      <div />
      <div className="h">Predicted normal</div>
      <div className="h">Predicted failure</div>
      <div className="side">Actually normal</div>
      <div className="cell good"><b>{f(tn)}</b><span>Correctly cleared</span></div>
      <div className="cell fa"><b>{f(fp)}</b><span>False alarms</span></div>
      <div className="side">Actually failed</div>
      <div className="cell miss"><b>{f(fn)}</b><span>Missed failures</span></div>
      <div className="cell good"><b>{f(tp)}</b><span>Caught failures</span></div>
    </div>
  );
}

/* ------------------------------------------------------------------ small pieces */
export function Sparkline({ values, color = "var(--accent)", width = 120, height = 32 }) {
  const v = values.filter((x) => x !== null && x !== undefined);
  if (v.length < 2) return null;
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = v.map((val, i) => `${(i / (v.length - 1)) * (width - 4) + 2},${height - 3 - ((val - lo) / span) * (height - 6)}`).join(" ");
  return <svg width={width} height={height} aria-hidden="true"><polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" /></svg>;
}

/** Value bar with a threshold marker (e.g. recall vs. minimum acceptable recall). */
export function Bullet({ value, threshold, max = 1, tone = "ok", label }) {
  return (
    <div className="bullet" role="img" aria-label={label}>
      <div className="v" style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: `var(--${tone === "ok" ? "ok" : tone === "warn" ? "warn-fill" : "bad"})` }} />
      {threshold !== undefined && <div className="t" style={{ left: `${(threshold / max) * 100}%` }} title={`Threshold ${threshold}`} />}
    </div>
  );
}

export function Legend({ items }) {
  return <div className="legend">{items.map((i) => <span key={i.name}><i style={{ background: i.color }} />{i.name}</span>)}</div>;
}
