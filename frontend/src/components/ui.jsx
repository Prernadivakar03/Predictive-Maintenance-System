import { useEffect } from "react";
import Icon from "./Icon.jsx";

export function Panel({ title, sub, right, children, flush, tone, className = "" }) {
  return (
    <section className={`panel ${tone ? `tone-${tone}` : ""} ${className}`}>
      {(title || right) && (
        <header className="panel-h">
          {title && <h2>{title}</h2>}
          {sub && <span className="sub">{sub}</span>}
          {right && <div className="right">{right}</div>}
        </header>
      )}
      <div className={`panel-b ${flush ? "flush" : ""}`}>{children}</div>
    </section>
  );
}

export function PageHead({ title, children, actions }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}

export function Badge({ tone = "", children, plain }) {
  return <span className={`badge ${tone} ${plain ? "plain" : ""}`}>{children}</span>;
}

export function Button({ variant = "", size = "", icon, busy, children, ...rest }) {
  return (
    <button type="button" className={`btn ${variant} ${size}`} disabled={busy || rest.disabled} {...rest}>
      {(icon || busy) && <Icon name={busy ? "loader" : icon} spin={busy} />}
      {children}
    </button>
  );
}

export function Stats({ items }) {
  return (
    <div className="stats">
      {items.map((s) => (
        <div key={s.label} className={`stat ${s.tone || ""}`}>
          <div className="k">{s.label}</div>
          <div className="v">{s.value}{s.unit && <small>{s.unit}</small>}</div>
          {s.note && <div className="n">{s.note}</div>}
        </div>
      ))}
    </div>
  );
}

export function KV({ items }) {
  return (
    <dl className="kv">
      {items.filter(Boolean).map(([k, v]) => (<div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd>{v}</dd></div>))}
    </dl>
  );
}

export function Empty({ title, children, action }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Banner({ tone = "", icon, title, children, action }) {
  return (
    <div className={`banner ${tone}`} role={tone === "bad" ? "alert" : undefined}>
      <Icon name={icon || (tone === "ok" ? "check" : tone ? "alert" : "info")} />
      <div style={{ flex: 1 }}>
        {title && <strong>{title}</strong>}
        {children && <div className={title ? "muted" : ""}>{children}</div>}
      </div>
      {action}
    </div>
  );
}

export function Skeleton({ height = 14, width = "100%" }) {
  return <div className="skeleton" style={{ height, width }} />;
}

export function PageSkeleton() {
  return (
    <div className="stack" aria-busy="true" aria-label="Loading">
      <Skeleton height={86} /><Skeleton height={220} /><Skeleton height={220} />
    </div>
  );
}

/** Standard async states for a page: loading skeleton, API error banner, then children(data). */
export function Async({ state, children, emptyWhen }) {
  const { data, error, loading, reload } = state;
  if (!data && loading) return <PageSkeleton />;
  if (!data && error) {
    return (
      <Banner tone="bad" title={error.status === 0 ? "API unreachable" : "Could not load this page"}
        action={<Button size="sm" onClick={reload} icon="refresh">Retry</Button>}>
        {error.message}
      </Banner>
    );
  }
  if (emptyWhen && data && emptyWhen(data)) return null;
  return children(data);
}

export function Modal({ title, children, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}

export function ConfirmModal({ title, children, confirmLabel, danger, busy, onConfirm, onClose }) {
  return (
    <Modal title={title} onClose={onClose}>
      <div className="muted">{children}</div>
      <div className="row">
        <Button onClick={onClose}>Cancel</Button>
        <Button variant={danger ? "danger" : "primary"} busy={busy} onClick={onConfirm}>{confirmLabel}</Button>
      </div>
    </Modal>
  );
}

export function Segmented({ value, onChange, options, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Tabs({ value, onChange, options }) {
  return (
    <div className="tabs" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export const StatusBadge = ({ status }) => {
  const map = {
    succeeded: ["ok", "Succeeded"], running: ["info", "Running"], failed: ["bad", "Failed"], interrupted: ["warn", "Interrupted"],
    PROMOTED: ["ok", "Promoted"], REJECTED_INFERIOR: ["bad", "Rejected"], FAILED: ["bad", "Failed"], SKIPPED: ["", "Skipped"],
    FINISHED: ["ok", "Finished"],
  };
  const [tone, label] = map[status] || ["", status];
  return <Badge tone={tone}>{label}</Badge>;
};
