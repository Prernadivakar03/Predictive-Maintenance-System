import { useState } from "react";
import { figureUrl } from "../lib/api.js";
import { Modal } from "./ui.jsx";

export default function Figures({ figures }) {
  const [open, setOpen] = useState(null);
  if (!figures?.length) return <div className="muted">No figures have been generated yet.</div>;
  return (
    <>
      <div className="figure-grid">
        {figures.map((f) => (
          <button key={f.name} className="figure" onClick={() => setOpen(f)} aria-label={`Enlarge ${f.title}`}>
            <img src={figureUrl(f.name)} alt={f.title} loading="lazy" />
            <figcaption>{f.title}</figcaption>
          </button>
        ))}
      </div>
      {open && (
        <Modal title={open.title} onClose={() => setOpen(null)}>
          <div className="lightbox" style={{ display: "grid", justifyItems: "center" }}><img src={figureUrl(open.name)} alt={open.title} /></div>
        </Modal>
      )}
    </>
  );
}
