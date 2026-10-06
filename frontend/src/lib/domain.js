// Sensor schema, presets and physical failure-mode rules (AI4I 2020 dataset documentation).
export const SENSORS = [
  { key: "air", api: "Air temperature [K]", label: "Air temperature", unit: "K", step: 0.1, fallback: [295, 305] },
  { key: "process", api: "Process temperature [K]", label: "Process temperature", unit: "K", step: 0.1, fallback: [305, 315] },
  { key: "rpm", api: "Rotational speed [rpm]", label: "Rotational speed", unit: "rpm", step: 1, fallback: [1100, 2900] },
  { key: "torque", api: "Torque [Nm]", label: "Torque", unit: "Nm", step: 0.1, fallback: [3, 77] },
  { key: "wear", api: "Tool wear [min]", label: "Tool wear", unit: "min", step: 1, fallback: [0, 255] },
];

export const PRESETS = [
  { id: "normal", label: "Normal operation", type: "M", air: 298.1, process: 308.6, rpm: 1551, torque: 42.8, wear: 0 },
  { id: "heat", label: "Poor heat dissipation", type: "L", air: 302.4, process: 310.4, rpm: 1290, torque: 52, wear: 90 },
  { id: "power", label: "Power spike", type: "M", air: 299.2, process: 309.1, rpm: 2400, torque: 63, wear: 60 },
  { id: "overstrain", label: "Overstrain", type: "L", air: 298.5, process: 309.5, rpm: 1450, torque: 68, wear: 215 },
  { id: "worn", label: "Worn tool", type: "H", air: 299, process: 309.8, rpm: 1500, torque: 45, wear: 238 },
];

export const toPayload = (v) => ({
  Type: v.type,
  "Air temperature [K]": Number(v.air),
  "Process temperature [K]": Number(v.process),
  "Rotational speed [rpm]": Number(v.rpm),
  "Torque [Nm]": Number(v.torque),
  "Tool wear [min]": Number(v.wear),
});

export const derive = (v) => {
  const powerW = Number(v.torque) * Number(v.rpm) * (2 * Math.PI) / 60;
  return {
    tempDiff: Number(v.process) - Number(v.air),
    powerKw: powerW / 1000,
    powerW,
    wearTorque: Number(v.wear) * Number(v.torque),
  };
};

const OSF_LIMIT = { L: 11000, M: 12000, H: 13000 };

/**
 * Proximity (0..1, 1 = rule triggered) to each physical failure mode.
 * Rules from the AI4I 2020 documentation: HDF (temp gap < 8.6 K and speed < 1380 rpm), PWF (power < 3500 W or > 9000 W),
 * OSF (wear x torque above a product-type limit), TWF (tool replaced/fails around 200-240 min).
 */
export function failureChecks(v) {
  const d = derive(v);
  const hdfGap = Math.max(0, Math.min(1, (10.6 - d.tempDiff) / 2));
  const hdfRpm = Math.max(0, Math.min(1, (1500 - Number(v.rpm)) / 120));
  const hdf = Math.min(hdfGap, hdfRpm);
  const pwf = d.powerW < 3500 ? Math.min(1, 0.7 + (3500 - d.powerW) / 3000)
    : d.powerW > 9000 ? 1 : Math.max(0, Math.min(0.7, (d.powerW - 7500) / 1500 * 0.7));
  const osfLimit = OSF_LIMIT[v.type] ?? 12000;
  const osf = Math.max(0, Math.min(1, d.wearTorque / osfLimit));
  const twf = Number(v.wear) >= 200 ? 1 : Math.max(0, Math.min(0.9, (Number(v.wear) - 150) / 50));
  const level = (p, hit = 1) => (p >= hit ? "bad" : p >= 0.7 ? "warn" : "ok");
  return [
    { id: "HDF", label: "Heat dissipation", detail: `Temp gap ${d.tempDiff.toFixed(1)} K at ${Math.round(v.rpm)} rpm (risk below 8.6 K and 1380 rpm)`, value: hdf, level: level(hdf) },
    { id: "PWF", label: "Power", detail: `${Math.round(d.powerW).toLocaleString()} W (safe band 3,500\u20139,000 W)`, value: pwf, level: level(pwf) },
    { id: "OSF", label: "Overstrain", detail: `Wear \u00d7 torque ${Math.round(d.wearTorque).toLocaleString()} (limit ${osfLimit.toLocaleString()} for type ${v.type})`, value: osf, level: level(osf) },
    { id: "TWF", label: "Tool wear", detail: `${Math.round(v.wear)} min (failures cluster at 200\u2013240 min)`, value: twf, level: level(twf) },
  ];
}

export const probabilityBand = (p) => (p >= 0.5 ? "bad" : p >= 0.25 ? "warn" : "ok");
