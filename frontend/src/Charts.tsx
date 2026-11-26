import { useMemo, useState } from "react";

export interface Series { name: string; color: string; points: { x: string; y: number | null }[]; dashed?: boolean }
export interface Marker { x: string; y: number; kind: "BUY" | "SELL" }

const W = 760, H = 260, PAD = { l: 64, r: 16, t: 14, b: 26 };

export function LineChart({ series, markers = [], fmt }: { series: Series[]; markers?: Marker[]; fmt: (v: number) => string }) {
  const [hover, setHover] = useState<number | null>(null);
  const xs = useMemo(() => Array.from(new Set(series.flatMap(s => s.points.map(p => p.x)))).sort(), [series]);
  const ys = series.flatMap(s => s.points.map(p => p.y)).filter((v): v is number => v != null)
    .concat(markers.map(m => m.y));
  if (!xs.length || !ys.length) return <div className="empty">Sin datos todavía</div>;
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const xi = new Map(xs.map((x, i) => [x, i]));
  const X = (x: string) => PAD.l + ((xi.get(x) ?? 0) / Math.max(xs.length - 1, 1)) * (W - PAD.l - PAD.r);
  const Y = (y: number) => PAD.t + (1 - (y - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => lo + t * (hi - lo));
  const path = (s: Series) => s.points.filter(p => p.y != null)
    .map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y as number).toFixed(1)}`).join("");
  const hx = hover != null ? xs[hover] : null;

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        onMouseMove={e => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (xs.length - 1));
          setHover(i >= 0 && i < xs.length ? i : null);
        }}
        onMouseLeave={() => setHover(null)}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD.l} x2={W - PAD.r} y1={Y(t)} y2={Y(t)} className="grid" />
            <text x={PAD.l - 8} y={Y(t) + 4} textAnchor="end" className="axis">{fmt(t)}</text>
          </g>
        ))}
        <text x={PAD.l} y={H - 6} className="axis">{xs[0]}</text>
        <text x={W - PAD.r} y={H - 6} textAnchor="end" className="axis">{xs[xs.length - 1]}</text>
        {series.map(s => <path key={s.name} d={path(s)} fill="none" stroke={s.color} strokeWidth={2}
          strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" />)}
        {markers.map((m, i) => xi.has(m.x) && (
          <path key={i} transform={`translate(${X(m.x)},${Y(m.y)})`}
            d={m.kind === "BUY" ? "M0,-7 L6,4 L-6,4 Z" : "M0,7 L6,-4 L-6,-4 Z"}
            className={m.kind === "BUY" ? "mk-buy" : "mk-sell"} />
        ))}
        {hx && <line x1={X(hx)} x2={X(hx)} y1={PAD.t} y2={H - PAD.b} className="cursor" />}
      </svg>
      <div className="legend">
        {series.map(s => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}
        {markers.length > 0 && <><span><i className="lg-buy" />Compra</span><span><i className="lg-sell" />Venta</span></>}
        {hx && <span className="hover-read">{hx}: {series.map(s => {
          const p = s.points.find(q => q.x === hx); return p?.y != null ? `${s.name} ${fmt(p.y)}` : null;
        }).filter(Boolean).join(" · ")}</span>}
      </div>
    </div>
  );
}
