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

export interface Candle { date: string; open: number; high: number; low: number; close: number; sma20: number | null; bb_up?: number | null; bb_dn?: number | null }
export interface TradeMark { date: string; price: number; side: string }

const CW = 760, CH = 320, CP = { l: 64, r: 16, t: 14, b: 26 };

/** Gráfico de velas OHLC con SMA20, Bandas de Bollinger y entradas/salidas de la corrida. */
export function CandleChart({ candles, trades = [], fmt }: { candles: Candle[]; trades?: TradeMark[]; fmt: (v: number) => string }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!candles.length) return <div className="empty">Sin velas todavía</div>;
  const vals = candles.flatMap(c => [c.high, c.low, c.bb_up ?? c.high, c.bb_dn ?? c.low]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
  const n = candles.length, step = (CW - CP.l - CP.r) / n, bw = Math.max(2, step * 0.62);
  const X = (i: number) => CP.l + step * (i + 0.5);
  const Y = (v: number) => CP.t + (1 - (v - lo) / (hi - lo)) * (CH - CP.t - CP.b);
  const idx = new Map(candles.map((c, i) => [c.date, i]));
  const line = (key: "sma20" | "bb_up" | "bb_dn") => candles.map((c, i) => c[key] == null ? null : `${X(i).toFixed(1)},${Y(c[key] as number).toFixed(1)}`)
    .reduce((acc: string, pt, i, arr) => pt == null ? acc : acc + (i && arr[i - 1] != null ? "L" : "M") + pt, "");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => lo + t * (hi - lo));
  const h = hover != null ? candles[hover] : null;
  const markShape: Record<string, { d: string; cls: string; below: boolean }> = {
    BUY: { d: "M0,-7 L6,4 L-6,4 Z", cls: "mk-buy", below: true }, COVER: { d: "M0,-7 L6,4 L-6,4 Z", cls: "mk-cover", below: true },
    SELL: { d: "M0,7 L6,-4 L-6,-4 Z", cls: "mk-sell", below: false }, SHORT: { d: "M0,7 L6,-4 L-6,-4 Z", cls: "mk-short", below: false },
  };
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="Velas diarias BTC/USDT con SMA20 y Bandas de Bollinger"
        onMouseMove={e => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const i = Math.floor((((e.clientX - r.left) / r.width) * CW - CP.l) / step);
          setHover(i >= 0 && i < n ? i : null);
        }} onMouseLeave={() => setHover(null)}>
        {ticks.map((t, i) => <g key={i}><line x1={CP.l} x2={CW - CP.r} y1={Y(t)} y2={Y(t)} className="grid" />
          <text x={CP.l - 8} y={Y(t) + 4} textAnchor="end" className="axis">{fmt(t)}</text></g>)}
        <text x={CP.l} y={CH - 6} className="axis">{candles[0].date}</text>
        <text x={CW - CP.r} y={CH - 6} textAnchor="end" className="axis">{candles[n - 1].date}</text>
        <path d={line("bb_up")} className="bb" /><path d={line("bb_dn")} className="bb" />
        <path d={line("sma20")} className="sma" />
        {candles.map((c, i) => {
          const up = c.close >= c.open, x = X(i);
          return <g key={c.date} className={up ? "c-up" : "c-dn"}>
            <line x1={x} x2={x} y1={Y(c.high)} y2={Y(c.low)} />
            <rect x={x - bw / 2} y={Y(Math.max(c.open, c.close))} width={bw} height={Math.max(1, Math.abs(Y(c.open) - Y(c.close)))} />
          </g>;
        })}
        {trades.map((t, k) => {
          const i = idx.get(t.date); const m = markShape[t.side];
          if (i == null || !m) return null;
          const c = candles[i]; const y = m.below ? Y(c.low) + 10 : Y(c.high) - 10;
          return <path key={k} transform={`translate(${X(i)},${y})`} d={m.d} className={m.cls}><title>{`${t.side} ${t.date} @ ${fmt(t.price)}`}</title></path>;
        })}
        {hover != null && <line x1={X(hover)} x2={X(hover)} y1={CP.t} y2={CH - CP.b} className="cursor" />}
      </svg>
      <div className="legend">
        <span><i className="lg-up" />Vela alcista</span><span><i className="lg-dn" />Vela bajista</span>
        <span><i style={{ background: "var(--c3)" }} />SMA20</span><span><i className="lg-bb" />Bollinger (20, 2σ)</span>
        <span><i className="lg-buy" />Compra</span><span><i style={{ background: "var(--c1)" }} />Cover</span><span><i className="lg-short" />Short</span><span><i className="lg-sell" />Venta</span>
        {h && <span className="hover-read">{h.date} · A {fmt(h.open)} · M {fmt(h.high)} · m {fmt(h.low)} · C {fmt(h.close)}{h.sma20 ? ` · SMA20 ${fmt(h.sma20)}` : ""}</span>}
      </div>
    </div>
  );
}
