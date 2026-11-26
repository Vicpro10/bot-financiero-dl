import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, clearKey, getKey, setKey, type EventRow, type Market, type RunDetail, type RunMetrics, type Status } from "./api";
import { CandleChart, LineChart, type TradeMark } from "./Charts";

const usd = (v: number | null | undefined, d = 2) =>
  v == null ? "—" : v.toLocaleString("es-MX", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number | null | undefined, d = 2) => (v == null ? "no definido" : `${(v * 100).toFixed(d)}%`);
const signed = (v: number | null | undefined) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${usd(v)}`);
const cls = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "");

const FAULTS: [string, string][] = [
  ["mexc_down", "API MEXC sin respuesta / 429"],
  ["stale_data", "Dato atrasado (solo live)"],
  ["ollama_down", "Ollama detenido"],
  ["invalid_json", "JSON inválido del LLM"],
  ["insufficient_balance", "Saldo insuficiente"],
  ["duplicate_event", "Evento duplicado"],
];

function Login({ onOk }: { onOk: () => void }) {
  const [k, setK] = useState("");
  const [err, setErr] = useState("");
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setKey(k.trim());
    try { await api("/status"); onOk(); }
    catch (x) { clearKey(); setErr(x instanceof ApiError && x.status === 401 ? "Clave incorrecta: acceso rechazado." : String(x)); }
  };
  return (
    <div className="login">
      <form onSubmit={submit} className="card">
        <h1>Bot Financiero DL</h1>
        <p className="muted">Agente Paper sobre datos públicos de MEXC. Escriba la API key del backend (está en <code>backend/.env</code>, nunca en el frontend).</p>
        <input type="password" placeholder="AGENT_API_KEY" value={k} onChange={e => setK(e.target.value)} autoFocus />
        <button className="primary" disabled={!k}>Entrar</button>
        {err && <p className="neg">{err}</p>}
      </form>
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(!!getKey());
  const [st, setSt] = useState<Status | null>(null);
  const [mk, setMk] = useState<Market | null>(null);
  const [runs, setRuns] = useState<RunMetrics[]>([]);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({ mode: "replay", strategy: "model", replay_days: 60, allow_short: false });
  const [mkErr, setMkErr] = useState("");
  const [wsOn, setWsOn] = useState(false);
  const refreshRef = useRef<() => void>(() => {});
  const [llm, setLlm] = useState<{ q: string; a: string }>({ q: "Resume en una frase qué es el modo Paper.", a: "" });

  const refresh = useCallback(async () => {
    try {
      const [s, r, ev] = await Promise.all([api<Status>("/status"), api<RunMetrics[]>("/runs"), api<EventRow[]>("/events?limit=60")]);
      setSt(s); setRuns(r); setEvents(ev);
      const id = sel ?? s.run?.id ?? r[0]?.run_id;
      if (id) setDetail(await api<RunDetail>(`/runs/${id}`));
      setErr("");
    } catch (x) {
      if (x instanceof ApiError && x.status === 401) { clearKey(); setAuthed(false); }
      setErr(String((x as Error).message));
    }
  }, [sel]);

  const refreshMarket = useCallback(async () => {
    try { setMk(await api<Market>("/market?days=120")); setMkErr(""); } catch (x) { setMkErr((x as Error).message); }
  }, []);

  refreshRef.current = refresh;
  // Con WebSocket conectado el panel se actualiza por eventos; el sondeo REST queda como respaldo lento.
  useEffect(() => { if (!authed) return; refresh(); const t = setInterval(refresh, wsOn ? 10000 : 2500); return () => clearInterval(t); }, [authed, refresh, wsOn]);
  useEffect(() => {
    if (!authed) return;
    let ws: WebSocket | null = null, retry: number | undefined, closed = false, pending: number | undefined;
    const connect = () => {
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.onopen = () => ws?.send(JSON.stringify({ type: "auth", key: getKey() }));
      ws.onmessage = ev => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "hello") setWsOn(true);
        else if (msg.type === "ticker") setMk(prev => prev ? { ...prev, ticker: { ...prev.ticker, last: msg.last, change_pct: msg.change_pct, high: msg.high, low: msg.low, time: msg.time } } : prev);
        else if (["decision", "status", "event"].includes(msg.type)) {
          window.clearTimeout(pending); pending = window.setTimeout(() => refreshRef.current(), 250);
        }
      };
      ws.onclose = () => { setWsOn(false); if (!closed) retry = window.setTimeout(connect, 3000); };
    };
    connect();
    return () => { closed = true; window.clearTimeout(retry); ws?.close(); };
  }, [authed]);
  useEffect(() => { if (!authed) return; refreshMarket(); const t = setInterval(refreshMarket, 30000); return () => clearInterval(t); }, [authed, refreshMarket]);

  if (!authed) return <Login onOk={() => setAuthed(true)} />;

  const act = async (path: string, body?: unknown) => {
    setBusy(true);
    try { await api(path, { method: "POST", body }); setSel(null); await refresh(); }
    catch (x) { setErr((x as Error).message); }
    finally { setBusy(false); }
  };

  const m = detail?.metrics;
  const p = st?.portfolio;
  const lastPrice = mk?.ticker.last ?? detail?.equity.at(-1)?.price ?? 0;
  const equityNow = p ? p.cash + p.qty * (st?.status === "running" && st.run?.mode === "replay" ? (detail?.equity.at(-1)?.price ?? lastPrice) : lastPrice) : null;
  const status = st?.status ?? "…";
  const dec = st?.last_decision;

  // gráfica de precio de la corrida (replay) o del mercado
  const priceRows = detail?.equity.length ? detail.equity : [];
  const trades: TradeMark[] = (detail?.orders ?? []).filter(o => o.status === "FILLED" && o.exec_price)
    .map(o => ({ date: o.exec_date, price: o.exec_price as number, side: o.side }));
  const posName = !p ? "—" : p.qty > 0 ? "LONG" : p.qty < 0 ? "SHORT" : "EFECTIVO";
  const sideCls = (s: string) => (s === "BUY" || s === "COVER" ? "pos" : s === "SHORT" ? "pos-short" : s === "SELL" ? "neg" : "");
  const cap = detail?.metrics && runs.find(r => r.run_id === detail.metrics.run_id) ? (st?.run?.initial_capital ?? 50000) : 50000;
  const bh = priceRows.length ? priceRows.map(r => ({ x: r.date, y: cap * r.price / priceRows[0].price })) : [];

  return (
    <div className="app">
      <header>
        <div>
          <h1>Bot Financiero DL <span className="muted">· {st?.symbol ?? "BTCUSDT"} · MEXC</span></h1>
          <p className="muted small">Modelo {st?.model.version} ({st?.model.kind}) · regla: pronóstico &gt; δ={pct(st?.model.signal_rule.delta, 2)}{st?.model.signal_rule.sma_filter ? " y cierre > SMA20" : ""} · <b>Modo Paper: órdenes reales deshabilitadas</b></p>
        </div>
        <div className="hdr-right">
          <span className={`ws ${wsOn ? "on" : "off"}`} title="Canal WebSocket del backend">{wsOn ? "● WebSocket en vivo" : "○ WebSocket desconectado · usando REST"}</span>
          <span className={`pill s-${status}`}>{status.toUpperCase()}</span>
          <button className="ghost" onClick={() => { clearKey(); setAuthed(false); }}>Salir</button>
        </div>
      </header>
      {st?.status_reason && <div className={`banner ${status === "halted" ? "b-err" : ""}`}>{st.status_reason}</div>}
      {err && <div className="banner b-err">{err}</div>}

      <section className="controls card">
        <label>Modo
          <select value={form.mode} onChange={e => setForm({ ...form, mode: e.target.value })}>
            <option value="replay">Replay histórico (etiquetado)</option>
            <option value="live">Live MEXC (vela diaria)</option>
          </select>
        </label>
        <label>Estrategia
          <select value={form.strategy} onChange={e => setForm({ ...form, strategy: e.target.value })}>
            <option value="model">Modelo sin LLM</option>
            <option value="llm">Agente con LLM (Ollama)</option>
          </select>
        </label>
        {form.mode === "replay" && <label>Días
          <input type="number" min={20} max={365} value={form.replay_days} onChange={e => setForm({ ...form, replay_days: +e.target.value })} />
        </label>}
        <label className="check" title="Permite abrir cortos en Paper, 100% respaldados por efectivo (sin apalancamiento)">
          <input type="checkbox" checked={form.allow_short} onChange={e => setForm({ ...form, allow_short: e.target.checked })} />
          Long/Short (permitir cortos)
        </label>
        <div className="btns">
          <button className="primary" disabled={busy || status === "running"} onClick={() => act("/agent/start", form)}>Iniciar</button>
          <button disabled={busy || status !== "running"} onClick={() => act("/agent/pause")}>Pausar</button>
          <button disabled={busy || !["paused", "halted"].includes(status)} onClick={() => act("/agent/resume")}>Reanudar</button>
          <button className="danger" disabled={busy || ["stopped", "finished"].includes(status)} onClick={() => act("/agent/stop")}>Detener</button>
          <button className="ghost" disabled={busy || !p || p.qty === 0 || status === "running"} onClick={() => act("/agent/liquidate")}>Liquidar posición</button>
        </div>
      </section>

      <section className="grid4">
        <div className="card kpi">
          <h3>Precio MEXC (en vivo)</h3>
          {mkErr && !mk ? <div className="err-box"><span className="neg small">No se pudo leer MEXC: {mkErr}</span><button onClick={refreshMarket}>Reintentar</button></div>
            : !mk ? <><div className="skeleton" /><p className="muted small">Cargando datos de MEXC…</p></>
            : <>
              <div className="big">{usd(mk.ticker.last)} <small>USDT</small></div>
              <div className={cls(mk.ticker.change_pct)}>{`${(mk.ticker.change_pct * 100).toFixed(2)}% 24h`}{wsOn ? " · en vivo" : ""}</div>
              <div className="muted small">Última vela cerrada: {mk.last_closed_candle} (UTC){mkErr ? " · ⚠ dato no actualizado" : ""}</div>
            </>}
        </div>
        <div className="card kpi">
          <h3>Pronóstico del modelo</h3>
          <div className={`big ${cls(mk?.forecast.next_return)}`}>{mk ? `${(mk.forecast.next_return * 100).toFixed(3)}%` : "—"}</div>
          <div>Solo largo: <b className={mk?.forecast.signal === "LONG" ? "pos" : ""}>{mk?.forecast.signal === "LONG" ? "COMPRAR / MANTENER" : "EFECTIVO"}</b></div>
          <div>Long/Short: <b className={mk?.forecast.signal_long_short === "LONG" ? "pos" : mk?.forecast.signal_long_short === "SHORT" ? "pos-short" : ""}>{mk?.forecast.signal_long_short ?? "—"}</b></div>
          <div className="muted small">Rendimiento estimado del día siguiente a {mk?.forecast.cutoff}</div>
        </div>
        <div className="card kpi">
          <h3>Patrimonio Paper</h3>
          <div className="big">{usd(m?.patrimonio ?? equityNow)} <small>USDT</small></div>
          <div className={cls(m?.pnl_total)}>PnL {signed(m?.pnl_total)} ({pct(m?.rendimiento_neto)})</div>
          <div className="muted small">Efectivo {usd(p?.cash)} · <b className={posName === "LONG" ? "pos-long" : posName === "SHORT" ? "pos-short" : ""}>{posName}</b> {p ? Math.abs(p.qty).toFixed(6) : "—"} BTC{p && p.qty !== 0 ? ` · Entry ${usd(p.entry_price)}` : ""}</div>
        </div>
        <div className="card kpi">
          <h3>Indicadores (cierre)</h3>
          <table className="mini"><tbody>
            <tr><td>SMA20</td><td>{usd(mk?.indicators.sma20)}</td></tr>
            <tr><td>Distancia SMA20</td><td className={cls(mk?.indicators.dist_sma20)}>{pct(mk?.indicators.dist_sma20)}</td></tr>
            <tr><td>Volatilidad 20d</td><td>{pct(mk?.indicators.vol20)}</td></tr>
            <tr><td>RSI 14</td><td>{mk?.indicators.rsi14.toFixed(1)}</td></tr>
          </tbody></table>
        </div>
      </section>

      <section className="grid2">
        <div className="card">
          <h3>Última decisión del agente</h3>
          {dec ? <div className="decision">
            <div><span className="muted">Vela</span> {dec.candle_date} · <span className="muted">fuente</span> {dec.proposal_source}</div>
            <div className="act">Propuesta: <b>{dec.proposed_action}</b> → Final: <b className={sideCls(dec.final_action)}>{dec.final_action}</b>
              <span className={`pill r-${dec.risk_status}`}>{dec.risk_status}</span></div>
            <div className="reason">“{dec.reason}”</div>
            <div className="muted small">Riesgo: {dec.risk_reason}{dec.llm_latency_ms ? ` · latencia LLM ${dec.llm_latency_ms.toFixed(0)} ms` : ""}</div>
          </div> : <p className="muted">Inicie una corrida para ver decisiones.</p>}
        </div>
        <div className="card">
          <h3>Estado de riesgo</h3>
          <table className="mini"><tbody>
            <tr><td>Exposición máxima</td><td>{pct(st?.risk.max_exposure as number, 0)}</td></tr>
            <tr><td>Pérdida diaria máxima</td><td>{pct(st?.risk.max_daily_loss as number, 0)}</td></tr>
            <tr><td>Drawdown que detiene</td><td>{pct(st?.risk.max_drawdown_halt as number, 0)}</td></tr>
            <tr><td>Antigüedad máx. del dato</td><td>{String(st?.risk.max_data_age_hours)} h</td></tr>
            <tr><td>Costo por operación</td><td>{pct(st?.risk.fee_rate as number)} · desliz. {String(st?.risk.slippage_bps)} pb</td></tr>
            <tr><td>Drawdown actual</td><td>{pct(m?.max_drawdown)} (máx.)</td></tr>
          </tbody></table>
        </div>
      </section>

      <section className="grid-candles">
        <div className="card">
          <h3>Velas diarias MEXC con indicadores {detail ? `· entradas y salidas de la corrida #${detail.metrics.run_id}` : ""}</h3>
          {mkErr && !mk ? <div className="err-box"><span className="neg small">Sin velas de MEXC: {mkErr}</span><button onClick={refreshMarket}>Reintentar</button></div>
            : !mk ? <><div className="skeleton" style={{ height: 260 }} /><p className="muted small">Cargando velas…</p></>
            : <CandleChart candles={mk.candles} trades={trades} fmt={v => usd(v, 0)} />}
        </div>
        <div className="card">
          <h3>Patrimonio vs comprar y mantener</h3>
          <LineChart fmt={v => usd(v, 0)} series={[
            { name: "Agente (Paper)", color: "var(--c2)", points: (detail?.equity ?? []).map(r => ({ x: r.date, y: r.equity })) },
            { name: "Comprar y mantener", color: "var(--c4)", dashed: true, points: bh },
          ]} />
        </div>
      </section>

      <section className="card">
        <h3>Comparación de corridas (mismo periodo y costos)</h3>
        <div className="scroll"><table>
          <thead><tr><th>#</th><th>Modo</th><th>Estrategia</th><th>Tipo</th><th>Periodo</th><th>Ciclos</th><th>Rend. neto</th><th>Comprar y mantener</th><th>Max DD</th><th>Sharpe</th><th>Sortino</th><th>VaR/CVaR 95%</th><th>Stop</th><th>Operaciones (L/S)</th><th>Win Rate</th><th>Rechazadas</th><th>Latencia LLM</th><th>Estado</th></tr></thead>
          <tbody>{runs.map(r => (
            <tr key={r.run_id} className={detail?.metrics.run_id === r.run_id ? "sel" : ""} onClick={() => setSel(r.run_id)}>
              <td>{r.run_id}</td><td>{r.mode}</td><td>{r.strategy === "llm" ? `LLM ${r.llm_model}` : "modelo sin LLM"}</td><td>{r.allow_short ? "Long/Short" : "Solo largo"}</td>
              <td className="small">{r.desde} → {r.hasta}</td><td>{r.ciclos}</td>
              <td className={cls(r.rendimiento_neto)}>{pct(r.rendimiento_neto)}</td>
              <td className={cls(r.buy_and_hold?.rendimiento_neto)}>{pct(r.buy_and_hold?.rendimiento_neto)}</td>
              <td>{pct(r.max_drawdown)}</td><td>{r.sharpe_anual?.toFixed(2) ?? "—"}</td><td>{r.sortino_anual?.toFixed(2) ?? "—"}</td>
              <td>{pct(r.var95_diario)} / {pct(r.cvar95_diario)}</td><td>{r.stop_loss_pct ? pct(r.stop_loss_pct, 0) : "—"}</td><td>{r.operaciones} ({r.operaciones_long}/{r.operaciones_short})</td><td>{pct(r.win_rate, 1)}</td><td>{r.acciones_rechazadas}</td>
              <td>{r.latencia_llm_ms ? `${r.latencia_llm_ms.toFixed(0)} ms` : "—"}</td><td>{r.status}</td>
            </tr>))}</tbody>
        </table></div>
      </section>

      <section className="grid2">
        <div className="card">
          <h3>Operaciones Paper y PnL</h3>
          <div className="scroll tall"><table>
            <thead><tr><th>Fecha ejec.</th><th>Lado</th><th>Estado</th><th>Precio</th><th>Cantidad</th><th>Monto</th><th>Costo</th><th>PnL</th></tr></thead>
            <tbody>{(detail?.orders ?? []).map(o => (
              <tr key={o.id} title={o.reason}>
                <td>{o.exec_date}</td><td className={sideCls(o.side)}>{o.side}</td><td>{o.status}</td>
                <td>{usd(o.exec_price ?? o.ref_price)}</td><td>{o.qty?.toFixed(6) ?? "—"}</td><td>{usd(o.notional)}</td><td>{usd(o.fee)}</td>
                <td className={cls(o.pnl)}>{signed(o.pnl)}</td>
              </tr>))}</tbody>
          </table></div>
          <p className="muted small">PnL realizado: <b className={cls(m?.pnl_realizado)}>{signed(m?.pnl_realizado)}</b> (largos {signed(m?.pnl_long)} · cortos {signed(m?.pnl_short)}) · costos totales {usd(m?.costos_totales)}</p>
        </div>
        <div className="card">
          <h3>Decisiones por vela</h3>
          <div className="scroll tall"><table>
            <thead><tr><th>Vela</th><th>Cierre</th><th>Pronóstico</th><th>Señal</th><th>Propuesta</th><th>Final</th><th>Riesgo</th></tr></thead>
            <tbody>{(detail?.decisions ?? []).map(d => (
              <tr key={d.id} title={`${d.reason} | ${d.risk_reason}`}>
                <td>{d.candle_date}</td><td>{usd(d.close, 0)}</td><td className={cls(d.forecast)}>{(d.forecast * 100).toFixed(3)}%</td>
                <td>{d.model_signal}</td><td>{d.proposed_action}</td><td>{d.final_action}</td><td><span className={`pill r-${d.risk_status}`}>{d.risk_status}</span></td>
              </tr>))}</tbody>
          </table></div>
        </div>
      </section>

      <section className="grid2">
        <div className="card">
          <h3>Bitácora de eventos y errores</h3>
          <div className="scroll tall"><table>
            <thead><tr><th>Hora (UTC)</th><th>Nivel</th><th>Tipo</th><th>Mensaje</th></tr></thead>
            <tbody>{events.map(e => (
              <tr key={e.id} className={`lv-${e.level}`}><td className="small">{e.time.slice(5, 19).replace("T", " ")}</td><td>{e.level}</td><td>{e.kind}</td><td>{e.message}</td></tr>
            ))}</tbody>
          </table></div>
        </div>
        <div className="card">
          <h3>Demostración de errores</h3>
          <p className="muted small">Programa un fallo para el siguiente ciclo. El agente debe mostrar un mensaje útil, registrarlo y no operar indebidamente. La clave incorrecta, el reinicio del backend y el botón Detener se prueban directamente.</p>
          <div className="faults">{FAULTS.map(([k, label]) => (
            <button key={k} className={st?.faults_pending.includes(k) ? "armed" : ""} onClick={() => act(`/faults/${k}`)}>{label}</button>
          ))}</div>
          <h3 style={{ marginTop: 18 }}>Gateway del LLM local</h3>
          <div className="llm">
            <input value={llm.q} onChange={e => setLlm({ ...llm, q: e.target.value })} />
            <button onClick={async () => {
              setLlm({ ...llm, a: "consultando Ollama…" });
              try { const r = await api<any>("/llm/query", { method: "POST", body: { prompt: llm.q } }); setLlm({ ...llm, a: `[${r.model}, ${r.latency_ms.toFixed(0)} ms, ${r.inference}] ${r.answer}` }); }
              catch (x) { setLlm({ ...llm, a: `Error: ${(x as Error).message}` }); }
            }}>Consultar</button>
          </div>
          {llm.a && <p className="small">{llm.a}</p>}
        </div>
      </section>
      <footer className="muted small">Datos: {mk?.source ?? "MEXC"} · Simulación educativa en modo Paper; no es asesoría de inversión. Las corridas replay usan velas históricas y se identifican como tales.</footer>
    </div>
  );
}
