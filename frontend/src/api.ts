// Cliente de la API. La clave la escribe la persona al entrar y vive solo en sessionStorage
// (no se compila en el bundle ni se usa ninguna variable VITE_* para secretos).

const KEY = "agent_api_key";

export function getKey(): string {
  try { return sessionStorage.getItem(KEY) ?? ""; } catch { return ""; }
}
export function setKey(k: string) {
  try { sessionStorage.setItem(KEY, k); } catch { /* sin almacenamiento: la clave dura lo que la pestaña */ }
}
export function clearKey() {
  try { sessionStorage.removeItem(KEY); } catch { /* nada */ }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const r = await fetch(`/api${path}`, {
    method: opts.method ?? "GET",
    headers: { "X-API-Key": getKey(), ...(opts.body ? { "Content-Type": "application/json" } : {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch { /* cuerpo no JSON */ }
    throw new ApiError(r.status, typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

export interface Candle { date: string; open: number; high: number; low: number; close: number; sma20: number | null; bb_up: number | null; bb_dn: number | null; rsi14: number | null }
export interface Market {
  ticker: { symbol: string; last: number; change_pct: number; high: number; low: number; time: number };
  source: string;
  last_closed_candle: string;
  indicators: { close: number; sma20: number; dist_sma20: number; vol20: number; rsi14: number };
  forecast: { cutoff: string; next_return: number; signal: string; signal_long_short: string };
  candles: Candle[];
}
export interface RunMetrics {
  run_id: number; mode: string; strategy: string; allow_short: boolean; operaciones_long: number; operaciones_short: number;
  pnl_long: number; pnl_short: number; sharpe_anual: number | null; sortino_anual: number | null;
  var95_diario: number | null; cvar95_diario: number | null; volatilidad_anual: number | null; stop_loss_pct: number | null; model_version: string; llm_model: string | null; status: string;
  ciclos: number; operaciones: number; operaciones_cerradas: number; win_rate: number | null; costos_totales: number;
  acciones_rechazadas: number; latencia_llm_ms: number | null; pnl_realizado: number; patrimonio?: number; pnl_total?: number;
  rendimiento_neto?: number; max_drawdown?: number; desde?: string; hasta?: string;
  buy_and_hold?: { rendimiento_neto: number; max_drawdown: number };
}
export interface Order {
  id: number; side: string; status: string; ref_price: number; exec_price: number | null; qty: number | null;
  notional: number | null; fee: number | null; pnl: number | null; reason: string; exec_date: string;
}
export interface Decision {
  id: number; candle_date: string; close: number; sma20: number; rsi14: number; vol20: number; forecast: number;
  model_signal: string; proposed_action: string; proposal_source: string; reason: string; final_action: string;
  risk_status: string; risk_reason: string; llm_latency_ms: number | null;
}
export interface EquityRow { date: string; price: number; cash: number; qty: number; equity: number }
export interface RunDetail { metrics: RunMetrics; equity: EquityRow[]; decisions: Decision[]; orders: Order[] }
export interface EventRow { id: number; run_id: number | null; time: string; level: string; kind: string; message: string }
export interface Status {
  status: string; status_reason: string; symbol: string; real_orders: string; faults_pending: string[];
  run: any; portfolio: { cash: number; qty: number; entry_cost: number; peak_equity: number; entry_price: number } | null;
  last_decision: Decision | null;
  model: { version: string; kind: string; signal_rule: { delta: number; sma_filter: boolean }; trained_on: string;
           features: string[]; splits: Record<string, { inicio: string; fin: string; n: number }> };
  risk: Record<string, unknown>;
}
