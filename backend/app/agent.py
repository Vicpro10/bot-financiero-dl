"""Ciclo del agente (protocolo, sección 15):

1 velas cerradas MEXC y validación -> 2 variables + pronóstico -> 3 propuesta (Ollama o regla del modelo)
-> 4 validación JSON -> 5 riesgo determinista -> 6 ejecución Paper + persistencia (idempotente) -> 7 UI.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import features as F
from .config import RiskConfig, Settings
from .db import DB, now_iso
from .llm import LLMInvalidResponse, LLMUnavailable, OllamaClient
from .mexc import MarketDataError, MexcClient, SymbolRules
from .model import LoadedModel
from .paper import CLOSE_ACTIONS, TRADE_ACTIONS, Portfolio, allowed_actions, check, close_action, execute

log = logging.getLogger("agent")

FAULTS = {"mexc_down", "stale_data", "ollama_down", "invalid_json", "insufficient_balance", "duplicate_event"}


def signal_to_action(signal: str, p: Portfolio) -> str:
    """Señal objetivo -> una acción por vela. Un giro largo→corto se hace en dos velas (cerrar y luego abrir)."""
    if signal == "LONG":
        return "BUY" if p.qty == 0 else "HOLD" if p.qty > 0 else "COVER"
    if signal == "SHORT":
        return "SHORT" if p.qty == 0 else "HOLD" if p.qty < 0 else "SELL"
    return "SELL" if p.qty > 0 else "COVER" if p.qty < 0 else "HOLD"


class Agent:
    def __init__(self, settings: Settings, risk: RiskConfig, db: DB, model: LoadedModel, mexc: MexcClient, llm: OllamaClient):
        self.s, self.risk, self.db, self.model, self.mexc, self.llm = settings, risk, db, model, mexc, llm
        self.status = "stopped"
        self.status_reason = ""
        self.run_id: int | None = None
        self.task: asyncio.Task | None = None
        self.rules: SymbolRules | None = None
        self.faults: set[str] = set()
        self.last_cycle: dict | None = None
        self._resume = asyncio.Event()
        self._resume.set()
        self.recover_after_restart()

    # ------------------------------------------------------------------ estado
    def recover_after_restart(self):
        """Al reiniciar el backend: las corridas activas quedan en pausa con su estado intacto (sin duplicar)."""
        for r in self.db.q("SELECT id FROM runs WHERE status IN ('running','paused')"):
            self.db.ex("UPDATE runs SET status='paused', updated_at=? WHERE id=?", (now_iso(), r["id"]))
            self.db.event("restart", "Backend reiniciado: corrida recuperada desde SQLite en pausa. Pulse 'Reanudar'.", "WARN", r["id"])
            self.run_id, self.status, self.status_reason = r["id"], "paused", "Recuperada tras reinicio"

    def portfolio(self) -> Portfolio | None:
        r = self.db.one("SELECT * FROM runs WHERE id=?", (self.run_id,)) if self.run_id else None
        if not r:
            return None
        return Portfolio(cash=r["cash"], qty=r["qty"], entry_cost=r["entry_cost"], peak_equity=r["peak_equity"] or r["initial_capital"],
                         day_start_equity=r["day_start_equity"] or r["initial_capital"], day_key=r["day_key"] or "",
                         entry_price=r["entry_price"] or 0.0)

    def _save_portfolio(self, p: Portfolio, last_candle: str | None = None):
        self.db.ex("UPDATE runs SET cash=?, qty=?, entry_cost=?, entry_price=?, peak_equity=?, day_start_equity=?, day_key=?, "
                   "last_candle=COALESCE(?, last_candle), updated_at=? WHERE id=?",
                   (p.cash, p.qty, p.entry_cost, p.entry_price, p.peak_equity, p.day_start_equity, p.day_key, last_candle,
                    now_iso(), self.run_id))

    def _set_status(self, status: str, reason: str = ""):
        self.status, self.status_reason = status, reason
        if self.run_id:
            self.db.ex("UPDATE runs SET status=?, updated_at=? WHERE id=?", (status, now_iso(), self.run_id))
        self.db.notify({"type": "status", "status": status, "reason": reason, "run_id": self.run_id})

    # ------------------------------------------------------------------ control humano
    async def start(self, mode: str, strategy: str, replay_days: int = 60, allow_short: bool = False,
                    stop_loss_pct: float | None = None) -> int:
        if self.task and not self.task.done():
            raise RuntimeError("Ya hay una corrida activa. Deténgala antes de iniciar otra.")
        if mode not in ("replay", "live") or strategy not in ("llm", "model"):
            raise ValueError("mode debe ser replay|live y strategy llm|model")
        self.rules = await self.mexc.symbol_rules(self.s.symbol)
        cap = self.risk.initial_capital
        rs = re = None
        if mode == "replay":
            end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
            rs = (end - pd.Timedelta(days=replay_days)).strftime("%Y-%m-%d")
            re = end.strftime("%Y-%m-%d")
        self.run_id = self.db.ex(
            "INSERT INTO runs(mode,strategy,symbol,model_version,llm_model,initial_capital,fee_rate,replay_start,replay_end,status,"
            "cash,qty,entry_cost,peak_equity,day_start_equity,created_at,updated_at,allow_short,stop_loss_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mode, strategy, self.s.symbol, self.model.version, self.s.ollama_model if strategy == "llm" else None, cap,
             self.risk.fee_rate, rs, re, "running", cap, 0.0, 0.0, cap, cap, now_iso(), now_iso(), int(bool(allow_short)),
             float(self.risk.stop_loss_pct if stop_loss_pct is None else stop_loss_pct)))
        self.db.event("start", f"Corrida {self.run_id} iniciada: modo={mode} estrategia={strategy} {'largo/corto' if allow_short else 'solo largo'}"
                      + (f" replay {rs}→{re}" if rs else ""), run_id=self.run_id,
                      detail={"risk": asdict(self.risk), "model": self.model.version})
        self._launch()
        return self.run_id

    def _launch(self):
        self.status, self.status_reason = "running", ""
        self._resume.set()
        self.task = asyncio.create_task(self._loop())

    def pause(self):
        if self.status == "running":
            self._resume.clear()
            self._set_status("paused", "Pausado por el usuario")
            self.db.event("pause", "Agente en pausa: no se procesan nuevas velas.", run_id=self.run_id)

    async def resume(self):
        if self.status in ("paused", "halted") and self.run_id:
            r = self.db.one("SELECT mode FROM runs WHERE id=?", (self.run_id,))
            if self.rules is None:
                self.rules = await self.mexc.symbol_rules(self.s.symbol)
            self.db.event("resume", "Agente reanudado.", run_id=self.run_id)
            self._set_status("running")
            if self.task and not self.task.done():
                self._resume.set()
            else:
                self._launch()

    async def stop(self):
        """Detener = bloquear nuevas órdenes. NO liquida la posición (eso es la acción explícita 'liquidar')."""
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self._resume.set()
        if self.run_id:
            p = self.portfolio()
            self._set_status("stopped", "Detenido por el usuario")
            self.db.event("stop", f"Agente detenido. Nuevas órdenes bloqueadas. Posición abierta: {p.qty:.6f} {self.s.symbol[:-4]}; "
                                  "use 'Liquidar' para cerrarla.", "WARN", self.run_id)

    async def liquidate(self) -> dict | None:
        if not self.run_id:
            raise RuntimeError("No hay corrida.")
        p = self.portfolio()
        act = close_action(p)
        if act is None:
            return None
        if self.rules is None:
            self.rules = await self.mexc.symbol_rules(self.s.symbol)
        price = (await self.mexc.ticker(self.s.symbol))["last"]
        o = execute(act, p, price, self.risk, self.rules)
        self._record_order(None, o, "Liquidación manual del usuario", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
        self._save_portfolio(p)
        self.db.event("liquidate", f"Posición liquidada manualmente a {o['exec_price']}", run_id=self.run_id)
        return o

    # ------------------------------------------------------------------ bucle
    async def _loop(self):
        r = self.db.one("SELECT * FROM runs WHERE id=?", (self.run_id,))
        try:
            if r["mode"] == "replay":
                await self._replay(r)
            else:
                await self._live(r)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # dependencia caída: pausar nuevas decisiones y registrar
            log.exception("bucle")
            self._set_status("halted", str(e))
            self.db.event("halt", f"Agente pausado por error: {e}", "ERROR", self.run_id)

    async def _fetch(self, start: str) -> pd.DataFrame:
        if "mexc_down" in self.faults:
            self.faults.discard("mexc_down")
            raise MarketDataError("[FALLO INYECTADO] MEXC sin respuesta / 429 tras reintentos con espera creciente.")
        df = await self.mexc.daily_candles(self.s.symbol, start=start)
        df = F.validate_candles(df, min_rows=F.WINDOW + 25)
        self.db.save_candles(self.s.symbol, df)
        return df

    async def _replay(self, r):
        start = (pd.Timestamp(r["replay_start"], tz="UTC") - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        df = await self._fetch(start)
        feats = F.build_features(df)
        dates = feats.index.strftime("%Y-%m-%d")
        for i in range(len(feats) - 1):  # necesita t+1 para ejecutar en la apertura
            d = dates[i]
            if d < r["replay_start"] or d > r["replay_end"]:
                continue
            await self._resume.wait()
            cur = self.db.one("SELECT last_candle FROM runs WHERE id=?", (self.run_id,))
            if cur["last_candle"] and d <= cur["last_candle"]:
                continue  # ya procesada (reanudación tras reinicio)
            await self.cycle(feats, i, exec_price=float(feats["open"].iloc[i + 1]), exec_date=dates[i + 1],
                             value_price=float(feats["close"].iloc[i + 1]), data_age_hours=None)
            await asyncio.sleep(self.s.replay_step_seconds)
        self._set_status("finished", "Replay completado")
        self.db.event("finished", "Replay completado.", run_id=self.run_id)

    async def _live(self, r):
        while True:
            await self._resume.wait()
            start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
            df = await self._fetch(start)
            feats = F.build_features(df)
            i = len(feats) - 1
            d = feats.index[i].strftime("%Y-%m-%d")
            close_ms = int(df["close_time"].iloc[-1])
            age_h = (time.time() * 1000 - close_ms) / 3.6e6
            if "stale_data" in self.faults:
                self.faults.discard("stale_data")
                age_h = 72.0
            cur = self.db.one("SELECT last_candle FROM runs WHERE id=?", (self.run_id,))
            if cur["last_candle"] != d or "duplicate_event" in self.faults:
                tick = await self.mexc.ticker(self.s.symbol)
                await self.cycle(feats, i, exec_price=tick["last"], exec_date=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
                                 value_price=tick["last"], data_age_hours=age_h)
            await asyncio.sleep(self.s.live_poll_seconds)

    # ------------------------------------------------------------------ un ciclo
    async def cycle(self, feats: pd.DataFrame, i: int, exec_price: float, exec_date: str, value_price: float,
                    data_age_hours: float | None) -> dict:
        run = self.db.one("SELECT * FROM runs WHERE id=?", (self.run_id,))
        d = feats.index[i].strftime("%Y-%m-%d")
        # 6 (idempotencia): no procesar dos veces la misma vela
        if "duplicate_event" in self.faults:
            self.faults.discard("duplicate_event")
            d_dup = run["last_candle"] or d
            if self.db.one("SELECT id FROM decisions WHERE run_id=? AND candle_date=?", (self.run_id, d_dup)):
                self.db.event("duplicate", f"Evento duplicado para la vela {d_dup}: ignorado (idempotencia).", "WARN", self.run_id)
                return {"duplicate": d_dup}
        if self.db.one("SELECT id FROM decisions WHERE run_id=? AND candle_date=?", (self.run_id, d)):
            self.db.event("duplicate", f"Vela {d} ya procesada: se ignora.", "WARN", self.run_id)
            return {"duplicate": d}

        p = self.portfolio()
        force_low_cash = False
        allow_short = bool(run["allow_short"])
        if "insufficient_balance" in self.faults and p.qty == 0:
            self.faults.discard("insufficient_balance")
            force_low_cash = True  # demostración: la orden de compra se evalúa con un saldo de 0.50 USDT
        row = feats.iloc[i]
        # 2 variables y pronóstico (mismo contrato que el entrenamiento)
        forecast = self.model.predict_row(feats, i)
        msig = self.model.signal(forecast, float(row["close"]), float(row["sma20"]), allow_short=allow_short)
        allowed = allowed_actions(p, allow_short and self.risk.allow_short)
        eq = p.equity(float(row["close"]))
        ctx = {
            "symbol": self.s.symbol, "candle_date": d, "allowed_actions": allowed,
            "close": round(float(row["close"]), 2), "sma20": round(float(row["sma20"]), 2),
            "dist_sma20": round(float(row["dist_sma20"]), 5), "vol20": round(float(row["vol20"]), 5),
            "rsi14": round(float(row["rsi14"]), 2), "last_5_returns": [round(float(x), 5) for x in feats["ret"].iloc[i - 4 : i + 1]],
            "model_forecast_next_return": round(forecast, 6), "model_signal": msig,
            "model_action": signal_to_action(msig, p),
            "model_rule": self.model.meta["signal_rule"], "position": p.position, "cash": round(p.cash, 2),
            "position_qty": round(p.qty, 6), "entry_price": round(p.entry_price, 2), "equity": round(eq, 2),
            "short_allowed": allow_short,
        }
        # 3-4 propuesta
        latency, raw, source = None, None, "modelo"
        if run["strategy"] == "llm":
            source = "ollama"
            try:
                if "ollama_down" in self.faults:
                    self.faults.discard("ollama_down")
                    raise LLMUnavailable("[FALLO INYECTADO] Ollama detenido (conexión rechazada).")
                if "invalid_json" in self.faults:
                    self.faults.discard("invalid_json")
                    from .llm import validate_proposal
                    validate_proposal("{acción: COMPRAR TODO", self.s.symbol, d, allowed)
                prop = await self.llm.propose(ctx)
                proposed, reason, latency, raw = prop.action, prop.reason, prop.latency_ms, prop.raw
            except LLMInvalidResponse as e:
                self.db.event("invalid_json", f"Propuesta rechazada para {d}: {e}", "ERROR", self.run_id)
                proposed, reason = "HOLD", f"Propuesta del LLM rechazada ({e}); no se opera."
                return self._persist(d, row, forecast, msig, "INVALID", source, reason, "HOLD", "REJECTED", str(e),
                                     latency, raw, p, None, exec_date, value_price)
            except LLMUnavailable as e:
                self.db.event("ollama_down", f"{e}. Se pausan nuevas decisiones.", "ERROR", self.run_id)
                self._set_status("halted", str(e))
                self._resume.clear()
                raise
        else:
            proposed = signal_to_action(msig, p)
            dl = self.model.meta["signal_rule"]["delta"]
            cmp_ = ">" if forecast > dl else "<" if forecast < -dl else "dentro de"
            reason = f"Pronóstico {forecast:+.3%} {cmp_} ±δ={dl:.2%}; señal {msig}."
        # 5a stop-loss por operación (determinista, antes que cualquier propuesta)
        sl = float(run["stop_loss_pct"] or 0)
        if sl > 0 and p.qty != 0 and p.entry_price > 0:
            adverse = (p.entry_price - float(row["close"])) / p.entry_price if p.qty > 0 else (float(row["close"]) - p.entry_price) / p.entry_price
            if adverse >= sl:
                proposed = close_action(p)
                reason = f"STOP-LOSS: movimiento adverso {adverse:.2%} ≥ {sl:.0%} desde la entrada {p.entry_price:,.2f}; se cierra la posición."
                self.db.event("stop_loss", f"{d}: {reason}", "WARN", self.run_id)
        # 5 riesgo determinista
        if force_low_cash:
            proposed, reason = "BUY", "[FALLO INYECTADO] compra con saldo insuficiente"
            low = Portfolio(cash=0.5, peak_equity=0.5, day_start_equity=0.5, day_key=exec_date)
            rr = check(proposed, self.s.symbol, low, exec_price, self.risk, self.rules, data_age_hours, exec_date, allow_short=allow_short)
        else:
            rr = check(proposed, self.s.symbol, p, exec_price, self.risk, self.rules, data_age_hours, exec_date, allow_short=allow_short)
        if rr.status != "APPROVED":
            self.db.event("risk", f"{d}: {proposed} → {rr.status}: {rr.reason}", "WARN", self.run_id)
        # 6 ejecución Paper
        order = None
        if rr.final_action in TRADE_ACTIONS:
            order = execute(rr.final_action, p, exec_price, self.risk, self.rules)
        res = self._persist(d, row, forecast, msig, proposed, source, reason, rr.final_action, rr.status, rr.reason,
                            latency, raw, p, order, exec_date, value_price)
        if rr.halt:
            self._set_status("halted", rr.reason)
            self._resume.clear()
            self.db.event("halt", rr.reason, "ERROR", self.run_id)
        return res

    def _persist(self, d, row, forecast, msig, proposed, source, reason, final, status, rreason, latency, raw, p, order,
                 exec_date, value_price) -> dict:
        did = self.db.ex(
            "INSERT INTO decisions(run_id,candle_date,close,sma20,dist_sma20,vol20,rsi14,forecast,model_signal,proposed_action,"
            "proposal_source,reason,final_action,risk_status,risk_reason,llm_latency_ms,llm_raw,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, d, float(row["close"]), float(row["sma20"]), float(row["dist_sma20"]), float(row["vol20"]),
             float(row["rsi14"]), forecast, msig, proposed, source, reason, final, status, rreason, latency, raw, now_iso()))
        if order:
            self._record_order(did, order, reason, exec_date)
        elif status == "REJECTED" and proposed in TRADE_ACTIONS:
            self.db.ex("INSERT INTO orders(run_id,decision_id,side,status,ref_price,reason,exec_date,created_at) VALUES (?,?,?,?,?,?,?,?)",
                       (self.run_id, did, proposed, "REJECTED", value_price, rreason, exec_date, now_iso()))
        eq = p.equity(value_price)
        p.peak_equity = max(p.peak_equity, eq)
        self.db.ex("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)", (self.run_id, exec_date, value_price, p.cash, p.qty, eq))
        self._save_portfolio(p, last_candle=d)
        self.last_cycle = {"candle_date": d, "forecast": forecast, "model_signal": msig, "proposed": proposed,
                           "final_action": final, "risk_status": status, "reason": reason, "equity": eq}
        self.db.notify({"type": "decision", "run_id": self.run_id, **self.last_cycle,
                        "order": order["side"] if order else None, "position": p.position})
        return self.last_cycle

    def _record_order(self, did, o, reason, exec_date):
        self.db.ex("INSERT INTO orders(run_id,decision_id,side,status,ref_price,exec_price,qty,notional,fee,pnl,reason,exec_date,created_at) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (self.run_id, did, o["side"], o["status"], o["ref_price"], o["exec_price"], o["qty"], o["notional"], o["fee"],
                    o["pnl"], reason, exec_date, now_iso()))


# ---------------------------------------------------------------------- métricas por corrida
def run_metrics(db: DB, run_id: int) -> dict:
    run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        return {}
    eq = db.q("SELECT date, price, equity FROM equity WHERE run_id=? ORDER BY date", (run_id,))
    orders = db.q("SELECT * FROM orders WHERE run_id=?", (run_id,))
    dec = db.q("SELECT llm_latency_ms, risk_status FROM decisions WHERE run_id=?", (run_id,))
    cap = run["initial_capital"]
    sells = [o for o in orders if o["side"] in CLOSE_ACTIONS and o["status"] == "FILLED"]
    longs = [o for o in sells if o["side"] == "SELL"]
    shorts = [o for o in sells if o["side"] == "COVER"]
    out = {"run_id": run_id, "mode": run["mode"], "strategy": run["strategy"], "allow_short": bool(run["allow_short"]),
           "operaciones_long": len(longs), "operaciones_short": len(shorts),
           "pnl_long": float(sum(o["pnl"] or 0 for o in longs)), "pnl_short": float(sum(o["pnl"] or 0 for o in shorts)), "model_version": run["model_version"],
           "llm_model": run["llm_model"], "status": run["status"], "ciclos": len(dec),
           "operaciones": sum(o["status"] == "FILLED" for o in orders),
           "operaciones_cerradas": len(sells),
           "win_rate": (sum((o["pnl"] or 0) > 0 for o in sells) / len(sells)) if sells else None,
           "costos_totales": float(sum(o["fee"] or 0 for o in orders)),
           "acciones_rechazadas": sum(d["risk_status"] in ("REJECTED", "BLOCKED") for d in dec),
           "latencia_llm_ms": float(np.mean([d["llm_latency_ms"] for d in dec if d["llm_latency_ms"]])) if any(d["llm_latency_ms"] for d in dec) else None,
           "pnl_realizado": float(sum(o["pnl"] or 0 for o in sells))}
    out["stop_loss_pct"] = run["stop_loss_pct"]
    if eq:
        e = pd.Series([x["equity"] for x in eq])
        out.update({"patrimonio": float(e.iloc[-1]), "pnl_total": float(e.iloc[-1] - cap), "rendimiento_neto": float(e.iloc[-1] / cap - 1),
                    "max_drawdown": float((1 - e / pd.concat([pd.Series([cap]), e]).cummax().iloc[1:].values).max()),
                    "desde": eq[0]["date"], "hasta": eq[-1]["date"]})
        out.update(risk_metrics(pd.concat([pd.Series([cap]), e]).pct_change().dropna()))
        # comprar y mantener en el mismo periodo y costos: compra al primer precio de ejecución, vende al último
        c = db.q("SELECT date, open, close FROM candles WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
                 (run["symbol"], eq[0]["date"], eq[-1]["date"]))
        if c:
            fee = run["fee_rate"]
            first = c[0]["open"] if run["mode"] == "replay" else eq[0]["price"]
            last = eq[-1]["price"]
            qty = cap / (first * (1 + fee))
            bh_curve = pd.Series([qty * x["close"] for x in c])
            out["buy_and_hold"] = {"rendimiento_neto": float(qty * last * (1 - fee) / cap - 1),
                                   "max_drawdown": float((1 - bh_curve / bh_curve.cummax()).max())}
    return out


def risk_metrics(r: pd.Series, conf: float = 0.95) -> dict:
    """Métricas de los apuntes de clase sobre rendimientos diarios del patrimonio (cripto: 365 días/año, tasa libre de riesgo 0).
    VaR/CVaR históricos a 1 día al 95%, como pérdida positiva (CVaR con peso fraccionario en la cola)."""
    r = r.astype(float)
    if len(r) < 5:
        return {"sharpe_anual": None, "sortino_anual": None, "var95_diario": None, "cvar95_diario": None, "volatilidad_anual": None}
    sd = r.std(ddof=1)
    down = float(np.sqrt(np.minimum(r, 0).pow(2).mean()))
    losses = np.sort(-r.to_numpy())[::-1]
    mass = (1 - conf) * len(r)
    w = np.clip(mass - np.arange(len(r)), 0, 1)
    return {"sharpe_anual": float(r.mean() / sd * np.sqrt(365)) if sd > 0 else None,
            "sortino_anual": float(r.mean() / down * np.sqrt(365)) if down > 0 else None,
            "var95_diario": float(np.quantile(losses, conf, method="inverted_cdf")),
            "cvar95_diario": float(losses @ w / mass) if mass > 0 else None,
            "volatilidad_anual": float(sd * np.sqrt(365))}
