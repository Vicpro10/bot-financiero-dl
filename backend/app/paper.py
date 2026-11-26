"""Motor de riesgo determinista + motor Paper (simulador propio).

No hay ninguna ruta hacia órdenes reales: este módulo solo actualiza un estado en memoria/SQLite.

Posiciones (v1.1):
- LONG  (qty > 0): se abre con BUY  y se cierra con SELL.
- SHORT (qty < 0): se abre con SHORT y se cierra con COVER (solo si la corrida lo permite).
- CASH  (qty = 0).
Sin apalancamiento: la exposición absoluta nunca supera max_exposure × patrimonio, y un corto queda
100% respaldado por el efectivo (el producto de la venta corta se suma al efectivo y no se puede reinvertir).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RiskConfig
from .mexc import SymbolRules

OPEN_ACTIONS = {"BUY", "SHORT"}
CLOSE_ACTIONS = {"SELL", "COVER"}
TRADE_ACTIONS = OPEN_ACTIONS | CLOSE_ACTIONS


@dataclass
class Portfolio:
    cash: float
    qty: float = 0.0
    entry_cost: float = 0.0       # LONG: efectivo pagado (con costo). SHORT: efectivo neto recibido (sin costo)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_key: str = ""
    entry_price: float = 0.0      # precio de entrada de la posición abierta

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price   # con qty < 0 el corto resta su valor de recompra

    @property
    def position(self) -> str:
        return "LONG" if self.qty > 0 else "SHORT" if self.qty < 0 else "CASH"


@dataclass
class RiskResult:
    final_action: str     # BUY | SELL | SHORT | COVER | HOLD
    status: str           # APPROVED | REJECTED | BLOCKED
    reason: str
    halt: bool = False


def allowed_actions(p: Portfolio, allow_short: bool = False) -> list[str]:
    """Efectivo: comprar, abrir corto (si se permite) o seguir en efectivo (HOLD).
    Largo: mantener (HOLD) o vender. Corto: mantener (HOLD) o recomprar (COVER)."""
    if p.qty > 0:
        return ["HOLD", "SELL"]
    if p.qty < 0:
        return ["HOLD", "COVER"]
    return ["BUY", "SHORT", "HOLD"] if allow_short else ["BUY", "HOLD"]


def floor_step(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def _open_qty(p: Portfolio, price: float, cfg: RiskConfig, rules: SymbolRules, side: str) -> tuple[float, float]:
    eq = p.equity(price)
    budget = min(p.cash, cfg.max_exposure * eq)
    px = price * (1 + cfg.slippage_bps / 1e4) if side == "BUY" else price * (1 - cfg.slippage_bps / 1e4)
    qty = floor_step(budget / (px * (1 + cfg.fee_rate)), rules.qty_step)
    return qty, px


def check(action: str, symbol: str, p: Portfolio, price: float, cfg: RiskConfig, rules: SymbolRules,
          data_age_hours: float | None, day_key: str, stopped: bool = False, allow_short: bool = False) -> RiskResult:
    if stopped:
        return RiskResult("HOLD", "BLOCKED", "Agente detenido: se bloquean nuevas órdenes.")
    if symbol not in cfg.allowed_symbols:
        return RiskResult("HOLD", "REJECTED", f"Símbolo {symbol} fuera de la lista permitida.")
    if action not in cfg.allowed_actions:
        return RiskResult("HOLD", "REJECTED", f"Acción {action} no permitida.")
    if action == "SHORT" and not (allow_short and cfg.allow_short):
        return RiskResult("HOLD", "REJECTED", "Ventas en corto deshabilitadas en esta corrida (modo solo largo).")
    if action not in allowed_actions(p, allow_short and cfg.allow_short):
        return RiskResult("HOLD", "REJECTED", f"Acción {action} incompatible con la posición {p.position} (sin recompras ni giros directos).")
    if data_age_hours is not None and data_age_hours > cfg.max_data_age_hours:
        return RiskResult("HOLD", "BLOCKED", f"Dato atrasado: la última vela tiene {data_age_hours:.1f} h (máx {cfg.max_data_age_hours} h).")

    eq = p.equity(price)
    if day_key != p.day_key:
        p.day_key, p.day_start_equity = day_key, eq
    p.peak_equity = max(p.peak_equity, eq)
    dd = 1 - eq / p.peak_equity if p.peak_equity > 0 else 0.0
    if dd >= cfg.max_drawdown_halt:
        if action in CLOSE_ACTIONS:
            return RiskResult(action, "APPROVED", f"Drawdown {dd:.1%} ≥ límite; se permite cerrar la posición.", halt=True)
        return RiskResult("HOLD", "BLOCKED", f"Drawdown {dd:.1%} ≥ límite {cfg.max_drawdown_halt:.0%}: agente detenido.", halt=True)
    day_loss = 1 - eq / p.day_start_equity if p.day_start_equity > 0 else 0.0
    if action in OPEN_ACTIONS and day_loss >= cfg.max_daily_loss:
        return RiskResult("HOLD", "BLOCKED", f"Pérdida diaria {day_loss:.1%} ≥ límite {cfg.max_daily_loss:.0%}: nuevas entradas bloqueadas.")

    if action in OPEN_ACTIONS:
        qty, px = _open_qty(p, price, cfg, rules, action)
        if qty <= 0 or qty * px < rules.min_notional:
            return RiskResult("HOLD", "REJECTED", f"Saldo insuficiente: {p.cash:.2f} {rules.quote_asset} no alcanza el mínimo {rules.min_notional}.")
    return RiskResult(action, "APPROVED", "Dentro de límites.")


def execute(action: str, p: Portfolio, ref_price: float, cfg: RiskConfig, rules: SymbolRules) -> dict | None:
    """Ejecuta en Paper y modifica `p`. Devuelve la orden o None si es HOLD."""
    slip = cfg.slippage_bps / 1e4
    if action in OPEN_ACTIONS:
        if p.qty != 0:
            raise ValueError("Ya hay una posición abierta.")
        qty, _ = _open_qty(p, ref_price, cfg, rules, action)
        px = round(ref_price * (1 + slip) if action == "BUY" else ref_price * (1 - slip), rules.price_decimals)
        notional = qty * px
        fee = notional * cfg.fee_rate
        if action == "BUY":
            if notional + fee > p.cash + 1e-6:
                raise ValueError("Saldo insuficiente al ejecutar la compra.")
            p.cash -= notional + fee
            p.qty = qty
            p.entry_cost = notional + fee
        else:  # SHORT: se vende BTC prestado; el producto queda como efectivo de respaldo
            p.cash += notional - fee
            p.qty = -qty
            p.entry_cost = notional - fee
        p.entry_price = px
        return {"side": action, "status": "FILLED", "ref_price": ref_price, "exec_price": px, "qty": qty,
                "notional": notional, "fee": fee, "pnl": None}
    if action == "SELL":
        px = round(ref_price * (1 - slip), rules.price_decimals)
        qty = p.qty
        notional = qty * px
        fee = notional * cfg.fee_rate
        pnl = notional - fee - p.entry_cost
        p.cash += notional - fee
        p.qty, p.entry_cost, p.entry_price = 0.0, 0.0, 0.0
        return {"side": "SELL", "status": "FILLED", "ref_price": ref_price, "exec_price": px, "qty": qty,
                "notional": notional, "fee": fee, "pnl": pnl}
    if action == "COVER":
        px = round(ref_price * (1 + slip), rules.price_decimals)
        qty = -p.qty
        notional = qty * px
        fee = notional * cfg.fee_rate
        pnl = p.entry_cost - (notional + fee)
        p.cash -= notional + fee
        p.qty, p.entry_cost, p.entry_price = 0.0, 0.0, 0.0
        return {"side": "COVER", "status": "FILLED", "ref_price": ref_price, "exec_price": px, "qty": qty,
                "notional": notional, "fee": fee, "pnl": pnl}
    return None


def close_action(p: Portfolio) -> str | None:
    return "SELL" if p.qty > 0 else "COVER" if p.qty < 0 else None
