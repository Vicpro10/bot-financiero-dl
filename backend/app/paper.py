"""Motor de riesgo determinista + motor Paper (simulador propio).

No hay ninguna ruta hacia órdenes reales: este módulo solo actualiza un estado en memoria/SQLite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RiskConfig
from .mexc import SymbolRules


@dataclass
class Portfolio:
    cash: float
    qty: float = 0.0
    entry_cost: float = 0.0       # efectivo gastado en la compra abierta (incluye costo)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_key: str = ""

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    @property
    def position(self) -> str:
        return "LONG" if self.qty > 0 else "CASH"


@dataclass
class RiskResult:
    final_action: str     # BUY | SELL | HOLD
    status: str           # APPROVED | REJECTED | BLOCKED
    reason: str
    halt: bool = False


def allowed_actions(p: Portfolio) -> list[str]:
    """Con efectivo: comprar o permanecer en efectivo (HOLD). Con posición: mantenerla (HOLD) o venderla."""
    return ["BUY", "HOLD"] if p.qty <= 0 else ["HOLD", "SELL"]


def floor_step(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def check(action: str, symbol: str, p: Portfolio, price: float, cfg: RiskConfig, rules: SymbolRules,
          data_age_hours: float | None, day_key: str, stopped: bool = False) -> RiskResult:
    if stopped:
        return RiskResult("HOLD", "BLOCKED", "Agente detenido: se bloquean nuevas órdenes.")
    if symbol not in cfg.allowed_symbols:
        return RiskResult("HOLD", "REJECTED", f"Símbolo {symbol} fuera de la lista permitida.")
    if action not in cfg.allowed_actions:
        return RiskResult("HOLD", "REJECTED", f"Acción {action} no permitida.")
    if action not in allowed_actions(p):
        return RiskResult("HOLD", "REJECTED", f"Acción {action} incompatible con la posición {p.position} (sin cortos ni recompras).")
    if data_age_hours is not None and data_age_hours > cfg.max_data_age_hours:
        return RiskResult("HOLD", "BLOCKED", f"Dato atrasado: la última vela tiene {data_age_hours:.1f} h (máx {cfg.max_data_age_hours} h).")

    eq = p.equity(price)
    if day_key != p.day_key:
        p.day_key, p.day_start_equity = day_key, eq
    p.peak_equity = max(p.peak_equity, eq)
    dd = 1 - eq / p.peak_equity if p.peak_equity > 0 else 0.0
    if dd >= cfg.max_drawdown_halt:
        if action == "SELL":
            return RiskResult("SELL", "APPROVED", f"Drawdown {dd:.1%} ≥ límite; se permite reducir riesgo.", halt=True)
        return RiskResult("HOLD", "BLOCKED", f"Drawdown {dd:.1%} ≥ límite {cfg.max_drawdown_halt:.0%}: agente detenido.", halt=True)
    day_loss = 1 - eq / p.day_start_equity if p.day_start_equity > 0 else 0.0
    if action == "BUY" and day_loss >= cfg.max_daily_loss:
        return RiskResult("HOLD", "BLOCKED", f"Pérdida diaria {day_loss:.1%} ≥ límite {cfg.max_daily_loss:.0%}: compras bloqueadas.")

    if action == "BUY":
        budget = min(p.cash, cfg.max_exposure * eq)
        px = price * (1 + cfg.slippage_bps / 1e4)
        qty = floor_step(budget / (px * (1 + cfg.fee_rate)), rules.qty_step)
        if qty <= 0 or qty * px < rules.min_notional:
            return RiskResult("HOLD", "REJECTED", f"Saldo insuficiente: {p.cash:.2f} {rules.quote_asset} no alcanza el mínimo {rules.min_notional}.")
    if action == "SELL" and p.qty <= 0:
        return RiskResult("HOLD", "REJECTED", "No hay posición para vender.")
    return RiskResult(action, "APPROVED", "Dentro de límites.")


def execute(action: str, p: Portfolio, ref_price: float, cfg: RiskConfig, rules: SymbolRules) -> dict | None:
    """Ejecuta en Paper y modifica `p`. Devuelve la orden o None si es HOLD."""
    if action == "BUY":
        px = round(ref_price * (1 + cfg.slippage_bps / 1e4), rules.price_decimals)
        eq = p.equity(ref_price)
        budget = min(p.cash, cfg.max_exposure * eq)
        qty = floor_step(budget / (px * (1 + cfg.fee_rate)), rules.qty_step)
        notional = qty * px
        fee = notional * cfg.fee_rate
        if notional + fee > p.cash + 1e-6:
            raise ValueError("Saldo insuficiente al ejecutar la compra.")
        p.cash -= notional + fee
        p.qty += qty
        p.entry_cost = notional + fee
        return {"side": "BUY", "status": "FILLED", "ref_price": ref_price, "exec_price": px, "qty": qty,
                "notional": notional, "fee": fee, "pnl": None}
    if action == "SELL":
        px = round(ref_price * (1 - cfg.slippage_bps / 1e4), rules.price_decimals)
        qty = p.qty
        notional = qty * px
        fee = notional * cfg.fee_rate
        pnl = notional - fee - p.entry_cost
        p.cash += notional - fee
        p.qty = 0.0
        p.entry_cost = 0.0
        return {"side": "SELL", "status": "FILLED", "ref_price": ref_price, "exec_price": px, "qty": qty,
                "notional": notional, "fee": fee, "pnl": pnl}
    return None
