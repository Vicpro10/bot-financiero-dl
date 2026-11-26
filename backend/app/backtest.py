"""Simulación histórica vectorial (protocolo, sección 9).

Señal emitida al cierre de t -> ejecución en la APERTURA de t+1 -> patrimonio valorado
al cierre de cada día. Solo largo o efectivo, 100% del capital, costo por lado.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def simulate(prices: pd.DataFrame, signals: pd.Series, capital: float = 100_000.0, fee: float = 0.001,
             liquidate_end: bool = True) -> tuple[pd.DataFrame, dict]:
    """prices: índice fecha, columnas open/close. signals: 'LONG'/'CASH' emitidas al cierre de cada fecha.

    Devuelve (tabla diaria, métricas). La primera ejecución posible es la apertura del día siguiente
    a la primera señal; la última señal no se ejecuta si no hay día siguiente.
    """
    idx = signals.index
    p = prices.loc[idx]
    cash, qty = capital, 0.0
    rows, trades = [], []
    entry_cost = 0.0
    total_fees = 0.0
    for i in range(1, len(idx)):
        sig = signals.iloc[i - 1]
        px_open = float(p["open"].iloc[i])
        action = "HOLD"
        if sig == "LONG" and qty == 0 and cash > 0:
            qty = cash / (px_open * (1 + fee))
            f = qty * px_open * fee
            entry_cost = cash
            total_fees += f
            cash = 0.0
            action = "BUY"
        elif sig == "CASH" and qty > 0:
            gross = qty * px_open
            f = gross * fee
            cash = gross - f
            total_fees += f
            trades.append(cash - entry_cost)
            qty = 0.0
            action = "SELL"
        close = float(p["close"].iloc[i])
        rows.append({"date": idx[i], "signal_prev_close": sig, "action": action, "exec_price": px_open if action != "HOLD" else np.nan,
                     "position_qty": qty, "cash": cash, "close": close, "equity": cash + qty * close, "fees_cum": total_fees})
    if not rows:
        return pd.DataFrame(), {"error": "sin periodos para simular"}
    tab = pd.DataFrame(rows).set_index("date")
    if liquidate_end and qty > 0:
        gross = qty * tab["close"].iloc[-1]
        f = gross * fee
        total_fees += f
        trades.append(gross - f - entry_cost)
        tab.iloc[-1, tab.columns.get_loc("equity")] = gross - f
        tab.iloc[-1, tab.columns.get_loc("fees_cum")] = total_fees
    return tab, metrics(tab["equity"], capital, trades, total_fees)


def buy_and_hold(prices: pd.DataFrame, idx: pd.Index, capital: float = 100_000.0, fee: float = 0.001):
    sig = pd.Series("LONG", index=idx)
    return simulate(prices, sig, capital, fee)


def max_drawdown(equity: pd.Series) -> float:
    return float((1 - equity / equity.cummax()).max()) if len(equity) else float("nan")


def metrics(equity: pd.Series, capital: float, trades: list[float], fees: float) -> dict:
    r = equity.pct_change().dropna()
    sd = r.std(ddof=1)
    return {
        "capital_inicial": capital,
        "patrimonio_final": float(equity.iloc[-1]),
        "rendimiento_neto": float(equity.iloc[-1] / capital - 1),
        "max_drawdown": max_drawdown(equity),
        "operaciones_cerradas": len(trades),
        "win_rate": float(sum(t > 0 for t in trades) / len(trades)) if trades else None,  # None = no definido
        "costos_totales": float(fees),
        "sharpe_anual": float(r.mean() / sd * math.sqrt(365)) if sd and sd > 0 else None,  # cripto: 365 días
        "dias": int(len(equity)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """error = estimado - real. Precisión direccional: acierto si sign(pred)==sign(real); ceros reales se excluyen."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    mask = y_true != 0
    da = float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()) if mask.any() else None
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err**2).mean())), "DA": da, "n": int(len(y_true))}
