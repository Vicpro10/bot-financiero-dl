"""Evaluación Long/Short del modelo de la T4 en el periodo TE (análisis adicional, v1.1).

IMPORTANTE (protocolo, sección 3): TE ya se consultó para la evaluación oficial (solo largo). Este análisis
agrega una regla simétrica decidida por la rúbrica del proyecto final (criterio 5: Entry, Long, Short), NO por
los resultados de TE: mismo modelo, mismo δ elegido en VA y ningún parámetro nuevo. Aun así se declara como
análisis posterior y no como evaluación intacta.

Uso (desde backend/):  python -m scripts.eval_long_short_te
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import features as F  # noqa: E402
from app.model import load_model  # noqa: E402

REPO = ROOT.parent
T4 = REPO / "reports" / "colab_t4"


def simulate_ls(px: pd.DataFrame, sig: pd.Series, capital=100_000.0, fee=0.001):
    """Señal al cierre de t -> ejecución en la apertura de t+1. LONG/SHORT/CASH, una acción por día, sin apalancamiento."""
    cash, qty, entry, fees, trades, eq = capital, 0.0, 0.0, 0.0, [], []
    for i in range(1, len(sig)):
        target, o, c = sig.iloc[i - 1], px["open"].iloc[i], px["close"].iloc[i]
        if qty > 0 and target != "LONG":                      # cerrar largo
            g = qty * o; f = g * fee; cash += g - f; fees += f; trades.append(("L", g - f - entry)); qty = 0.0
        elif qty < 0 and target != "SHORT":                   # cerrar corto
            g = -qty * o; f = g * fee; cash -= g + f; fees += f; trades.append(("S", entry - (g + f))); qty = 0.0
        elif qty == 0 and target == "LONG":
            q = cash / (o * (1 + fee)); f = q * o * fee; entry = cash; cash, qty, fees = 0.0, q, fees + f
        elif qty == 0 and target == "SHORT":
            q = cash / (o * (1 + fee)); g = q * o; f = g * fee; cash += g - f; entry = g - f; qty, fees = -q, fees + f
        eq.append(cash + qty * c)
    if qty != 0:   # liquidación final al último cierre
        c = px["close"].iloc[-1]; g = abs(qty) * c; f = g * fee; fees += f
        if qty > 0:
            trades.append(("L", g - f - entry)); cash += g - f
        else:
            trades.append(("S", entry - (g + f))); cash -= g + f
        eq[-1] = cash
    e = pd.Series(eq)
    pnl = [t[1] for t in trades]
    return {"rend_neto": float(e.iloc[-1] / capital - 1), "max_dd": float((1 - e / e.cummax()).max()), "operaciones_cerradas": len(trades),
            "largos": sum(t[0] == "L" for t in trades), "cortos": sum(t[0] == "S" for t in trades),
            "pnl_largos": float(sum(t[1] for t in trades if t[0] == "L")), "pnl_cortos": float(sum(t[1] for t in trades if t[0] == "S")),
            "win_rate": float(np.mean([x > 0 for x in pnl])) if pnl else None, "costos": float(fees)}


def main():
    m = load_model(ROOT / "models" / "current")
    raw = pd.read_csv(REPO / "data" / "raw" / "BTCUSDT_1d_mexc.csv") if (REPO / "data" / "raw" / "BTCUSDT_1d_mexc.csv").exists() else None
    feats = F.build_features(F.validate_candles(raw[["date", "open", "high", "low", "close", "volume"]]))
    sp = m.meta["splits"]["TE"]
    te_idx = feats.loc[sp["inicio"]:sp["fin"]].index
    rows = [feats.index.get_loc(d) for d in te_idx]
    preds = pd.Series([m.predict_row(feats, r) for r in rows], index=te_idx)
    te = feats.loc[te_idx]
    d = m.meta["signal_rule"]["delta"]
    sig_l = pd.Series(np.where(preds > d, "LONG", "CASH"), index=te_idx)
    sig_ls = pd.Series(np.where(preds > d, "LONG", np.where(preds < -d, "SHORT", "CASH")), index=te_idx)
    out = {}
    for fee in (0.001, 0.0):
        out[f"solo_largo_fee{fee}"] = simulate_ls(te, sig_l, fee=fee)
        out[f"long_short_fee{fee}"] = simulate_ls(te, sig_ls, fee=fee)
    bh_q = 100_000 / (te["open"].iloc[1] * 1.001)
    out["comprar_y_mantener"] = {"rend_neto": float(bh_q * te["close"].iloc[-1] * 0.999 / 100_000 - 1)}
    res = {"modelo": m.version, "periodo_TE": [sp["inicio"], sp["fin"]], "delta": d, "dias_senal": {
        "LONG": int((sig_ls == "LONG").sum()), "SHORT": int((sig_ls == "SHORT").sum()), "CASH": int((sig_ls == "CASH").sum())},
        "resultados": out, "nota": "Análisis posterior a la evaluación oficial de TE; regla simétrica sin parámetros nuevos."}
    (REPO / "reports" / "long_short_TE.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
