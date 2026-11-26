"""Modelo PROVISIONAL v0 (CPU) para que el agente funcione antes de tener la exportación de Colab T4.

Sigue el mismo protocolo que el notebook (TR 70% / VA 15% / TE 15% por fecha del objetivo,
escalador ajustado solo con TR, selección con VA, TE una sola vez) pero con scikit-learn en CPU.
NO sustituye la evidencia de entrenamiento en T4: el modelo oficial es el que exporta el notebook.

Uso:  python -m scripts.train_cpu_baseline   (desde backend/)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import backtest as B  # noqa: E402
from app import features as F  # noqa: E402
from app.mexc import MexcClient  # noqa: E402

SYMBOL = "BTCUSDT"
REPO = ROOT.parent
DATA = REPO / "data"
OUT = ROOT / "models" / "current"
REPORTS = REPO / "reports"
DELTAS = [0.0005, 0.001, 0.002, 0.003, 0.005]
FEE = 0.001  # 0.10% por lado (supuesto académico de la práctica)
SEED = 42


def split_by_target_date(df: pd.DataFrame):
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    parts = {"TR": df.iloc[:i_tr], "VA": df.iloc[i_tr:i_va], "TE": df.iloc[i_va:]}
    # comprobación: último objetivo de TR < primero de VA < ... (la fecha del objetivo es t+1)
    assert parts["TR"].index[-1] < parts["VA"].index[0] and parts["VA"].index[-1] < parts["TE"].index[0]
    return parts


def main():
    t0 = time.time()
    raw = asyncio.run(MexcClient().daily_candles(SYMBOL, start="2020-01-01"))
    raw = F.validate_candles(raw, min_rows=1500)
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    raw.to_csv(DATA / "raw" / f"{SYMBOL}_1d_mexc.csv", index=False)

    feats = F.build_features(raw)
    cols_ind = F.feature_columns(True)
    usable = feats.dropna(subset=cols_ind + ["target"])
    usable.to_csv(DATA / "processed" / f"{SYMBOL}_features_v1.csv")
    parts = split_by_target_date(usable)
    print({k: (str(v.index[0].date()), str(v.index[-1].date()), len(v)) for k, v in parts.items()})

    exps = []
    results = {}
    y = {k: v["target"].to_numpy() for k, v in parts.items()}
    # Referencia 1: pronóstico cero
    for k in ("TR", "VA"):
        exps.append({"id": "zero", "conjunto": k, "modelo": "cero", "variables": "-", **B.regression_metrics(y[k], np.zeros_like(y[k]))})

    for use_ind in (False, True):
        cols = F.feature_columns(use_ind)
        tag = "lags+ind" if use_ind else "lags"
        sx = StandardScaler().fit(parts["TR"][cols])  # SOLO TR
        X = {k: sx.transform(v[cols]) for k, v in parts.items()}
        y_mean, y_std = y["TR"].mean(), y["TR"].std()
        ys = (y["TR"] - y_mean) / y_std

        lr = LinearRegression().fit(X["TR"], ys)
        pred = {k: lr.predict(X[k]) * y_std + y_mean for k in X}
        for k in ("TR", "VA"):
            exps.append({"id": f"linreg_{tag}", "conjunto": k, "modelo": "linreg", "variables": tag, **B.regression_metrics(y[k], pred[k])})
        results[("linreg", tag)] = dict(model=lr, sx=sx, cols=cols, y_mean=y_mean, y_std=y_std, pred=pred)

        for hidden in ((16,), (32,), (64, 32)):
            mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-3,
                               learning_rate_init=1e-3, max_iter=300, random_state=SEED, early_stopping=False)
            # parada temprana manual con VA (cronológica): entrenar por épocas y quedarse con el mejor RMSE de VA
            best, best_state, bad = np.inf, None, 0
            for epoch in range(200):
                mlp.partial_fit(X["TR"], ys)
                pv = mlp.predict(X["VA"]) * y_std + y_mean
                rm = float(np.sqrt(((pv - y["VA"]) ** 2).mean()))
                if rm < best - 1e-7:
                    best, bad = rm, 0
                    best_state = ([w.copy() for w in mlp.coefs_], [b.copy() for b in mlp.intercepts_], epoch)
                else:
                    bad += 1
                    if bad >= 15:
                        break
            mlp.coefs_, mlp.intercepts_ = best_state[0], best_state[1]
            pred = {k: mlp.predict(X[k]) * y_std + y_mean for k in X}
            name = f"mlp{'x'.join(map(str, hidden))}_{tag}"
            for k in ("TR", "VA"):
                exps.append({"id": name, "conjunto": k, "modelo": "mlp_cpu", "variables": tag, "epoca_mejor": best_state[2],
                             **B.regression_metrics(y[k], pred[k])})
            results[("mlp", tag, hidden)] = dict(model=mlp, sx=sx, cols=cols, y_mean=y_mean, y_std=y_std, pred=pred, hidden=hidden)

    # Selección del modelo con RMSE de VA (incluye la regresión: no es obligatorio elegir el más complejo)
    def va_rmse(r):
        return float(np.sqrt(((r["pred"]["VA"] - y["VA"]) ** 2).mean()))
    key = min(results, key=lambda k: va_rmse(results[k]))
    best = results[key]
    print("Modelo elegido con VA:", key, "RMSE VA", va_rmse(best))

    # Selección de la regla de señal con VA: δ y filtro SMA20; criterio rendimiento neto VA, desempate menor drawdown
    va = parts["VA"]
    rule_rows = []
    for sma_filter in (False, True):
        for d in DELTAS:
            sig = pd.Series(np.where((best["pred"]["VA"] > d) & ((va["close"] > va["sma20"]) | (not sma_filter)), "LONG", "CASH"), index=va.index)
            _, m = B.simulate(va, sig, 100_000, FEE)
            rule_rows.append({"delta": d, "sma_filter": sma_filter, **m})
    rules = pd.DataFrame(rule_rows).sort_values(["rendimiento_neto", "max_drawdown"], ascending=[False, True])
    chosen = rules.iloc[0]
    print(rules[["delta", "sma_filter", "rendimiento_neto", "max_drawdown", "operaciones_cerradas"]].to_string())

    # Congelar y evaluar TE una sola vez
    te = parts["TE"]
    pte = best["pred"]["TE"]
    exps.append({"id": "final_" + "_".join(map(str, key)), "conjunto": "TE", "modelo": key[0], "variables": key[1], **B.regression_metrics(y["TE"], pte)})
    sig_te = pd.Series(np.where((pte > chosen.delta) & ((te["close"] > te["sma20"]) | (not chosen.sma_filter)), "LONG", "CASH"), index=te.index)
    sim, m_te = B.simulate(te, sig_te, 100_000, FEE)
    _, m_bh = B.buy_and_hold(te, te.index, 100_000, FEE)
    print("TE estrategia:", m_te)
    print("TE comprar y mantener:", m_bh)

    # Exportar con el formato que también genera el notebook
    OUT.mkdir(parents=True, exist_ok=True)
    w = {}
    if key[0] == "linreg":
        w["dense0_W"] = best["model"].coef_.reshape(-1, 1)
        w["dense0_b"] = np.array([best["model"].intercept_])
        n_dense = 1
    else:
        for i, (W, b) in enumerate(zip(best["model"].coefs_, best["model"].intercepts_)):
            w[f"dense{i}_W"], w[f"dense{i}_b"] = W, b
        n_dense = len(best["model"].coefs_)
    check_x = te[best["cols"]].to_numpy()[:8]
    w["check_x"] = check_x
    w["check_y"] = pte[:8]
    np.savez(OUT / "model.npz", **w)
    meta = {
        "version": "v0-cpu-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
        "kind": "mlp" if key[0] != "linreg" else "linear",
        "n_dense": n_dense,
        "feature_contract": F.FEATURE_CONTRACT_VERSION,
        "features": best["cols"],
        "x_mean": best["sx"].mean_.tolist(),
        "x_scale": best["sx"].scale_.tolist(),
        "y_mean": float(best["y_mean"]),
        "y_scale": float(best["y_std"]),
        "signal_rule": {"delta": float(chosen.delta), "sma_filter": bool(chosen.sma_filter)},
        "symbol": SYMBOL,
        "source": "MEXC spot v3 /api/v3/klines 1d (UTC)",
        "trained_on": "CPU local (scikit-learn) - PROVISIONAL, no es la evidencia T4",
        "splits": {k: {"inicio": str(v.index[0].date()), "fin": str(v.index[-1].date()), "n": len(v)} for k, v in parts.items()},
        "te_metrics": {"modelo": exps[-1], "estrategia": m_te, "buy_and_hold": m_bh},
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_seconds": round(time.time() - t0, 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(exps).to_csv(REPORTS / "experimentos_cpu_v0.csv", index=False)
    rules.to_csv(REPORTS / "umbrales_VA_cpu_v0.csv", index=False)
    sim.to_csv(REPORTS / "simulacion_TE_cpu_v0.csv")
    print("Exportado a", OUT)


if __name__ == "__main__":
    main()
