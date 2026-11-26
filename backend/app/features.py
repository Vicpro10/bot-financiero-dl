"""Variables del modelo (contrato de variables v1).

Las MISMAS definiciones se usan en el notebook de Colab (capítulos 7-11) y en el
backend del agente. Cualquier cambio aquí obliga a re-entrenar y a subir la
versión del contrato (FEATURE_CONTRACT_VERSION).

Todas las variables de la fila t usan solo información disponible al CIERRE de t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_CONTRACT_VERSION = "v1"
WINDOW = 20  # 20 rendimientos más recientes (protocolo, sección 3)

LAG_COLS = [f"ret_lag{k}" for k in range(WINDOW)]  # ret_lag0 = r_t = P_t/P_{t-1}-1
IND_COLS = ["dist_sma20", "vol20"]
SEQ_VARS = ["ret", "dist_sma20", "vol20"]  # variables por paso para la LSTM


class DataValidationError(ValueError):
    """Datos inválidos: el flujo se detiene con un mensaje que explica qué corregir."""


def validate_candles(df: pd.DataFrame, min_rows: int = WINDOW + 2) -> pd.DataFrame:
    """Audita velas diarias. No rellena huecos ni inventa volumen."""
    if df is None or len(df) == 0:
        raise DataValidationError("Conjunto vacío: la fuente no devolvió velas. Revise símbolo, intervalo y conexión.")
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas {sorted(missing)}; se esperaban {sorted(required)}.")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    if out["date"].duplicated().any():
        dups = out.loc[out["date"].duplicated(keep=False), "date"].dt.date.astype(str).unique().tolist()
        raise DataValidationError(f"Fechas duplicadas: {dups[:5]}. Elimine duplicados con una regla documentada.")
    if not out["date"].is_monotonic_increasing:
        raise DataValidationError("Fechas desordenadas: ordene por fecha antes de calcular variables.")
    num = out[["open", "high", "low", "close", "volume"]].astype(float)
    if not np.isfinite(num.to_numpy()).all():
        bad = out.loc[~np.isfinite(num).all(axis=1), "date"].dt.date.astype(str).tolist()
        raise DataValidationError(f"Valores no finitos en fechas {bad[:5]}.")
    if (num[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("Precios no positivos detectados; revise la fuente.")
    if len(out) < min_rows:
        raise DataValidationError(f"Ventana insuficiente: se recibieron {len(out)} velas y se requieren al menos {min_rows}.")
    out[["open", "high", "low", "close", "volume"]] = num
    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame indexado por fecha con variables, objetivo y precio.

    - ret_lag{k}: rendimiento simple de t-k (k=0..19).
    - dist_sma20: cierre_t / SMA20_t - 1 (SMA de 20 cierres hasta t).
    - vol20: desviación estándar muestral (ddof=1) de los últimos 20 rendimientos, sin anualizar.
    - rsi14: RSI de Wilder (informativo para la UI y el LLM; no entra al modelo v1).
    - target: y_t = P_{t+1}/P_t - 1 (NaN en la última fila: aún no observado).
    No se rellenan hacia atrás los valores iniciales: las filas sin indicador válido se descartan después.
    """
    d = df.set_index("date")
    c = d["close"]
    out = pd.DataFrame(index=d.index)
    out["open"], out["high"], out["low"], out["close"] = d["open"], d["high"], d["low"], c
    ret = c.pct_change(fill_method=None)
    out["ret"] = ret
    for k in range(WINDOW):
        out[f"ret_lag{k}"] = ret.shift(k)
    sma20 = c.rolling(20).mean()
    out["sma20"] = sma20
    out["dist_sma20"] = c / sma20 - 1
    out["vol20"] = ret.rolling(20).std(ddof=1)
    out["rsi14"] = rsi_wilder(c, 14)
    out["target"] = c.shift(-1) / c - 1
    return out


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder. Sin pérdidas -> 100; sin ganancias -> 0; ambas cero -> 50 (convención neutral)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    g = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    l = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    total = g + l
    rsi = 100 * g / total.replace(0, np.nan)
    return rsi.mask(total.eq(0) & g.notna(), 50.0)


def feature_columns(use_indicators: bool) -> list[str]:
    return LAG_COLS + (IND_COLS if use_indicators else [])


def mlp_vector(feats: pd.DataFrame, use_indicators: bool) -> np.ndarray:
    cols = feature_columns(use_indicators)
    return feats[cols].to_numpy(dtype=float)


def lstm_tensor(feats: pd.DataFrame, rows: np.ndarray, seq_vars: list[str]) -> np.ndarray:
    """Forma (ejemplos, 20 pasos, variables). El último paso es la fila t."""
    arr = feats[seq_vars].to_numpy(dtype=float)
    out = np.stack([arr[i - WINDOW + 1 : i + 1] for i in rows])
    if out.shape[1:] != (WINDOW, len(seq_vars)):
        raise DataValidationError(f"Forma incorrecta: esperada (n, {WINDOW}, {len(seq_vars)}), recibida {out.shape}.")
    return out
