"""Carga del modelo versionado y su escalador; inferencia en NumPy.

El notebook de Colab (T4) exporta los pesos de Keras a `model.npz` y el contrato a
`meta.json`. El backend no necesita TensorFlow: reproduce la propagación hacia
adelante de Dense/LSTM con NumPy y verifica, al cargar, que reproduce las
predicciones de referencia guardadas por el notebook (`check_x`, `check_y`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F


class ModelLoadError(RuntimeError):
    pass


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LoadedModel:
    meta: dict
    weights: dict

    @property
    def kind(self) -> str:
        return self.meta["kind"]  # "mlp" | "lstm" | "linear"

    @property
    def version(self) -> str:
        return self.meta["version"]

    # ---------- entradas ----------
    def make_input(self, feats: pd.DataFrame, row: int) -> np.ndarray:
        if self.kind == "lstm":
            x = F.lstm_tensor(feats, np.array([row]), self.meta["seq_vars"])
        else:
            x = feats[self.meta["features"]].to_numpy(dtype=float)[row : row + 1]
        if not np.isfinite(x).all():
            raise F.DataValidationError(f"Entradas no finitas en la fila {feats.index[row].date()}: faltan datos para la ventana.")
        return x

    # ---------- inferencia ----------
    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.meta["x_mean"])
        scale = np.asarray(self.meta["x_scale"])
        xs = (x - mean) / scale
        if self.kind == "lstm":
            h = self._lstm(xs)
            out = self._dense_stack(h, prefix="head")
        else:
            out = self._dense_stack(xs, prefix="dense")
        y = out.reshape(-1) * self.meta["y_scale"] + self.meta["y_mean"]
        if not np.isfinite(y).all():
            raise ModelLoadError("Predicción no finita: revise escalas y pesos del modelo.")
        return y

    def _dense_stack(self, x, prefix):
        n = self.meta[f"n_{prefix}"]
        for i in range(n):
            x = x @ self.weights[f"{prefix}{i}_W"] + self.weights[f"{prefix}{i}_b"]
            if i < n - 1:
                x = np.maximum(x, 0.0)  # ReLU en ocultas, lineal en la salida
        return x

    def _lstm(self, xs):
        # Keras: kernel (in, 4u), recurrent (u, 4u), bias (4u); compuertas i, f, c, o
        K, R, b = self.weights["lstm_kernel"], self.weights["lstm_recurrent"], self.weights["lstm_bias"]
        u = R.shape[0]
        n = xs.shape[0]
        h = np.zeros((n, u))
        c = np.zeros((n, u))
        for t in range(xs.shape[1]):
            z = xs[:, t, :] @ K + h @ R + b
            i, f, g, o = _sigmoid(z[:, :u]), _sigmoid(z[:, u : 2 * u]), np.tanh(z[:, 2 * u : 3 * u]), _sigmoid(z[:, 3 * u :])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h

    def predict_row(self, feats: pd.DataFrame, row: int) -> float:
        return float(self.predict_raw(self.make_input(feats, row))[0])

    def signal(self, forecast: float, close: float, sma20: float) -> str:
        """Regla seleccionada en VA: comprar/mantener si pronóstico > δ (y filtro SMA20 si está activo)."""
        rule = self.meta["signal_rule"]
        ok = forecast > rule["delta"]
        if rule.get("sma_filter"):
            ok = ok and close > sma20
        return "LONG" if ok else "CASH"


def load_model(model_dir: str | Path) -> LoadedModel:
    model_dir = Path(model_dir)
    meta_p, w_p = model_dir / "meta.json", model_dir / "model.npz"
    if not meta_p.exists() or not w_p.exists():
        raise ModelLoadError(f"No se encontró meta.json/model.npz en {model_dir}. Ejecute scripts/train_cpu_baseline.py o copie la exportación del notebook.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("feature_contract") != F.FEATURE_CONTRACT_VERSION:
        raise ModelLoadError(f"Contrato de variables {meta.get('feature_contract')} incompatible con {F.FEATURE_CONTRACT_VERSION}.")
    w = dict(np.load(w_p))
    m = LoadedModel(meta=meta, weights=w)
    if "check_x" in w:  # prueba de contrato: misma predicción que en el entrenamiento
        got = m.predict_raw(w["check_x"])
        if not np.allclose(got, w["check_y"], atol=1e-6, rtol=1e-4):
            raise ModelLoadError(f"La inferencia NumPy no reproduce al modelo entrenado (dif máx {np.abs(got - w['check_y']).max():.2e}).")
    return m
