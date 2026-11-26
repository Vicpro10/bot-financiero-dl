"""Genera DL_Finanzas_BTCUSDT_MEXC.ipynb (sin salidas). Ejecutar:  python notebook/build_notebook.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Práctica de Deep Learning para Finanzas — BTC/USDT (MEXC)
**Del pronóstico de rendimientos a las bases de un agente de trading** · Capítulos 7 a 11 de *Deep Learning for Finance* (S. Kaabar)

| Ficha del activo | |
|---|---|
| Mercado | Criptomonedas |
| Par | **BTC/USDT** (moneda base BTC, moneda cotizada USDT) |
| Plataforma de origen | **MEXC** spot, API pública v3 `/api/v3/klines`, intervalo `1d` |
| Zona horaria | **UTC**; cada vela diaria abre a las 00:00 UTC y cierra a las 23:59:59 UTC |
| Calendario | Continuo (365 días); no hay días sin negociación que rellenar |
| Ajustes | No aplican dividendos ni particiones (cripto); precios tal cual los publica MEXC |
| Frecuencia | Diaria, solo velas **cerradas** |

**Cómo ejecutar:** en Colab, *Entorno de ejecución → Cambiar tipo de entorno → GPU T4* y luego *Ejecutar todo*.
El notebook se detiene si no detecta una T4 (regla del protocolo). Todas las salidas se guardan en Google Drive
(`MyDrive/bot_financiero_dl/`) y en un `.zip` descargable.

**Pregunta de investigación:** ¿el modelo profundo reduce el error frente a una referencia sencilla y genera señales útiles después de costos?
""")

md("## 0. Entorno, semillas y bitácora")
code(r"""
import os, sys, json, time, math, random, platform, subprocess, pickle, zipfile, hashlib, warnings
import datetime as dt
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)

IN_COLAB = "google.colab" in sys.modules
ALLOW_CPU_TEST = os.environ.get("ALLOW_CPU_TEST") == "1"   # solo para probar el código fuera de Colab; NO es evidencia
SEED = 42
SYMBOL = "BTCUSDT"
START = "2020-01-01"
WINDOW = 20
FEE = 0.001            # 0.10% por compra y por venta (supuesto académico)
CAPITAL = 100_000.0

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(SEED)

if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT = Path("/content/drive/MyDrive/bot_financiero_dl")
    except Exception as e:   # sin Drive: se guarda localmente y se descarga el zip al final
        print("Drive no disponible:", e)
        OUT = Path("/content/bot_financiero_dl")
else:
    OUT = Path("salidas")
for sub in ["datos/raw", "datos/processed", "modelos", "checkpoints", "figuras", "resultados"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

BITACORA = []      # fecha | etapa | error | causa | acción correctiva | verificación
def log_event(etapa, error, causa, accion, verificacion):
    BITACORA.append({"fecha": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "etapa": etapa,
                     "error": error, "causa": causa, "accion_correctiva": accion, "verificacion": verificacion})
    print(f"[BITÁCORA] {etapa}: {error} -> {accion}")

EXPERIMENTOS = []
def log_exp(**k):
    EXPERIMENTOS.append(k)

print("Salidas en:", OUT.resolve())
""")

md("""
### 0.1 Evidencia obligatoria de la GPU T4
1) `nvidia-smi` · 2) PyTorch reconoce la GPU y ejecuta una operación en ella · 3) versiones, semilla y dispositivo efectivo.
Si falta la T4 se detiene el entrenamiento profundo y se registra el incidente.
""")
code(r"""
smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True) if subprocess.run(["where" if os.name == "nt" else "which", "nvidia-smi"], capture_output=True).returncode == 0 else None
smi_txt = smi.stdout if smi else "nvidia-smi no disponible"
print(smi_txt)
HAS_CUDA = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else None
if not (HAS_CUDA and "T4" in (GPU_NAME or "")):
    log_event("entorno", "T4 ausente o GPU no utilizada", f"cuda={HAS_CUDA}, gpu={GPU_NAME}",
              "detener entrenamiento profundo y reprogramar en Colab con T4", "pendiente")
    if not ALLOW_CPU_TEST:
        raise RuntimeError("No se detectó una NVIDIA T4. Cambie el entorno a GPU T4 y vuelva a ejecutar (protocolo, sección 4).")
    print("⚠️ PRUEBA LOCAL EN CPU: estas salidas NO sirven como evidencia de T4.")
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
a = torch.randn(2048, 2048, device=DEVICE)
t0 = time.time(); b = (a @ a).sum(); torch.cuda.synchronize() if HAS_CUDA else None
print(f"Operación de prueba en {b.device} ({(time.time()-t0)*1000:.1f} ms)")
ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": GPU_NAME,
       "device": str(DEVICE), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
       "seed": SEED, "colab": IN_COLAB, "fecha_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
print(json.dumps(ENV, indent=2))
(OUT / "resultados" / "evidencia_t4.txt").write_text(smi_txt + "\n\n" + json.dumps(ENV, indent=2) + f"\n\ntensor de prueba en: {b.device}\n", encoding="utf-8")
""")

md("## 1. Datos: descarga, registro y auditoría (capítulos 7 y 11)")
code(r"""
class DataValidationError(ValueError):
    pass

DAY_MS = 86_400_000
def mexc_daily(symbol=SYMBOL, start=START, retries=3):
    # Velas diarias CERRADAS de MEXC (pagina de 500 en 500 días). Reintentos con espera creciente.
    now_ms = int(time.time() * 1000)
    cur = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    while cur < now_ms:
        end = min(cur + 500 * DAY_MS - 1, now_ms)
        url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval=1d&startTime={cur}&endTime={end}&limit=1000"
        for k in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    rows += json.load(r); break
            except Exception as e:
                if k == retries - 1:
                    log_event("descarga", "descarga fallida", repr(e), "detener el flujo", "sin datos")
                    raise RuntimeError(f"MEXC no respondió para {symbol}: {e}. Revise la conexión y vuelva a ejecutar.") from e
                time.sleep(2 ** k)
        cur = end + 1
    df = pd.DataFrame([x[:7] for x in rows], columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    df = df[df.close_time.astype("int64") <= now_ms]                       # excluir la vela aún abierta
    df["date"] = pd.to_datetime(df.open_time.astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]]

raw = mexc_daily()
DESCARGA = {"fuente": "MEXC spot API v3 /api/v3/klines interval=1d", "simbolo": SYMBOL, "moneda_cotizada": "USDT",
            "zona_horaria": "UTC", "fecha_descarga_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "periodo": f"{raw.date.min().date()} a {raw.date.max().date()}", "filas_descargadas": len(raw),
            "ajustes": "no aplican (cripto); sin relleno de días ni volumen inventado"}
raw.to_csv(OUT / "datos/raw" / f"{SYMBOL}_1d_mexc_raw.csv", index=False)
(OUT / "datos/raw" / "metadatos_descarga.json").write_text(json.dumps(DESCARGA, indent=2, ensure_ascii=False))
DESCARGA
""")
code(r"""
DICCIONARIO = pd.DataFrame([
    ("date", "Apertura de la vela diaria (UTC)", "fecha"), ("open", "Precio de apertura", "USDT"),
    ("high", "Máximo del día", "USDT"), ("low", "Mínimo del día", "USDT"), ("close", "Cierre del día", "USDT"),
    ("volume", "Volumen negociado (lo reporta MEXC)", "BTC")], columns=["columna", "descripción", "unidad"])
display(DICCIONARIO) if "display" in dir() else print(DICCIONARIO)

def audit(df):
    # Auditoría con reglas explícitas. Devuelve datos limpios y un reporte; nunca rellena huecos.
    rep = {}
    if len(df) == 0:
        raise DataValidationError("Conjunto vacío: la fuente no devolvió datos (fuente MEXC, activo %s)." % SYMBOL)
    rep["filas_iniciales"] = len(df)
    d = df.copy()
    dup = d.date.duplicated(keep="last")
    rep["duplicadas_eliminadas"] = int(dup.sum())
    if dup.any():
        log_event("preparación", "fecha repetida", f"{dup.sum()} filas duplicadas", "conservar la última aparición", "índice único")
    d = d[~dup]
    rep["desordenadas"] = int((~d.date.is_monotonic_increasing))
    d = d.sort_values("date")
    num = d[["open", "high", "low", "close", "volume"]]
    bad = ~np.isfinite(num).all(axis=1) | (num[["open", "high", "low", "close"]] <= 0).any(axis=1)
    rep["filas_no_validas_eliminadas"] = int(bad.sum())
    if bad.any():
        log_event("preparación", "valores no finitos o precios no positivos", str(d.loc[bad, "date"].dt.date.tolist()[:5]),
                  "eliminar la fila (no se imputa con información futura)", "precios finitos y positivos")
    d = d[~bad]
    gaps = d.date.diff().dt.days.fillna(1)
    rep["huecos_calendario"] = int((gaps > 1).sum())
    rep["fechas_hueco"] = d.loc[gaps > 1, "date"].dt.date.astype(str).tolist()[:10]
    rep["filas_finales"] = len(d)
    assert d.date.is_unique and d.date.is_monotonic_increasing
    return d.reset_index(drop=True), rep

data, AUDITORIA = audit(raw)
print(json.dumps(AUDITORIA, indent=2))
if AUDITORIA["filas_finales"] < 1500:
    raise DataValidationError(f"Se requieren ≥1,500 observaciones válidas; hay {AUDITORIA['filas_finales']}.")
""")
code(r"""
def build_features(df):
    # Contrato de variables v1 (idéntico a backend/app/features.py). Fila t: solo información hasta el cierre de t.
    d = df.set_index("date"); c = d.close
    f = pd.DataFrame(index=d.index)
    f[["open", "high", "low", "close", "volume"]] = d[["open", "high", "low", "close", "volume"]]
    f["ret"] = c.pct_change(fill_method=None)
    for k in range(WINDOW):
        f[f"ret_lag{k}"] = f.ret.shift(k)                       # ret_lag0 = P_t/P_{t-1} - 1
    f["sma20"] = c.rolling(20).mean()
    f["dist_sma20"] = c / f.sma20 - 1                            # distancia a la SMA de 20 cierres
    f["vol20"] = f.ret.rolling(20).std(ddof=1)                   # volatilidad muestral de 20 rendimientos, sin anualizar
    delta = c.diff(); g = delta.clip(lower=0); l = (-delta).clip(lower=0)
    ag = g.ewm(alpha=1/14, adjust=False, min_periods=14).mean(); al = l.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    tot = ag + al
    f["rsi14"] = (100 * ag / tot.replace(0, np.nan)).mask(tot.eq(0) & ag.notna(), 50.0)  # ambas cero -> 50 (neutral)
    f["target"] = c.shift(-1) / c - 1                            # y_t = P_{t+1}/P_t - 1
    f["target_date"] = pd.Series(f.index, index=f.index).shift(-1)
    return f

LAGS = [f"ret_lag{k}" for k in range(WINDOW)]
IND = ["dist_sma20", "vol20"]
feats = build_features(data)
usable = feats.dropna(subset=LAGS + IND + ["target"]).copy()   # se retiran filas sin indicador válido o sin objetivo
print("Primera fecha con todas las variables válidas:", usable.index[0].date(), "| filas útiles:", len(usable))
usable.to_csv(OUT / "datos/processed" / f"{SYMBOL}_features_v1.csv")
""")
code(r"""
fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
feats.close.plot(ax=ax[0], lw=1, label="Cierre"); feats.sma20.plot(ax=ax[0], lw=1, label="SMA20"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("BTC/USDT diario (MEXC, UTC)")
feats.ret.plot(ax=ax[1], lw=.6); ax[1].set_title("Rendimiento diario simple")
feats.vol20.plot(ax=ax[2], lw=1, color="C3"); ax[2].set_title("Volatilidad histórica (desv. est. de 20 rendimientos, sin anualizar)")
plt.tight_layout(); plt.savefig(OUT / "figuras/precio_rend_vol.png", dpi=120); plt.show()
z = (feats.ret - feats.ret.mean()) / feats.ret.std()
extremos = feats.loc[z.abs() > 4, ["close", "ret"]]
print("Rendimientos extremos (|z|>4): se conservan (son movimientos reales del mercado, no errores):")
print(extremos.round(4).to_string())
""")

md(r"""
## 2. Capítulo 7 — Marco de trabajo, separación temporal y referencias
**Objetivo:** al cierre de *t* estimar $y_t = P_{t+1}/P_t - 1$ con los 20 rendimientos más recientes.
Los ejemplos se asignan a TR/VA/TE por la **fecha del objetivo** (70/15/15), sin división aleatoria.
""")
code(r"""
n = len(usable); i_tr, i_va = int(n * .70), int(n * .85)
SETS = {"TR": usable.iloc[:i_tr], "VA": usable.iloc[i_tr:i_va], "TE": usable.iloc[i_va:]}
assert SETS["TR"].target_date.iloc[-1] < SETS["VA"].target_date.iloc[0], "Fuga: el último objetivo de TR no es anterior al primero de VA"
assert SETS["VA"].target_date.iloc[-1] < SETS["TE"].target_date.iloc[0], "Fuga: el último objetivo de VA no es anterior al primero de TE"
PARTICIONES = pd.DataFrame({k: {"inicio (emisión)": v.index[0].date(), "fin (emisión)": v.index[-1].date(),
                                "primer objetivo": v.target_date.iloc[0].date(), "último objetivo": v.target_date.iloc[-1].date(),
                                "ejemplos": len(v)} for k, v in SETS.items()}).T
print(PARTICIONES.to_string())
PARTICIONES.to_csv(OUT / "resultados/particiones.csv")

# Control de aprendizaje: una fila completa
r = SETS["VA"].iloc[0]
print("\nCONTROL DE APRENDIZAJE (primera fila de VA)")
print(" Entradas: rendimientos de", (SETS['VA'].index[0] - pd.Timedelta(days=WINDOW-1)).date(), "a", SETS['VA'].index[0].date(),
      "| SMA20/vol20 con cierres hasta", SETS['VA'].index[0].date())
print(" Fecha de emisión (cierre t):", SETS['VA'].index[0].date(), "| Fecha objetivo (cierre t+1):", r.target_date.date(),
      f"| y = {r.close:.2f} -> {r.close*(1+r.target):.2f} = {r.target:.4%}")
print(" El modelo no conoce la respuesta: todas las entradas usan cierres ≤ t; el objetivo usa P_{t+1}, que se publica 24 h después.")
""")
code(r"""
def reg_metrics(y, p):
    # error = estimado - real. Precisión direccional: acierto si coinciden los signos; los rendimientos reales = 0 se excluyen.
    y, p = np.asarray(y, float), np.asarray(p, float); e = p - y; m = y != 0
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "DA": float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.any() else np.nan}

Y = {k: v.target.to_numpy() for k, v in SETS.items()}
def fit_scaler(cols, train_df):
    return StandardScaler().fit(train_df[cols])          # SOLO con el entrenamiento correspondiente

# Referencia 1: pronóstico cero
for k in ("TR", "VA"):
    log_exp(id="cap7_cero", capitulo=7, conjunto=k, modelo="cero", variables="-", config="-", semilla="-", dispositivo="cpu", tiempo_s=0, **reg_metrics(Y[k], np.zeros_like(Y[k])))
# Referencia 2: regresión lineal con los mismos 20 rezagos
t0 = time.time()
sx_lags = fit_scaler(LAGS, SETS["TR"])
lin = LinearRegression().fit(sx_lags.transform(SETS["TR"][LAGS]), Y["TR"])
PRED = {"linreg_lags": {k: lin.predict(sx_lags.transform(v[LAGS])) for k, v in SETS.items()}}
for k in ("TR", "VA"):
    log_exp(id="cap7_linreg", capitulo=7, conjunto=k, modelo="regresión lineal", variables="20 rezagos", config="OLS", semilla="-", dispositivo="cpu", tiempo_s=round(time.time()-t0, 3), **reg_metrics(Y[k], PRED["linreg_lags"][k]))
pd.DataFrame(EXPERIMENTOS).round(6)
""")
md(r"""
**Hipótesis técnica (verificable):** *la distancia del cierre respecto de su SMA de 20 días (`dist_sma20`) y la volatilidad de 20 días
contienen información sobre el rendimiento del día siguiente que no está en los rezagos.* Se contrasta en el capítulo 11 comparando el
mismo modelo con y sin esos indicadores en VA. Una figura no demuestra capacidad predictiva; abajo solo se describe la relación en TR.

**Seguimiento:** el modelo profundo tendría que bajar el RMSE/MAE de VA respecto al pronóstico cero (que en rendimientos diarios es muy difícil de superar).
""")
code(r"""
tr = SETS["TR"]
print("Correlación en TR (descriptiva, no causal):")
print(tr[["dist_sma20", "vol20", "rsi14", "ret_lag0", "target"]].corr()["target"].round(4).to_string())
""")

md("""
## 3. Capítulo 8 — MLP y LSTM entrenadas en la T4
Mismo objetivo, mismas variables (20 rezagos) y mismas particiones. Adam, pérdida MSE sobre el objetivo estandarizado con TR,
parada temprana con VA y restauración de los mejores pesos. Tres configuraciones por red (solo cambian las neuronas) y tres semillas para la mejor.
""")
code(r"""
class MLP(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        layers += [nn.Linear(d, 1)]                      # salida lineal: admite rendimientos positivos y negativos
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class LSTMNet(nn.Module):
    def __init__(self, n_vars, units):
        super().__init__()
        self.lstm = nn.LSTM(n_vars, units, batch_first=True)
        self.head = nn.Linear(units, 1)
    def forward(self, x):                               # x: (ejemplos, pasos, variables)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def seq_tensor(df_all, rows_index, vars_):
    # (ejemplos, 20 pasos, variables); el último paso es la fila t. Usa historia previa como contexto (permitido).
    arr = df_all[vars_].to_numpy(float); pos = df_all.index.get_indexer(rows_index)
    if (pos < WINDOW - 1).any():
        raise DataValidationError("Ventanas insuficientes: hay filas sin 20 pasos previos.")
    X = np.stack([arr[i - WINDOW + 1:i + 1] for i in pos])
    if X.shape[1:] != (WINDOW, len(vars_)):
        raise DataValidationError(f"Forma incorrecta: esperada (n,{WINDOW},{len(vars_)}), recibida {X.shape}")
    if not np.isfinite(X).all():
        raise DataValidationError("Valores no finitos dentro de las ventanas LSTM.")
    return X

def make_inputs(kind, vars_, train_df, sets):
    # Ajusta el escalador SOLO con train_df y transforma el resto sin reajustar.
    if kind == "mlp":
        sx = StandardScaler().fit(train_df[vars_])
        return {k: sx.transform(v[vars_]) for k, v in sets.items()}, sx
    base = feats.dropna(subset=vars_)
    sx = StandardScaler().fit(base.loc[base.index <= train_df.index[-1], vars_])   # pasos de TR
    scaled = base.copy(); scaled[vars_] = sx.transform(base[vars_])
    return {k: seq_tensor(scaled, v.index, vars_) for k, v in sets.items()}, sx

def train_net(model, Xtr, ytr, Xva, yva, lr=1e-3, epochs=400, batch=64, patience=30, seed=SEED, tag="m"):
    set_seed(seed)
    model = model.to(DEVICE)
    for p_ in model.parameters():
        assert p_.device.type == DEVICE.type
    ym, ys = ytr.mean(), ytr.std()
    tX = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Xt, yt, Xv, yv = tX(Xtr), tX((ytr - ym) / ys), tX(Xva), tX((yva - ym) / ys)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    best, best_state, best_ep, bad, hist = np.inf, None, 0, 0, {"tr": [], "va": []}
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), generator=g).to(DEVICE); tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                log_event("entrenamiento", "pérdida no finita", tag, "detener; revisar escalas/lr", "pendiente")
                raise FloatingPointError(f"Pérdida no finita en {tag}, época {ep}")
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lv = lossf(model(Xv), yv).item()
        hist["tr"].append(tot / len(Xt)); hist["va"].append(lv)
        if lv < best - 1e-6:
            best, best_ep, bad = lv, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)                  # restaurar los mejores pesos según VA
    torch.save(best_state, OUT / "checkpoints" / f"{tag}_seed{seed}.pt")
    model.eval()
    predict = lambda X: (model(tX(X)).detach().cpu().numpy() * ys + ym)
    info = {"epoca_mejor": best_ep, "epocas": len(hist["tr"]), "tiempo_s": round(time.time() - t0, 2),
            "dispositivo": str(next(model.parameters()).device), "y_mean": float(ym), "y_std": float(ys)}
    return model, predict, hist, info
""")
code(r"""
CONFIGS = {"mlp": [(16,), (32,), (64,)], "lstm": [8, 16, 32]}
LR, BATCH, EPOCHS = 1e-3, 64, 400
XIN = {"mlp": make_inputs("mlp", LAGS, SETS["TR"], SETS), "lstm": make_inputs("lstm", ["ret"], SETS["TR"], SETS)}
print("Forma LSTM (ejemplos × pasos × variables):", {k: v.shape for k, v in XIN["lstm"][0].items()})
RES8 = {}
for kind, cfgs in CONFIGS.items():
    X = XIN[kind][0]
    for cfg in cfgs:
        m = MLP(WINDOW, cfg) if kind == "mlp" else LSTMNet(1, cfg)
        tag = f"cap8_{kind}_{cfg if kind=='lstm' else 'x'.join(map(str,cfg))}"
        m, pred, hist, info = train_net(m, X["TR"], Y["TR"], X["VA"], Y["VA"], lr=LR, epochs=EPOCHS, batch=BATCH, tag=tag)
        P = {k: pred(X[k]) for k in ("TR", "VA")}
        RES8[(kind, cfg)] = {"hist": hist, "info": info, "params": n_params(m), "va_rmse": reg_metrics(Y["VA"], P["VA"])["RMSE"]}
        for k in ("TR", "VA"):
            log_exp(id=tag, capitulo=8, conjunto=k, modelo=kind.upper(), variables="20 rezagos", semilla=SEED,
                    config=f"capas={'LSTM('+str(cfg)+')' if kind=='lstm' else cfg}+Dense(1); params={n_params(m)}; Adam lr={LR}; batch={BATCH}; max_epocas={EPOCHS}; MSE",
                    dispositivo=info["dispositivo"], tiempo_s=info["tiempo_s"], epoca_mejor=info["epoca_mejor"], **reg_metrics(Y[k], P[k]))
        print(f"{tag:22s} params={n_params(m):6d} mejor época={info['epoca_mejor']:3d} RMSE VA={RES8[(kind,cfg)]['va_rmse']:.6f} [{info['dispositivo']}, {info['tiempo_s']}s]")
""")
code(r"""
BEST8 = {kind: min([k for k in RES8 if k[0] == kind], key=lambda k: RES8[k]["va_rmse"]) for kind in CONFIGS}
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for a_, kind in zip(ax, CONFIGS):
    h = RES8[BEST8[kind]]["hist"]; a_.plot(h["tr"], label="TR"); a_.plot(h["va"], label="VA")
    a_.axvline(RES8[BEST8[kind]]["info"]["epoca_mejor"], ls="--", c="gray", label="mejor época (VA)")
    a_.set_title(f"{kind.upper()} {BEST8[kind][1]} — pérdida MSE (y estandarizado)"); a_.legend()
plt.tight_layout(); plt.savefig(OUT / "figuras/curvas_cap8.png", dpi=120); plt.show()

SEEDS = [42, 7, 2024]
SEMILLAS = []
FINAL8 = {}
for kind, key in BEST8.items():
    X = XIN[kind][0]; vals = []
    for s in SEEDS:
        m = MLP(WINDOW, key[1]) if kind == "mlp" else LSTMNet(1, key[1])
        m, pred, hist, info = train_net(m, X["TR"], Y["TR"], X["VA"], Y["VA"], lr=LR, epochs=EPOCHS, batch=BATCH, seed=s, tag=f"cap8_{kind}_best")
        r_ = reg_metrics(Y["VA"], pred(X["VA"])); vals.append(r_["RMSE"])
        log_exp(id=f"cap8_{kind}_best_s{s}", capitulo=8, conjunto="VA", modelo=kind.upper(), variables="20 rezagos", semilla=s,
                config=str(key[1]), dispositivo=info["dispositivo"], tiempo_s=info["tiempo_s"], epoca_mejor=info["epoca_mejor"], **r_)
        if s == SEED:
            FINAL8[kind] = (m, pred)
    SEMILLAS.append({"red": kind.upper(), "config": str(key[1]), "RMSE_VA_media": np.mean(vals), "RMSE_VA_desv": np.std(vals, ddof=1)})
print(pd.DataFrame(SEMILLAS).to_string())
for kind in CONFIGS:
    PRED[f"{kind}_lags"] = {k: FINAL8[kind][1](XIN[kind][0][k]) for k in SETS}
tab = pd.DataFrame({name: {k: reg_metrics(Y[k], p[k])["RMSE"] for k in ("TR", "VA")} for name, p in
                    {"cero": {k: np.zeros_like(Y[k]) for k in SETS}, **PRED}.items()}).T
print("\nRMSE por modelo (selección con VA):\n", tab.round(6).to_string())
""")
md("""
**Seguimiento:** ¿una menor pérdida en TR mejora VA? Compare las curvas: la pérdida de TR sigue bajando mientras la de VA se estanca o sube
→ sobreajuste a partir de la época marcada; por eso se restauran los pesos de la mejor época de VA. Si TR y VA quedan altos y parecidos → subajuste
(los rendimientos diarios tienen poca señal).
""")

md("""
## 4. Capítulo 9 — Validación temporal con ventana expansiva y umbrales
Tres evaluaciones dentro de TR: cada entrenamiento termina antes del bloque que evalúa y el escalador se reajusta solo con ese entrenamiento.
""")
code(r"""
blocks = np.array_split(np.arange(len(SETS["TR"])), 4)
CV = []
for f_ in range(1, 4):
    tr_idx = np.concatenate(blocks[:f_]); ev_idx = blocks[f_]
    tr_df, ev_df = SETS["TR"].iloc[tr_idx], SETS["TR"].iloc[ev_idx]
    assert tr_df.target_date.iloc[-1] < ev_df.target_date.iloc[0]
    yt, ye = tr_df.target.to_numpy(), ev_df.target.to_numpy()
    sx = StandardScaler().fit(tr_df[LAGS])
    p_lin = LinearRegression().fit(sx.transform(tr_df[LAGS]), yt).predict(sx.transform(ev_df[LAGS]))
    Xf, _ = make_inputs("mlp", LAGS, tr_df, {"tr": tr_df, "ev": ev_df})
    # parada temprana con el último 15% del propio entrenamiento (sin tocar el bloque evaluado)
    cut = int(len(tr_df) * .85)
    m, pred, _, info = train_net(MLP(WINDOW, BEST8["mlp"][1]), Xf["tr"][:cut], yt[:cut], Xf["tr"][cut:], yt[cut:], lr=LR, tag=f"cap9_cv{f_}")
    p_mlp = pred(Xf["ev"])
    for name, p_ in [("cero", np.zeros_like(ye)), ("linreg", p_lin), ("mlp", p_mlp)]:
        r_ = reg_metrics(ye, p_)
        CV.append({"ventana": f_, "entrena_hasta": tr_df.index[-1].date(), "evalua": f"{ev_df.index[0].date()}→{ev_df.index[-1].date()}", "modelo": name, **r_})
        log_exp(id=f"cap9_cv{f_}_{name}", capitulo=9, conjunto=f"CV{f_}", modelo=name, variables="20 rezagos", config="expansiva", semilla=SEED, dispositivo=info["dispositivo"] if name == "mlp" else "cpu", tiempo_s=info["tiempo_s"] if name == "mlp" else 0, **r_)
CV = pd.DataFrame(CV); print(CV.round(6).to_string()); CV.to_csv(OUT / "resultados/ventanas_expansivas.csv", index=False)
""")
code(r"""
def simulate(px, signals, capital=CAPITAL, fee=FEE):
    # Señal al cierre de t -> ejecución en la apertura de t+1; patrimonio al cierre; largo o efectivo; liquidación final.
    idx = signals.index; cash, qty, entry, fees, trades, rows = capital, 0.0, 0.0, 0.0, [], []
    for i in range(1, len(idx)):
        sig, o, c = signals.iloc[i-1], px.open.iloc[i], px.close.iloc[i]; act = "MANTENER"
        if sig == "LONG" and qty == 0:
            qty = cash / (o * (1 + fee)); f = qty * o * fee; entry, cash, fees, act = cash, 0.0, fees + f, "COMPRA"
        elif sig == "CASH" and qty > 0:
            g_ = qty * o; f = g_ * fee; cash = g_ - f; fees += f; trades.append(cash - entry); qty, act = 0.0, "VENTA"
        rows.append({"fecha": idx[i].date(), "precio_ejecucion": o if act != "MANTENER" else np.nan, "senal_previa": sig, "accion": act,
                     "posicion_btc": qty, "efectivo": cash, "costos_acum": fees, "patrimonio": cash + qty * c})
    t = pd.DataFrame(rows)
    if qty > 0:
        g_ = qty * px.close.iloc[-1]; f = g_ * fee; fees += f; trades.append(g_ - f - entry)
        t.loc[t.index[-1], ["patrimonio", "costos_acum"]] = [g_ - f, fees]
    eq = t.patrimonio
    return t, {"rend_neto": eq.iloc[-1] / capital - 1, "max_dd": float((1 - eq / eq.cummax()).max()), "operaciones": len(trades),
               "win_rate": (np.mean([x > 0 for x in trades]) if trades else np.nan), "costos": fees, "patrimonio_final": eq.iloc[-1]}

DELTAS = [0.0005, 0.001, 0.002, 0.003, 0.005]   # hasta cinco umbrales positivos predefinidos
best_kind = min(["linreg_lags", "mlp_lags", "lstm_lags"], key=lambda k: reg_metrics(Y["VA"], PRED[k]["VA"])["RMSE"])
UMB = []
for d in DELTAS:
    sig = pd.Series(np.where(PRED[best_kind]["VA"] > d, "LONG", "CASH"), index=SETS["VA"].index)
    UMB.append({"modelo": best_kind, "delta": d, **simulate(SETS["VA"], sig)[1]})
UMB = pd.DataFrame(UMB).sort_values(["rend_neto", "max_dd"], ascending=[False, True]); print(UMB.round(4).to_string())
print("Regla preliminar elegida en VA: comprar si pronóstico >", UMB.iloc[0].delta, "(un umbral no es una probabilidad de acierto)")
""")
md("""
**Diferenciación fraccionaria (explicación):** diferenciar con orden *d* no entero (0<d<1) busca una serie estacionaria que conserve memoria
de largo plazo, a diferencia de los rendimientos (d=1) que la eliminan. Aquí el modelo ya usa **rendimientos** (estacionarios) y
`dist_sma20`/`vol20` (acotados); aplicarla al precio sería pertinente si se quisiera usar el nivel de precio como variable. Queda como ampliación opcional.

**Seguimiento:** ¿el resultado depende de un único periodo favorable? Revise la tabla de ventanas expansivas: si el modelo solo gana en una ventana, la mejora no es robusta.
TE no intervino en ninguna decisión hasta aquí.
""")

md(r"""
## 5. Capítulo 10 — Diseño del entorno del futuro agente (DQN/DDQN)
| Elemento | Definición |
|---|---|
| **Estado** $s_t$ | 20 rendimientos recientes, pronóstico $\hat y_t$, `vol20`, `dist_sma20`, efectivo/patrimonio, posición (0 = efectivo, 1 = comprado). Todo conocido al cierre de *t*. |
| **Acciones** | `COMPRAR` (solo si posición=0), `MANTENER` (conserva lo que haya: posición abierta **o** efectivo), `VENDER` (solo si posición=1). |
| **Restricciones** | Sin apalancamiento, sin cortos, compra ≤ efectivo disponible (incluyendo el costo). |
| **Ejecución** | Decisión al cierre de *t* → se ejecuta en la apertura de *t+1*. |
| **Recompensa** | $r_{t+1} = (W_{t+1} - W_t)/W_t$: cambio porcentual del patrimonio **después de costos**, calculado al cierre de *t+1*. |

**Pronóstico vs señal vs política:** el pronóstico es un número ($\hat y$); la señal es una regla fija sobre él (comprar si $\hat y>\delta$);
la política aprendida (DQN) asigna una acción a cada estado maximizando la recompensa acumulada, aprendida por prueba y error.
**Exploración** ε-greedy: con probabilidad ε se prueba una acción al azar (ε decrece). **Memoria de experiencias**: se guardan $(s,a,r,s',done)$ y se entrena con
lotes aleatorios para romper la correlación temporal. **Red objetivo**: copia de la red que se sincroniza cada N pasos para estabilizar el objetivo;
en **DDQN** la red online *elige* la acción y la objetivo la *evalúa*, reduciendo la sobreestimación.
""")
code(r"""
class TradingEnv:
    # Entorno mínimo para el futuro agente. No se entrena aquí (no lo exige la práctica).
    ACTIONS = ("COMPRAR", "MANTENER", "VENDER")
    def __init__(self, df, preds, capital=CAPITAL, fee=FEE):
        self.df, self.preds, self.capital, self.fee = df, preds, capital, fee
        self.reset()
    def reset(self):
        self.t, self.cash, self.qty = 0, self.capital, 0.0
        return self.state()
    def equity(self, price):
        return self.cash + self.qty * price
    def state(self):
        r = self.df.iloc[self.t]
        return {"fecha": r.name.date(), "rend_recientes": r[LAGS].to_numpy().round(4).tolist()[:5], "pronostico": float(self.preds[self.t]),
                "vol20": float(r.vol20), "dist_sma20": float(r.dist_sma20), "efectivo": self.cash, "posicion": int(self.qty > 0)}
    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"Acción inválida {action}")
        w0 = self.equity(self.df.close.iloc[self.t])
        nxt = self.df.iloc[self.t + 1]; o = nxt.open; cost = 0.0
        if action == "COMPRAR" and self.qty == 0:
            self.qty = self.cash / (o * (1 + self.fee)); cost = self.qty * o * self.fee; self.cash = 0.0
        elif action == "VENDER" and self.qty > 0:
            g_ = self.qty * o; cost = g_ * self.fee; self.cash = g_ - cost; self.qty = 0.0
        # COMPRAR con posición o VENDER sin posición = no-op (no permite cortos ni recompras)
        self.t += 1
        w1 = self.equity(nxt.close)
        reward = (w1 - w0) / w0
        done = self.t >= len(self.df) - 1
        return self.state(), reward, done, {"precio_ejecucion": o, "costo": cost, "patrimonio": w1}

env = TradingEnv(SETS["VA"], PRED[best_kind]["VA"])
s0 = env.reset()
print("ESTADO s_t:", s0)
s1, r1, done, info = env.step("COMPRAR")
print("ACCIÓN: COMPRAR al cierre de", s0["fecha"], "→ EJECUCIÓN en la apertura de", s1["fecha"], f"a {info['precio_ejecucion']:.2f} (costo {info['costo']:.2f})")
print(f"RECOMPENSA r = (W1-W0)/W0 = ({info['patrimonio']:.2f} - {CAPITAL:.2f}) / {CAPITAL:.2f} = {r1:.5f}")
print("NUEVO ESTADO s_t+1:", s1)
# ¿El agente podría explotar un error del simulador? Comprobaciones:
e2 = TradingEnv(SETS["VA"], PRED[best_kind]["VA"]); e2.reset()
_, r_a, _, _ = e2.step("VENDER")                 # vender sin posición no genera dinero
assert e2.cash == CAPITAL and e2.qty == 0 and abs(r_a) < 1e-12
e2.step("COMPRAR"); q = e2.qty; e2.step("COMPRAR")   # recomprar no duplica la posición
assert e2.qty == q and e2.cash == 0
print("✔ Sin ganancias indebidas: vender sin posición y recomprar son no-ops; se usa la apertura de t+1 (no el cierre de t) y se cobran costos.")
""")

md("""
## 6. Capítulo 11 — Indicadores técnicos como variables y como filtro
A. Modelo con rezagos vs B. mismo modelo + `dist_sma20` y `vol20` (mismas fechas de VA). Después, C. estrategia solo con el pronóstico vs D. con filtro *cierre > SMA20*.
""")
code(r"""
IND_DOC = pd.DataFrame([
    ("dist_sma20", "Separación del precio respecto de su tendencia de 20 días", "20 cierres", "cierre_t / SMA20_t − 1", str(feats.dist_sma20.first_valid_index().date())),
    ("vol20", "Variabilidad reciente", "20 rendimientos", "desv. est. muestral (ddof=1), sin anualizar", str(feats.vol20.first_valid_index().date())),
    ("rsi14", "Momentum (solo UI/LLM, no entra al modelo)", "14 cambios", "Wilder: 100·G/(G+P); G+P=0 → 50", str(feats.rsi14.first_valid_index().date())),
], columns=["indicador", "representa", "ventana", "fórmula", "primera fecha válida"])
print(IND_DOC.to_string())
print("\nCorrelaciones en TR (redundancia):\n", SETS["TR"][["ret_lag0", "dist_sma20", "vol20"]].corr().round(3).to_string())

ALL = LAGS + IND
XIN_I = {"mlp": make_inputs("mlp", ALL, SETS["TR"], SETS), "lstm": make_inputs("lstm", ["ret", "dist_sma20", "vol20"], SETS["TR"], SETS)}
sx_all = fit_scaler(ALL, SETS["TR"]); lin_i = LinearRegression().fit(sx_all.transform(SETS["TR"][ALL]), Y["TR"])
PRED["linreg_ind"] = {k: lin_i.predict(sx_all.transform(v[ALL])) for k, v in SETS.items()}
MODELS11 = {}
for kind in ("mlp", "lstm"):
    X = XIN_I[kind][0]; cfg = BEST8[kind][1]
    m = MLP(len(ALL), cfg) if kind == "mlp" else LSTMNet(3, cfg)
    m, pred, hist, info = train_net(m, X["TR"], Y["TR"], X["VA"], Y["VA"], lr=LR, tag=f"cap11_{kind}_ind")
    PRED[f"{kind}_ind"] = {k: pred(X[k]) for k in SETS}; MODELS11[kind] = (m, info)
    for k in ("TR", "VA"):
        log_exp(id=f"cap11_{kind}_ind", capitulo=11, conjunto=k, modelo=kind.upper(), variables="20 rezagos + dist_sma20 + vol20", config=str(cfg), semilla=SEED,
                dispositivo=info["dispositivo"], tiempo_s=info["tiempo_s"], epoca_mejor=info["epoca_mejor"], **reg_metrics(Y[k], PRED[f"{kind}_ind"][k]))
for k in ("TR", "VA"):
    log_exp(id="cap11_linreg_ind", capitulo=11, conjunto=k, modelo="regresión lineal", variables="20 rezagos + dist_sma20 + vol20", config="OLS", semilla="-", dispositivo="cpu", tiempo_s=0, **reg_metrics(Y[k], PRED["linreg_ind"][k]))
AB = pd.DataFrame([{"modelo": m_, "A_rezagos_MAE": reg_metrics(Y["VA"], PRED[f"{m_}_lags"]["VA"])["MAE"], "A_rezagos_RMSE": reg_metrics(Y["VA"], PRED[f"{m_}_lags"]["VA"])["RMSE"],
                    "B_con_ind_MAE": reg_metrics(Y["VA"], PRED[f"{m_}_ind"]["VA"])["MAE"], "B_con_ind_RMSE": reg_metrics(Y["VA"], PRED[f"{m_}_ind"]["VA"])["RMSE"]} for m_ in ("linreg", "mlp", "lstm")])
AB["indicadores_mejoran_RMSE"] = AB.B_con_ind_RMSE < AB.A_rezagos_RMSE
print("\nComparación A vs B en VA:\n", AB.round(6).to_string())
print("\nDesviación estándar de las predicciones en VA (detecta predicciones casi constantes):")
print({k: float(np.std(v["VA"])) for k, v in PRED.items()})
""")
code(r"""
# Selección final con VA (TE aún cerrado): modelo por RMSE de VA; regla (δ, filtro) por rendimiento neto de VA, desempate menor drawdown.
CANDIDATOS = ["linreg_lags", "linreg_ind", "mlp_lags", "mlp_ind", "lstm_lags", "lstm_ind"]
va_rmse = {c: reg_metrics(Y["VA"], PRED[c]["VA"])["RMSE"] for c in CANDIDATOS}
FINAL = min(va_rmse, key=va_rmse.get)
DL_BEST = min([c for c in CANDIDATOS if not c.startswith("linreg")], key=va_rmse.get)
print("RMSE VA:", {k: round(v, 6) for k, v in sorted(va_rmse.items(), key=lambda x: x[1])})
print("Modelo final elegido con VA:", FINAL, "| mejor modelo profundo:", DL_BEST)
CD = []
va = SETS["VA"]
for model_name in sorted({FINAL, DL_BEST}):
    for filt in (False, True):
        for d in DELTAS:
            ok = PRED[model_name]["VA"] > d
            if filt:
                ok &= (va.close > va.sma20).to_numpy()
            sig = pd.Series(np.where(ok, "LONG", "CASH"), index=va.index)
            CD.append({"modelo": model_name, "estrategia": "D: pronóstico + filtro SMA20" if filt else "C: solo pronóstico", "filtro": filt, "delta": d, **simulate(va, sig)[1]})
CD = pd.DataFrame(CD).sort_values(["rend_neto", "max_dd"], ascending=[False, True])
print(CD.round(4).to_string())
def pick_rule(model_name):
    r_ = CD[CD.modelo == model_name].iloc[0]
    return {"delta": float(r_.delta), "sma_filter": bool(r_.filtro)}
RULES = {m_: pick_rule(m_) for m_ in {FINAL, DL_BEST}}
print("Reglas congeladas:", RULES)
""")

md("## 7. Manejo de errores: cuatro fallos provocados en copias pequeñas")
code(r"""
small = data.tail(60).copy()           # copia pequeña; los datos originales no se tocan
def check_df(df):
    if len(df) == 0:
        raise DataValidationError("Conjunto vacío. Verifique la descarga (fuente, activo, periodo).")
    if df.date.duplicated().any():
        raise DataValidationError(f"Fecha duplicada: {df.date[df.date.duplicated()].dt.date.tolist()}. Aplique la regla 'conservar la última'.")
    v = df[["open", "high", "low", "close"]]
    if not np.isfinite(v).all().all() or (v <= 0).any().any():
        raise DataValidationError("Valor no válido (NaN/inf o precio ≤ 0). Identifique la fecha y elimine la fila sin usar información futura.")
    return True

casos = {
    "conjunto vacío": small.iloc[0:0],
    "fecha duplicada": pd.concat([small, small.tail(1)]),
    "valor no válido": small.assign(close=small.close.where(small.index != small.index[-5], -1.0)),
}
for nombre, df_ in casos.items():
    try:
        check_df(df_)
        print("✘ no se detectó", nombre)
    except DataValidationError as e:
        print(f"✔ {nombre}: detenido -> {e}")
        log_event("prueba controlada", nombre, "copia pequeña modificada a propósito", "detener el flujo con mensaje", "excepción específica capturada")
# recuperación en fecha duplicada: aplicar la regla documentada y verificar
fixed, rep = audit(casos["fecha duplicada"]); assert fixed.date.is_unique; print("  recuperación: se eliminaron", rep["duplicadas_eliminadas"], "filas; índice único verificado")
# dimensiones incorrectas para la LSTM
try:
    bad = torch.zeros(4, WINDOW, 5, device=DEVICE)           # 5 variables en vez de 1
    with torch.no_grad():
        FINAL8["lstm"][0](bad)
    print("✘ no se detectó la forma incorrecta")
except RuntimeError as e:
    print("✔ dimensiones incorrectas: detenido antes de predecir -> esperada (n, 20, 1), recibida (4, 20, 5)")
    log_event("prueba controlada", "dimensiones incorrectas", "tensor (4,20,5) a una LSTM de 1 variable", "detener e informar forma esperada y recibida", "RuntimeError específico")
""")

md("## 8. Controles antes de TE, evaluación única en TE y simulación")
code(r"""
# Controles obligatorios
for k_, p_ in PRED.items():
    for s_ in SETS:
        assert np.isfinite(p_[s_]).all(), f"Predicciones no finitas en {k_}/{s_}"
assert (SETS["TE"].target_date.iloc[0] > SETS["VA"].target_date.iloc[-1])
assert list(OUT.glob("checkpoints/*.pt")), "Faltan checkpoints"
CONGELADO = {"modelo_final": FINAL, "mejor_dl": DL_BEST, "reglas": RULES, "deltas": DELTAS, "configs": {k: str(v[1]) for k, v in BEST8.items()},
             "lr": LR, "batch": BATCH, "semilla": SEED, "env": ENV}
blob = json.dumps(CONGELADO, sort_keys=True, default=str)
CONGELADO["hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
(OUT / "resultados/config_congelada.json").write_text(json.dumps(CONGELADO, indent=2, default=str))
print("Configuración congelada", CONGELADO["hash"], "— a partir de aquí TE se abre UNA sola vez.")

# Evaluación de las versiones finales PREDEFINIDAS en TR, VA y TE (no se vuelve a elegir con TE)
FINALES = ["linreg_lags", "linreg_ind", "mlp_lags", "mlp_ind", "lstm_lags", "lstm_ind"]
rows = []
for f_ in ["cero"] + FINALES:
    for s_ in SETS:
        p_ = np.zeros_like(Y[s_]) if f_ == "cero" else PRED[f_][s_]
        rows.append({"modelo": f_, "conjunto": s_, **reg_metrics(Y[s_], p_)})
        if s_ == "TE":
            log_exp(id=f"final_{f_}", capitulo=11, conjunto="TE", modelo=f_, variables="ver id", config="congelada " + CONGELADO["hash"], semilla=SEED, dispositivo=str(DEVICE), tiempo_s=0, **reg_metrics(Y[s_], p_))
HIST = pd.DataFrame(rows).pivot(index="modelo", columns="conjunto", values=["MAE", "RMSE", "DA"])
print(HIST.round(6).to_string())
dl_vs_lin = {s_: reg_metrics(Y[s_], PRED[DL_BEST][s_])["RMSE"] < min(reg_metrics(Y[s_], PRED[l][s_])["RMSE"] for l in ("linreg_lags", "linreg_ind")) for s_ in SETS}
print("\n¿El mejor modelo profundo", DL_BEST, "tiene menor RMSE que la regresión lineal?", dl_vs_lin)
""")
code(r"""
te = SETS["TE"]
SIMS = {}
for name in sorted({FINAL, DL_BEST}):
    ok = PRED[name]["TE"] > RULES[name]["delta"]
    if RULES[name]["sma_filter"]:
        ok &= (te.close > te.sma20).to_numpy()
    sig = pd.Series(np.where(ok, "LONG", "CASH"), index=te.index)
    SIMS[name] = (sig,) + simulate(te, sig)
    SIMS[name + " (0% fee, MEXC)"] = (sig,) + simulate(te, sig, fee=0.0)
SIMS["comprar y mantener"] = (None,) + simulate(te, pd.Series("LONG", index=te.index))
SIMS["efectivo"] = (None, None, {"rend_neto": 0.0, "max_dd": 0.0, "operaciones": 0, "win_rate": np.nan, "costos": 0.0, "patrimonio_final": CAPITAL})
RES_SIM = pd.DataFrame({k: v[2] for k, v in SIMS.items()}).T
print(RES_SIM.round(4).to_string())
fig, ax = plt.subplots(figsize=(12, 4))
for k, v in SIMS.items():
    if v[1] is not None:
        ax.plot(pd.to_datetime(v[1].fecha), v[1].patrimonio, label=k)
ax.axhline(CAPITAL, c="gray", ls=":", label="efectivo"); ax.legend(); ax.set_title("Patrimonio en TE (100,000 USDT, ejecución en la apertura de t+1)")
plt.tight_layout(); plt.savefig(OUT / "figuras/simulacion_TE.png", dpi=120); plt.show()

sig_f = SIMS[FINAL][0]
pred_te = pd.DataFrame({"fecha_emision": te.index.date, "fecha_objetivo": te.target_date.dt.date, "rendimiento_real": Y["TE"],
                        "rendimiento_estimado": PRED[FINAL]["TE"], "error": PRED[FINAL]["TE"] - Y["TE"], "senal": sig_f.values})
pred_te.to_csv(OUT / "resultados/predicciones_TE.csv", index=False)
SIMS[FINAL][1].to_csv(OUT / "resultados/simulacion.csv", index=False)
pred_te.head()
""")
code(r"""
# Pronóstico del siguiente periodo NO observado (valor real vacío hasta observarse)
last = feats.dropna(subset=LAGS + IND).iloc[[-1]]
cut_date = last.index[0]
def predict_last(name):
    kind, var = name.split("_")
    if kind == "linreg":
        return float((lin if var == "lags" else lin_i).predict((sx_lags if var == "lags" else sx_all).transform(last[LAGS if var == "lags" else ALL]))[0])
    if kind == "mlp":
        X, sx = (XIN if var == "lags" else XIN_I)["mlp"]; model = FINAL8["mlp"][0] if var == "lags" else MODELS11["mlp"][0]
        xin = sx.transform(last[LAGS if var == "lags" else ALL])
    else:
        vars_ = ["ret"] if var == "lags" else ["ret", "dist_sma20", "vol20"]
        X, sx = (XIN if var == "lags" else XIN_I)["lstm"]; model = FINAL8["lstm"][0] if var == "lags" else MODELS11["lstm"][0]
        base = feats.dropna(subset=vars_).copy(); base[vars_] = sx.transform(base[vars_]); xin = seq_tensor(base, last.index, vars_)
    info = MODELS11[kind][1] if var == "ind" else None
    ym, ys = Y["TR"].mean(), Y["TR"].std()
    with torch.no_grad():
        return float(model(torch.tensor(xin, dtype=torch.float32, device=DEVICE)).cpu().numpy()[0] * ys + ym)
fc = predict_last(FINAL)
ok = fc > RULES[FINAL]["delta"] and (not RULES[FINAL]["sma_filter"] or last.close.iloc[0] > last.sma20.iloc[0])
fut = pd.DataFrame([{"fecha_corte": cut_date.date(), "proximo_periodo": (cut_date + pd.Timedelta(days=1)).date(), "modelo": FINAL,
                     "rendimiento_estimado": fc, "senal": "LONG" if ok else "CASH", "rendimiento_real": None}])
fut.to_csv(OUT / "resultados/prediccion_futura.csv", index=False); fut
""")

md("## 9. Exportación: modelo, escalador, contrato de variables y registros")
code(r"""
def export_for_backend(name, folder):
    # Formato que carga backend/app/model.py (inferencia NumPy) + prueba de contrato check_x/check_y.
    kind, var = name.split("_"); folder = OUT / "modelos" / folder; folder.mkdir(parents=True, exist_ok=True)
    w, meta = {}, {"version": f"v1-{kind}-{var}-t4-{dt.date.today():%Y%m%d}", "feature_contract": "v1", "symbol": SYMBOL,
                    "signal_rule": RULES.get(name, {"delta": DELTAS[0], "sma_filter": False}), "trained_on": f"{ENV['gpu'] or 'CPU (prueba)'} · torch {torch.__version__}",
                    "splits": {k: {"inicio": str(v.index[0].date()), "fin": str(v.index[-1].date()), "n": len(v)} for k, v in SETS.items()},
                    "config_hash": CONGELADO["hash"], "seed": SEED, "source": DESCARGA["fuente"]}
    ym, ys = float(Y["TR"].mean()), float(Y["TR"].std())
    if kind == "linreg":
        model_, sx = (lin, sx_lags) if var == "lags" else (lin_i, sx_all)
        # la regresión se ajustó sobre y sin estandarizar -> y_mean=0, y_scale=1
        w["dense0_W"], w["dense0_b"] = model_.coef_.reshape(-1, 1), np.array([model_.intercept_])
        meta.update(kind="linear", n_dense=1, features=LAGS if var == "lags" else ALL, x_mean=sx.mean_.tolist(), x_scale=sx.scale_.tolist(), y_mean=0.0, y_scale=1.0)
        cols = meta["features"]; cx = SETS["TE"][cols].to_numpy()[:8]; cy = model_.predict(sx.transform(SETS["TE"][cols]))[:8]
    elif kind == "mlp":
        model_ = FINAL8["mlp"][0] if var == "lags" else MODELS11["mlp"][0]; sx = (XIN if var == "lags" else XIN_I)["mlp"][1]
        lins = [m_ for m_ in model_.net if isinstance(m_, nn.Linear)]
        for i, l_ in enumerate(lins):
            w[f"dense{i}_W"], w[f"dense{i}_b"] = l_.weight.detach().cpu().numpy().T, l_.bias.detach().cpu().numpy()
        meta.update(kind="mlp", n_dense=len(lins), features=LAGS if var == "lags" else ALL, x_mean=sx.mean_.tolist(), x_scale=sx.scale_.tolist(), y_mean=ym, y_scale=ys)
        cx = SETS["TE"][meta["features"]].to_numpy()[:8]; cy = PRED[name]["TE"][:8]
    else:
        model_ = FINAL8["lstm"][0] if var == "lags" else MODELS11["lstm"][0]; sx = (XIN if var == "lags" else XIN_I)["lstm"][1]
        vars_ = ["ret"] if var == "lags" else ["ret", "dist_sma20", "vol20"]
        L = model_.lstm
        w["lstm_kernel"] = L.weight_ih_l0.detach().cpu().numpy().T          # compuertas i,f,g,o (mismo orden que Keras)
        w["lstm_recurrent"] = L.weight_hh_l0.detach().cpu().numpy().T
        w["lstm_bias"] = (L.bias_ih_l0 + L.bias_hh_l0).detach().cpu().numpy()
        w["head0_W"], w["head0_b"] = model_.head.weight.detach().cpu().numpy().T, model_.head.bias.detach().cpu().numpy()
        meta.update(kind="lstm", n_head=1, seq_vars=vars_, x_mean=sx.mean_.tolist(), x_scale=sx.scale_.tolist(), y_mean=ym, y_scale=ys)
        base = feats.dropna(subset=vars_); cx = seq_tensor(base, SETS["TE"].index[:8], vars_); cy = PRED[name]["TE"][:8]
    w["check_x"], w["check_y"] = cx, cy
    np.savez(folder / "model.npz", **w)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    with open(folder / "scaler.pkl", "wb") as fh:
        pickle.dump(sx, fh)
    if kind != "linreg":
        torch.save(model_.state_dict(), folder / "model_torch.pt")
    print("exportado", name, "→", folder)

export_for_backend(FINAL, "final")
export_for_backend(DL_BEST, "dl_best")
pd.DataFrame(EXPERIMENTOS).to_csv(OUT / "resultados/experimentos.csv", index=False)
pd.DataFrame(BITACORA, columns=["fecha", "etapa", "error", "causa", "accion_correctiva", "verificacion"]).to_csv(OUT / "resultados/bitacora_errores.csv", index=False)
RES_SIM.to_csv(OUT / "resultados/resumen_simulacion_TE.csv")
HIST.to_csv(OUT / "resultados/resultados_historicos.csv")
zpath = Path("/content/entrega_dl_finanzas.zip" if IN_COLAB else "entrega_dl_finanzas.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p_ in OUT.rglob("*"):
        if p_.is_file():
            z.write(p_, p_.relative_to(OUT))
print("ZIP:", zpath, f"({zpath.stat().st_size/1e6:.1f} MB)")
if IN_COLAB:
    from google.colab import files
    files.download(str(zpath))
""")

md("## 10. Resumen automático y reflexión por capítulo")
code(r"""
print("=" * 70)
print(f"Activo: {SYMBOL} (MEXC, UTC) · {DESCARGA['periodo']} · {AUDITORIA['filas_finales']} velas válidas")
print("Particiones:\n" + PARTICIONES[["inicio (emisión)", "fin (emisión)", "ejemplos"]].to_string())
print(f"Modelo final (VA): {FINAL} | Mejor DL: {DL_BEST} | Regla: {RULES[FINAL]}")
print("RMSE TE:", {m_: round(reg_metrics(Y['TE'], PRED[m_]['TE'])['RMSE'], 6) for m_ in FINALES}, "| cero:", round(reg_metrics(Y['TE'], np.zeros_like(Y['TE']))['RMSE'], 6))
print("¿DL < regresión lineal en RMSE?", dl_vs_lin)
print("Simulación TE:\n" + RES_SIM[["rend_neto", "max_dd", "operaciones", "win_rate", "costos"]].round(4).to_string())
print("Dispositivo de entrenamiento:", ENV["device"], ENV["gpu"])
print("=" * 70)
""")
md("""
### Reflexión (complete con sus palabras usando las cifras de arriba)
- **Cap. 7 — Qué hicimos / por qué / evidencia / qué cambiaríamos:** auditoría y partición temporal por fecha del objetivo; el pronóstico cero es una referencia muy exigente para rendimientos diarios.
- **Cap. 8:** MLP y LSTM en T4 con parada temprana; comparar las curvas TR/VA y la variación entre semillas.
- **Cap. 9:** ventanas expansivas: ¿el resultado se sostiene en las tres ventanas? Umbral elegido en VA.
- **Cap. 10:** entorno con estado, acciones, recompensa neta de costos y comprobación anti-explotación; base para DQN/DDQN.
- **Cap. 11:** ¿los indicadores aportan información o solo complejidad? (tabla A vs B) ¿El filtro SMA20 reduce drawdown? (C vs D).
- **Conclusión sobre la hipótesis técnica:** se acepta/rechaza según si `dist_sma20` y `vol20` reducen el RMSE de VA del mismo modelo.
- **Límites:** un solo activo y un solo periodo de prueba; los costos son un supuesto académico; resultados pasados no garantizan resultados futuros.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("DL_Finanzas_BTCUSDT_MEXC.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook escrito en", out, "con", len(cells), "celdas")
