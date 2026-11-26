# Bot Financiero DL: agente de trading Paper con Deep Learning sobre datos reales de MEXC

> **Autor:** Vicente Rafael Reyes Vieyra · Proyecto de **Deep Learning para Finanzas** (Prof. Rubén Miguel Rodríguez Rangel).
> El proyecto va del pronóstico de rendimientos (capítulos 7 a 11 de *Deep Learning for Finance*, S. Kaabar) a un **agente de IA** que lee datos reales del par **BTC/USDT** en **MEXC**, consulta un modelo predictivo y un **LLM local (Ollama)**, pasa controles de riesgo deterministas y opera **solo en modo Paper** (dinero virtual), mostrando **operaciones y PnL** en un panel React + TypeScript.

**Las órdenes reales están deshabilitadas por diseño.** No existe código que envíe órdenes a MEXC; la ruta `/api/real/*` responde `403`. No se necesita ninguna clave de MEXC porque solo se consumen datos públicos.

---

## Índice
1. [Qué hace el sistema](#1-qué-hace-el-sistema)
2. [Arquitectura](#2-arquitectura)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Instalación y ejecución paso a paso](#4-instalación-y-ejecución-paso-a-paso)
5. [Notebook de Colab (T4): capítulos 7 a 11](#5-notebook-de-colab-t4-capítulos-7-a-11)
6. [El agente: ciclo, riesgo y Paper](#6-el-agente-ciclo-riesgo-y-paper)
7. [API del backend](#7-api-del-backend)
8. [Manejo de errores y pruebas](#8-manejo-de-errores-y-pruebas)
9. [Resultados](#9-resultados)
10. [Versiones y avances](#10-versiones-y-avances)
11. [Correspondencia con las rúbricas](#11-correspondencia-con-las-rúbricas)
12. [Seguridad y secretos](#12-seguridad-y-secretos)
13. [Limitaciones y trabajo futuro](#13-limitaciones-y-trabajo-futuro)

---

## 1. Qué hace el sistema

| Pieza | Qué hace |
|---|---|
| **Datos** | Descarga velas diarias **cerradas** de BTC/USDT desde la API pública de MEXC (`/api/v3/klines`, UTC, desde 2020-01-01, más de 2,100 velas). Excluye la vela abierta, audita los datos (duplicados, orden, huecos, no finitos y precios ≤ 0) y no rellena ni inventa volumen. |
| **Modelo** | Pronostica el rendimiento del día siguiente $y_t = P_{t+1}/P_t - 1$ con los 20 rendimientos más recientes, más `dist_sma20` y `vol20` (contrato de variables `v1`). Compara el pronóstico cero, la regresión lineal, MLP y LSTM. |
| **Señal** | **Long** si el pronóstico supera un umbral δ (elegido en VA); **Short** si es menor que −δ (modo Long/Short, opcional); en otro caso, efectivo. Cada operación registra su **Entry** (precio de entrada) y su PnL al cerrarse. |
| **Agente** | Repite el ciclo: velas de MEXC, variables, pronóstico, propuesta (LLM local o regla del modelo), validación del JSON, riesgo determinista, ejecución Paper, SQLite y UI. |
| **Panel** | Precio en vivo de MEXC por **WebSocket**, **gráfico de velas** con SMA20, Bandas de Bollinger y entradas/salidas, estados de carga y error, pronóstico, indicadores, acción propuesta y motivo, estado de riesgo, posición, patrimonio, **operaciones con PnL**, curva de patrimonio contra comprar y mantener, comparación de corridas, bitácora de errores y botones **Iniciar / Pausar / Reanudar / Detener / Liquidar**. |

## 2. Arquitectura

```mermaid
flowchart LR
  UI["React + TypeScript<br/>(Vite, :5173)"] -- "REST /api/* (X-API-Key)" --> API["Backend FastAPI<br/>autenticado (:8000)"]
  API -- "WebSocket /ws: ticker cada 5 s, decisiones, estado, errores" --> UI
  API --> MX["MEXC API pública<br/>klines 1d · ticker · exchangeInfo"]
  API --> MOD["Modelo predictivo versionado<br/>model.npz + meta.json (escalador)"]
  API --> GW["Gateway LLM<br/>(misma API key)"] --> OL["Ollama LOCAL<br/>llama3.2:3b @127.0.0.1:11434"]
  MOD --> RISK["Motor de riesgo<br/>determinista"]
  OL -->|"JSON {action, symbol, candle_date, reason}"| VAL["Validación de JSON"] --> RISK
  RISK --> PAPER["Motor Paper<br/>saldo · costos · precisión · deslizamiento"]
  PAPER --> DB[("SQLite<br/>velas · decisiones · órdenes · patrimonio · eventos")]
  DB --> API
  NB["Notebook Colab T4<br/>caps. 7-11"] -. "exporta modelo y escalador" .-> MOD
```

- **El LLM nunca recibe claves y no puede saltarse el riesgo.** Solo propone una acción de una lista permitida; el motor de riesgo decide.
- **Ollama corre local** y solo el backend puede consultarlo; Ollama no trae autenticación por defecto, así que la capa de API key está implementada en el backend.
- **La inferencia del modelo no necesita TensorFlow ni PyTorch en el servidor.** El notebook exporta los pesos y `backend/app/model.py` reproduce la red en NumPy. Al cargar el modelo, el backend comprueba que reproduce 8 predicciones de referencia del entrenamiento (prueba de contrato).

## 3. Estructura del repositorio

```
bot-financiero-dl/
├── README.md                      ← este archivo
├── .gitignore                     ← excluye .env, bases de datos y node_modules
├── notebook/
│   ├── DL_Finanzas_BTCUSDT_MEXC.ipynb   ← práctica caps. 7-11 (ejecutar en Colab con T4)
│   └── build_notebook.py               ← genera el .ipynb (fuente versionable del notebook)
├── backend/
│   ├── .env.example               ← plantilla de configuración SIN secretos
│   ├── requirements.txt
│   ├── config/risk.json           ← límites de riesgo (el LLM no puede cambiarlos)
│   ├── models/current/            ← modelo en uso: MLP entrenada en Tesla T4 (model.npz, meta.json, scaler.pkl, model_torch.pt)
│   ├── models/v0_cpu/             ← modelo provisional anterior (referencia)
│   ├── app/
│   │   ├── main.py      ← API FastAPI, autenticación, gateway del LLM
│   │   ├── agent.py     ← ciclo del agente, control humano, recuperación tras reinicio
│   │   ├── paper.py     ← motor de riesgo determinista y simulador Paper
│   │   ├── llm.py       ← cliente Ollama y validación estricta del JSON
│   │   ├── mexc.py      ← cliente de datos públicos de MEXC (solo lectura)
│   │   ├── features.py  ← contrato de variables v1 (igual que en el notebook)
│   │   ├── model.py     ← carga del modelo y escalador, inferencia NumPy (MLP/LSTM/lineal)
│   │   ├── backtest.py  ← simulación vectorial y métricas
│   │   ├── config.py, db.py
│   ├── scripts/train_cpu_baseline.py   ← modelo provisional v0 en CPU
│   └── tests/test_controls.py          ← 23 pruebas automatizadas (controles, Long/Short, stop-loss, WebSocket)
├── frontend/                      ← React 18 + TypeScript + Vite
│   └── src/ (App.tsx, Charts.tsx, api.ts, styles.css)
├── data/raw, data/processed       ← datos originales y procesados (CSV)
├── reports/                       ← experimentos, umbrales, simulaciones exportadas
│   └── colab_t4/                  ← EVIDENCIA OFICIAL de Colab T4: resultados, figuras, modelos y checkpoints
└── docs/                          ← presentación ejecutiva y evidencias
```

## 4. Instalación y ejecución paso a paso

### Requisitos
| Herramienta | Versión probada | Para qué |
|---|---|---|
| Python | 3.14.4 (funciona con ≥ 3.11) | backend |
| Node.js / npm | 24.15 / 11.12 | frontend |
| Ollama | 0.40.x | LLM local |
| Modelo LLM | `llama3.2:3b` (~2 GB) | propuestas del agente |
| Git | 2.x | repositorio |
| Google Colab | GPU **NVIDIA T4** | entrenamiento profundo (obligatorio) |

### 4.1 Backend
```powershell
cd backend
python -m pip install -r requirements.txt
copy .env.example .env
python -c "import secrets;print(secrets.token_urlsafe(32))"
```
Pegue la clave generada en `AGENT_API_KEY=` dentro de `backend/.env`. Luego:
```powershell
python -m uvicorn app.main:app --port 8000
```
Documentación interactiva de la API: http://127.0.0.1:8000/docs

> Si `backend/models/current/` está vacío, ejecute `python -m scripts.train_cpu_baseline`, que descarga los datos de MEXC y entrena el modelo provisional v0, o copie la exportación del notebook (sección 5.3).

### 4.2 LLM local (Ollama)
```powershell
winget install Ollama.Ollama
ollama pull llama3.2:3b
ollama run llama3.2:3b "hola"
```
Ollama escucha en `127.0.0.1:11434` (solo local). Verifique la inferencia local desde el panel (**Gateway del LLM local**) o con `GET /api/llm/info`, que muestra la versión de Ollama y los modelos descargados.

### 4.3 Frontend
```powershell
cd frontend
npm install
npm run dev
```
Abra http://localhost:5173 y escriba la `AGENT_API_KEY`. La clave **no** está en el código del frontend: se guarda solo en `sessionStorage` de la pestaña.

### 4.4 Uso rápido
1. **Modo**: *Replay histórico* (repite velas pasadas, identificado como replay) o *Live MEXC* (procesa cada vela diaria nueva cuando cierra a las 00:00 UTC).
2. **Estrategia**: *Modelo sin LLM* (regla del modelo) o *Agente con LLM* (Ollama propone y el riesgo valida).
3. **Iniciar**. Las operaciones, el PnL y la curva de patrimonio se actualizan cada 2.5 s.
4. **Pausar / Reanudar / Detener**. *Detener* bloquea nuevas órdenes **sin liquidar**; para cerrar la posición use **Liquidar posición** (acción explícita).
5. Para la comparación que pide la rúbrica, corra replays con el mismo número de días en ambas estrategias. La tabla muestra rendimiento neto, comprar y mantener, drawdown, operaciones, Win Rate, acciones rechazadas y latencia del LLM.

## 5. Notebook de Colab (T4): capítulos 7 a 11

### 5.1 Cómo ejecutarlo
1. Suba `notebook/DL_Finanzas_BTCUSDT_MEXC.ipynb` a Google Colab.
2. *Entorno de ejecución → Cambiar tipo de entorno → **GPU T4***.
3. *Ejecutar todo*. Autorice Google Drive cuando lo pida; las salidas se guardan en `MyDrive/bot_financiero_dl/`.
4. Al final se descarga `entrega_dl_finanzas.zip` con datos, modelos, checkpoints, CSV y figuras.

**Si no detecta una T4, el notebook se detiene** y registra el incidente en la bitácora (regla del protocolo). La variable `ALLOW_CPU_TEST=1` existe solo para probar el código fuera de Colab; esa salida **no** es evidencia.

### 5.2 Contenido por capítulo
| Sección | Capítulo | Qué contiene |
|---|---|---|
| 0 | — | Evidencia T4: `nvidia-smi`, operación en la GPU, versiones, semilla y `evidencia_t4.txt` |
| 1 | 7 / 11 | Ficha del activo, descarga MEXC con metadatos, diccionario, auditoría, rendimientos extremos y gráficas de precio, rendimiento y volatilidad |
| 2 | 7 | Objetivo, partición TR/VA/TE por fecha del objetivo (70/15/15) con verificación, **fila de control de aprendizaje**, pronóstico cero y regresión lineal (MAE, RMSE y DA en TR y VA), hipótesis técnica |
| 3 | 8 | MLP y LSTM en T4 (forma `ejemplos × 20 × variables`), tres configuraciones por red, parada temprana con VA y restauración de pesos, curvas TR/VA, tres semillas (media ± desviación), checkpoints |
| 4 | 9 | Tres ventanas expansivas dentro de TR (escalador reajustado en cada una), cinco umbrales δ en VA, diferenciación fraccionaria |
| 5 | 10 | Entorno `TradingEnv`: estado, acciones, restricciones, recompensa neta de costos, **transición manual calculada**, prueba anti-explotación, DQN/DDQN |
| 6 | 11 | Indicadores documentados (fórmula, ventana, primera fecha válida), comparación **A rezagos vs B rezagos + indicadores**, redundancia, predicciones casi constantes, **C solo pronóstico vs D con filtro SMA20** |
| 7 | transversal | **Cuatro fallos provocados**: conjunto vacío, fecha duplicada, valor no válido y dimensiones incorrectas |
| 8 | 7 / 9 | Controles antes de TE, configuración congelada con hash, **evaluación única en TE**, simulación (100,000 USDT, 0.10% por lado, ejecución en la apertura de t+1), comprar y mantener, efectivo, `predicciones_TE.csv`, `prediccion_futura.csv` |
| 9 | — | Exportación: `experimentos.csv`, `bitacora_errores.csv`, `simulacion.csv`, modelo, escalador y zip |
| 10 | — | Resumen automático y plantilla de reflexión por capítulo |

### 5.3 Usar el modelo de Colab en el agente
Del zip, copie `modelos/final/` (o `modelos/dl_best/` para usar el mejor modelo profundo) a `backend/models/current/` y reinicie el backend. Al arrancar se verifica:
- el contrato de variables (`feature_contract = v1`);
- que la inferencia en NumPy reproduce las predicciones de PyTorch (`check_x` / `check_y`).

## 6. El agente: ciclo, riesgo y Paper

**Ciclo obligatorio** (`backend/app/agent.py`):
1. Consulta las velas **cerradas** de MEXC y valida activo, orden, duplicados, valores y antigüedad.
2. Calcula las variables con la **misma definición** del entrenamiento (`features.py`) y obtiene el pronóstico.
3. Con la estrategia LLM, consulta Ollama con **datos estructurados**: precio, SMA20, distancias, RSI, pronóstico, señal del modelo, posición y acciones permitidas.
4. Exige un JSON `{action, symbol, candle_date, reason}` y valida tipos, acción, símbolo y fecha. Si el JSON es inválido, hace un reintento; si vuelve a fallar, rechaza la propuesta y no opera.
5. Aplica el **riesgo determinista** (`paper.py`).
6. Ejecuta en **Paper** y guarda la decisión, la orden, el patrimonio y los eventos. Es **idempotente**: hay una sola decisión por vela y por corrida (`UNIQUE(run_id, candle_date)`).
7. La UI muestra el resultado y el agente espera la siguiente vela (live) o avanza (replay).

**Configuración de riesgo** (`backend/config/risk.json`, fijada antes de la prueba; solo se cambia con la API key y con el agente detenido):

| Parámetro | Valor | Significado |
|---|---|---|
| `initial_capital` | 50,000 USDT | capital inicial del agente |
| `fee_rate` | 0.0% | costo por compra o venta (protocolo, sección 15; MEXC 0 fees) |
| `slippage_bps` | 2 pb | **supuesto de deslizamiento**: la compra se ejecuta 0.02% más cara y la venta 0.02% más barata |
| `max_exposure` | 95% | máximo del patrimonio invertido en BTC |
| `max_daily_loss` | 5% | si el patrimonio cae 5% en el día, se bloquean compras |
| `max_drawdown_halt` | 25% | si el drawdown supera 25%, el agente se detiene (solo permite vender) |
| `max_data_age_hours` | 30 h | si la última vela cerrada es más antigua, se bloquean órdenes |
| `stop_loss_pct` | 0 (apagado; por corrida en `/api/agent/start`) | cierra la posición si el movimiento adverso desde la entrada supera el porcentaje |
| `allow_short` | sí (cada corrida elige) | permite el modo Long/Short en Paper, sin apalancamiento |
| `allowed_symbols` | BTCUSDT | cualquier otro símbolo se rechaza |

**Simulador Paper:** respeta saldo (no compra más que el efectivo incluyendo costos), **tamaño mínimo y precisión del par** (leídos de `exchangeInfo` de MEXC: paso 0.000001 BTC, precio con 2 decimales, mínimo 1 USDT), costos y momento de ejecución:
- En **replay**, la señal del cierre de *t* se ejecuta en la **apertura de t+1** y el patrimonio se valora al cierre de *t+1*.
- En **live**, se ejecuta al último precio de MEXC tras el cierre de la vela.

Las ejecuciones y los rechazos quedan registrados. Sin apalancamiento.

**Entry, Long y Short (v1.1).** Cada corrida elige *Solo largo* (por defecto, como pide el protocolo de la práctica) o *Long/Short* (casilla del panel o `allow_short: true`):

| Posición | Abre | Cierra | Cómo se valora |
|---|---|---|---|
| **Long** | `BUY` | `SELL` | efectivo + cantidad × precio |
| **Short** | `SHORT` (vende BTC prestado; el producto queda como efectivo de respaldo) | `COVER` (recompra) | efectivo − cantidad × precio |
| Efectivo | — | — | efectivo |

- **Entry:** al abrir se guarda el precio de entrada (`entry_price`), que el panel muestra junto a la posición. El PnL se calcula al cerrar: en un largo es venta − compra; en un corto es venta corta − recompra. Ambos incluyen costos y deslizamiento.
- **Sin apalancamiento:** la exposición absoluta nunca supera el 95% del patrimonio. No se permiten dos posiciones a la vez ni el giro directo de largo a corto; el giro toma dos velas, primero se cierra y luego se abre.
- **La regla de corto es simétrica a la de largo** (pronóstico < −δ, el mismo δ elegido en VA), sin parámetros nuevos.

## 7. API del backend

Todas las rutas requieren la cabecera `X-API-Key`, excepto `/api/health`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | vivo; confirma `real_orders: disabled` |
| GET | `/api/status` | estado del agente, cartera, última decisión, modelo y riesgo |
| GET | `/api/market` | ticker en vivo de MEXC, última vela cerrada, indicadores y **pronóstico del siguiente día** |
| POST | `/api/agent/start` | `{mode: replay\|live, strategy: model\|llm, replay_days, allow_short}` |
| WS | `/ws` | canal en vivo; el primer mensaje es `{"type":"auth","key":"..."}` (la clave no viaja en la URL). Empuja `ticker`, `decision`, `status` y `event` |
| POST | `/api/agent/pause` · `/resume` · `/stop` · `/liquidate` | control humano |
| GET | `/api/runs` · `/api/runs/{id}` | métricas por corrida (con comprar y mantener), patrimonio, decisiones y órdenes |
| GET | `/api/events` | bitácora de eventos y errores |
| PUT | `/api/config/risk` | cambiar límites (solo con el agente detenido) |
| GET | `/api/llm/info` · POST `/api/llm/query` | gateway del LLM local: versión, modelos y consulta con latencia |
| POST | `/api/faults/{tipo}` | demostración de errores: `mexc_down`, `stale_data`, `ollama_down`, `invalid_json`, `insufficient_balance`, `duplicate_event` |

## 8. Manejo de errores y pruebas

### 8.1 Los nueve errores del protocolo
| Error | Cómo se demuestra | Respuesta del sistema |
|---|---|---|
| API sin respuesta o con límite (429) | botón *API MEXC sin respuesta* o red caída | 3 reintentos con espera creciente (1, 2 y 4 s); después **pausa** el agente (`halted`) y registra el evento |
| Dato atrasado | botón *Dato atrasado* (modo live) | la orden se **bloquea** (`BLOCKED`) con la antigüedad en horas |
| Ollama detenido | cerrar Ollama o botón | 2 reintentos; luego pausa nuevas decisiones; no opera |
| JSON inválido | botón *JSON inválido* | propuesta **rechazada** y registrada; la acción final es HOLD |
| Clave incorrecta | entrar con otra clave o llamar sin cabecera | `401` y evento `auth` registrado |
| Saldo insuficiente | botón *Saldo insuficiente* | orden **rechazada** (`REJECTED`) con el motivo |
| Evento duplicado | botón *Evento duplicado* | se ignora por idempotencia; no se duplican operaciones |
| Reinicio del backend | detener uvicorn y volver a iniciarlo | la corrida se recupera de SQLite **en pausa**, sin duplicar velas ya procesadas |
| Botón de detención | *Detener* | bloquea nuevas órdenes y muestra la posición abierta; *Liquidar* es una acción aparte |

### 8.2 Pruebas automatizadas
```powershell
cd backend
python -m pytest -q tests
```
Resultado actual: **23 passed**. Cubren la validación de datos (vacío, duplicado, no finito, ventana insuficiente), la exclusión de la vela abierta, el contrato del modelo, el riesgo (saldo, dato atrasado, sin cortos ni recompras, símbolo, detención, pérdida diaria), el JSON del LLM, Ollama caído, la idempotencia, la recuperación tras reinicio, MEXC caído, la detención y la API sin clave o con clave incorrecta.

## 9. Resultados

> **Evidencia oficial: notebook ejecutado en Google Colab con NVIDIA Tesla T4** (2026-11-26 16:59 UTC; Python 3.13.16, torch 2.14.1+cu130, CUDA 13.0, semilla 42, configuración congelada `7414f5e18dfc513e`). Todos los entrenamientos de MLP y LSTM registran `dispositivo = cuda:0`. Archivos en `reports/colab_t4/`: `evidencia_t4.txt` (salida de `nvidia-smi`), `experimentos.csv`, `predicciones_TE.csv`, `prediccion_futura.csv`, `simulacion.csv`, `bitacora_errores.csv`, figuras, modelos y checkpoints.

### 9.1 Particiones (BTC/USDT, MEXC, diario UTC)
| Conjunto | Emisión desde | Emisión hasta | Último objetivo | Ejemplos |
|---|---|---|---|---|
| TR (70%) | 2020-02-09 | 2024-11-09 | 2024-11-10 | 1,736 |
| VA (15%) | 2024-11-10 | 2025-11-16 | 2025-11-17 | 372 |
| TE (15%) | 2025-11-17 | 2026-11-24 | 2026-11-25 | 373 |

### 9.2 Modelos en la T4
**Capítulo 8.** Selección por RMSE de VA: MLP con 64 neuronas ocultas (0.022747) y LSTM con 32 unidades (0.022827). Variación entre semillas (42, 7 y 2024) en RMSE de VA: MLP de 0.022747 a 0.022988; LSTM de 0.022794 a 0.022837. Cada entrenamiento tardó de 1.5 a 6 s en la T4.

| Modelo (versión final) | RMSE TR | RMSE VA | RMSE TE | MAE TE | DA TE |
|---|---|---|---|---|---|
| Pronóstico cero | 0.034337 | 0.022833 | 0.023122 | 0.015917 | — |
| Regresión lineal, rezagos | 0.033925 | 0.022818 | 0.023510 | 0.016425 | 45.0% |
| Regresión lineal, rezagos + indicadores | 0.033880 | **0.022739** | 0.023403 | 0.016405 | 44.0% |
| MLP, rezagos (**mejor DL**, en el agente) | 0.033800 | 0.022747 | 0.023351 | 0.016409 | 45.8% |
| MLP, rezagos + indicadores | 0.033510 | 0.023021 | 0.023480 | 0.016515 | 44.0% |
| LSTM, rezagos | 0.034200 | 0.022837 | 0.023181 | 0.016034 | 48.8% |
| LSTM, rezagos + indicadores | 0.034045 | 0.022790 | **0.023060** | **0.015984** | **49.3%** |

- **¿El deep learning supera a la regresión lineal?** En **TE, sí**: la MLP (0.023351) y las dos LSTM (0.023181 y 0.023060) tienen menor RMSE que las dos regresiones (0.023510 y 0.023403). En **VA** la regresión con indicadores gana por 0.000008, un empate práctico. Por la regla del protocolo, el modelo "final" elegido con VA es la regresión; el **agente usa el mejor modelo profundo (MLP)**, exportado por la regla fijada antes de abrir TE.
- **¿Supera al pronóstico cero?** En TE solo la LSTM con indicadores lo hace (0.023060 contra 0.023122). En rendimientos diarios la señal es pequeña frente al ruido.
- **Hipótesis técnica (capítulo 11):** los indicadores mejoran el RMSE de VA en la regresión (0.022818 → 0.022739) y en la LSTM (0.022837 → 0.022790), pero lo empeoran en la MLP (0.022747 → 0.023021). La hipótesis se acepta para la regresión y la LSTM, no en general.
- **Ventanas expansivas (capítulo 9):** en las 3 ventanas el pronóstico cero tuvo el menor RMSE o quedó prácticamente empatado. La ventaja de los modelos no es robusta en el tiempo.
- **Regla elegida en VA:** comprar si el pronóstico es mayor que δ = 0.20%, sin filtro SMA20. En VA, la MLP con esa regla rindió +34.2% (con 0.10% por lado).
- **Simulación en TE (100,000 USDT, ejecución en la apertura de t+1):**

| Estrategia | Rend. neto | Max DD | Operaciones | Win Rate | Costos |
|---|---|---|---|---|---|
| MLP, 0.10% por lado | −22.7% | 36.9% | 97 | 46.4% | 15,909 |
| MLP, 0% (MEXC) | **−6.2%** | 31.1% | 97 | 49.5% | 0 |
| Regresión, 0.10% por lado | −31.6% | 35.4% | 81 | 43.2% | 12,539 |
| Comprar y mantener | −35.2% | 39.5% | 1 | — | 165 |
| Efectivo | 0.0% | 0.0% | 0 | — | 0 |

- **Pronóstico futuro:** con corte el 2026-11-25, el rendimiento estimado para el 2026-11-26 es +0.065%. La señal es EFECTIVO (no supera δ). El valor real queda vacío hasta observarse.
- **Fallos provocados:** conjunto vacío, fecha duplicada, valor no válido y dimensiones incorrectas, todos detenidos con excepciones específicas (`bitacora_errores.csv`).

### 9.2b Modelo provisional v0 (CPU, histórico)
Antes de tener la T4, el agente usó un modelo provisional (`backend/models/v0_cpu/`). Se conserva como referencia.
- En VA, la **regresión lineal con rezagos e indicadores** tuvo el menor RMSE (0.02321), por delante de los MLP de scikit-learn. Por eso es el modelo provisional. Regla elegida en VA: δ = 0.20% sin filtro SMA20.
- **TE, simulación con 100,000 USDT y 0.10% por lado:** estrategia **−27.8%** frente a comprar y mantener **−37.5%** (BTC cayó de ~95k a ~60k USDT en TE). Drawdown máximo: 33.5% contra 39.5%. 79 operaciones, Win Rate 45.6%, costos 12,488 USDT.
- **Agente Paper, replay de 90 días (2026-08-28 → 2026-11-25), 50,000 USDT, 0% de costo:** **+0.14%** (PnL +70.59 USDT) frente a comprar y mantener **−24.7%**. 90 ciclos, 45 operaciones, Win Rate 45.5%, drawdown máximo 11.8%.

### 9.3 Agente Paper con el modelo de la T4 (`v1-mlp-lags-t4-20261126`)
Replays con 50,000 USDT, 0% de costo y 2 pb de deslizamiento, sobre velas reales de MEXC (periodo dentro de TE, fuera de TR y VA):

| Corrida | Periodo | Rend. neto | PnL (USDT) | Max DD | Operaciones | Win Rate | Comprar y mantener |
|---|---|---|---|---|---|---|---|
| Modelo sin LLM, 90 días | 2026-08-28 → 2026-11-25 | **+7.03%** | **+3,513.67** | 4.9% | 51 | 52.0% | −24.66% |
| Modelo sin LLM, 60 días | 2026-09-27 → 2026-11-25 | **+2.55%** | +1,276.36 | 4.9% | 35 | 47.1% | −28.41% |
| Agente con LLM (prompt v2), 60 días | 2026-09-27 → 2026-11-25 | −10.68% | −5,340.63 | 17.5% | 37 | 38.9% | −28.41% |

**Conexión Live con MEXC (no replay), 2026-11-26:** el agente leyó la última vela cerrada (2026-11-25, cierre 60,454.44 USDT). El modelo de la T4 pronosticó +0.259% (señal LONG), el riesgo aprobó la compra (`APPROVED`) y el simulador Paper ejecutó **BUY 0.785539 BTC a 60,467.97 USDT**, el precio en vivo de MEXC con 2 pb de deslizamiento. Después se pulsó *Detener*: se bloquean nuevas órdenes y la posición queda abierta, como define el protocolo.

Latencia media del LLM: 6,145 ms por decisión (Ollama 0.40.18, `llama3.2:3b`, CPU local). 0 acciones rechazadas por riesgo.

### 9.3b Primera comparación con el modelo v0 (CPU)
Replay de 60 días (2026-09-27 → 2026-11-25), 50,000 USDT, 0% de costo, 2 pb de deslizamiento, mismo modelo v0. Ollama local con `llama3.2:3b` (Ollama 0.40.18, CPU, sin nube).

| Estrategia | Rend. neto | PnL (USDT) | Max DD | Operaciones | Win Rate | Latencia LLM | Rechazadas |
|---|---|---|---|---|---|---|---|
| Modelo sin LLM | **−7.73%** | −3,863.65 | 11.8% | 29 | 42.9% | — | 0 |
| Agente con LLM, prompt v1 | −19.36% | −9,678.08 | 21.8% | 60 | 40.0% | 7,226 ms | 0 |
| Agente con LLM, prompt v2 | −10.24% | −5,118.50 | 12.4% | 33 | 37.5% | 5,921 ms | 0 |
| Comprar y mantener | −28.41% | −14,205 | — | 1 | — | — | — |

**¿El LLM aporta valor?** En este periodo, **no**. Con el prompt v1, el modelo de 3B interpretó mal los indicadores (llamó "sobrecompra" a un RSI de 24) y cambió de posición todos los días. El prompt v2 agrega definiciones correctas de RSI, `dist_sma20` y HOLD, y la acción que propone el modelo (`model_action`). Con eso bajó a 33 operaciones y a −10.2%, pero sigue por debajo del modelo sin LLM. Las tres estrategias activas perdieron menos que el mercado. El LLM sirve como capa de explicación (motivo legible por decisión), no como fuente de rendimiento. Se hizo un solo cambio de prompt y se reportan las dos versiones.

Con el modelo de la T4 se repite el patrón: sin LLM +2.55%, con LLM −10.68%. El LLM de 3B no aporta rendimiento en estos periodos.

### 9.4 Prueba del notebook (CPU, solo para validar el código)
Antes de Colab, el notebook se ejecutó completo en CPU con `ALLOW_CPU_TEST=1` (118 s). Esa corrida **no** es la evidencia oficial, que debe salir de la T4.
- Particiones del notebook: TR 2020-02-09 → 2024-11-09 (1,736) · VA 2024-11-10 → 2025-11-16 (372) · TE 2025-11-17 → 2026-11-24 (373). Arrancan 19 días después que en el modelo v0 porque todas las versiones usan el mismo periodo evaluable que la LSTM con indicadores.
- RMSE de VA: regresión lineal con indicadores 0.022739, MLP con rezagos 0.022747, LSTM con indicadores 0.022790, cero 0.022833. Entre la regresión y la MLP hay un empate práctico (diferencia de 0.000008).
- En TE, la MLP tuvo menor RMSE que la regresión lineal (0.023351 contra 0.023403) y la LSTM con indicadores fue la única por debajo del pronóstico cero (0.023060 contra 0.023122).
- Simulación en TE con la MLP: −22.7% con 0.10% por lado y −6.2% con 0% (MEXC), frente a −35.3% de comprar y mantener.

### 9.5 Entry, Long y Short (v1.1)
**Replays con el modelo de la T4** (50,000 USDT, 0% de costo, 2 pb de deslizamiento, mismos periodos):

| Corrida | Periodo | Rend. neto | PnL (USDT) | Max DD | Operaciones (largos/cortos cerrados) | Win Rate | Comprar y mantener |
|---|---|---|---|---|---|---|---|
| Solo largo, 90 días | 2026-08-28 → 2026-11-25 | +7.03% | +3,513.67 | 4.9% | 51 | 52.0% | −24.66% |
| **Long/Short, 90 días** | 2026-08-28 → 2026-11-25 | **+11.08%** | **+5,539.64** | 4.5% | 57 (24 / 4) | 50.0% | −24.66% |
| Solo largo, 60 días | 2026-09-27 → 2026-11-25 | +2.55% | +1,276.36 | 4.9% | 35 | 47.1% | −28.41% |
| **Long/Short, 60 días** | 2026-09-27 → 2026-11-25 | **+4.05%** | **+2,025.87** | 4.0% | 39 (17 / 2) | 47.4% | −28.41% |

**Métricas de riesgo por corrida** (90 días, fórmulas de los apuntes: Sharpe y Sortino anualizados a 365 días con tasa libre de riesgo 0, y VaR/CVaR históricos a 1 día al 95%). Las tres corridas se ven en el panel:

| Corrida (90 días) | Rend. neto | Sharpe | Sortino | Volatilidad anual | VaR 95% | CVaR 95% | Stop-loss |
|---|---|---|---|---|---|---|---|
| Solo largo | +7.03% | 1.42 | 2.60 | 20.9% | 1.79% | 2.27% | — |
| Long/Short | +11.08% | 2.11 | 4.25 | 21.2% | 1.66% | 2.11% | — (con 5% no se activa) |
| Long/Short + stop 2% | +10.51% | 2.01 | 4.05 | 21.2% | 1.66% | 2.11% | 2% (2 activaciones) |

**Año completo de prueba TE** (`backend/scripts/eval_long_short_te.py` → `reports/long_short_TE.json`; 100,000 USDT, ejecución en la apertura de t+1). La simulación solo largo reproduce exactamente las cifras del notebook de la T4 (−22.71% y −6.17%), lo que valida el cálculo:

| Estrategia | Costo | Rend. neto | Max DD | Largos / cortos | PnL largos | PnL cortos | Win Rate |
|---|---|---|---|---|---|---|---|
| Solo largo | 0.10% | −22.71% | 36.9% | 97 / 0 | −22,713 | — | 46.4% |
| Long/Short | 0.10% | −14.87% | 36.7% | 92 / 18 | −14,235 | −636 | 46.4% |
| Solo largo | 0% (MEXC) | −6.17% | 31.1% | 97 / 0 | −6,166 | — | 49.5% |
| **Long/Short** | **0% (MEXC)** | **+6.07%** | 30.0% | 92 / 18 | +3,169 | +2,903 | 49.1% |
| Comprar y mantener | 0.10% | −35.25% | 39.5% | — | — | — | — |

De los 373 días de TE, la señal fue LONG 163 días, SHORT 37 y efectivo 173. **Declaración de contaminación (protocolo, sección 3):** TE ya se había usado en la evaluación oficial solo largo. La regla corta se agregó por el criterio 5 de la rúbrica del proyecto (Entry, Long, Short), no por estos resultados: es la regla simétrica, con el mismo δ y ningún parámetro ajustado. Aun así es un análisis posterior, no una evaluación intacta; la confirmación requiere un periodo nuevo.

### 9.6 Lectura honesta
- La estrategia pierde menos que comprar y mantener en mercados bajistas porque pasa a efectivo, pero **en TE no obtuvo rentabilidad positiva**. El protocolo pide "+20%"; la rúbrica aclara que "no se exige obtener ganancias ni superar al modelo sencillo". Se reportan las cifras reales, sin ajustar el sistema después de ver TE.
- Superar al pronóstico cero en RMSE con rendimientos diarios es difícil: la señal útil es pequeña frente al ruido.
- En la T4, el deep learning supera a la regresión lineal en TE pero empata con ella en VA. En los replays recientes el agente con la MLP fue positivo (+7.03% en 90 días, +2.55% en 60) mientras el mercado caía de 25% a 28%. En el TE completo (un año bajista, −35%) perdió −6.2% con 0% de costo. No alcanza el +20% en TE, y así se reporta.

## 10. Versiones y avances

| Versión | Fecha | Avance | Contenido |
|---|---|---|---|
| **v0.1** | 2026-11-26 | Avance 1 (cap. 7) | Descarga MEXC paginada y auditoría, contrato de variables v1, partición TR/VA/TE por fecha del objetivo, pronóstico cero y regresión lineal |
| **v0.2** | 2026-11-26 | Avance 2 (cap. 8) | MLP y LSTM en PyTorch con parada temprana, tres configuraciones, tres semillas y evidencia de T4 en el notebook |
| **v0.3** | 2026-11-26 | Avance 3 (cap. 9) | Ventanas expansivas, umbrales δ en VA, simulación con ejecución en la apertura de t+1 |
| **v0.4** | 2026-11-26 | Avance 4 (cap. 10) | `TradingEnv` con estado, acciones, recompensa neta, transición manual y prueba anti-explotación |
| **v0.5** | 2026-11-26 | Avance 5 (cap. 11) | Indicadores `dist_sma20` y `vol20`, comparación A/B y C/D, TE único, pronóstico futuro y cuatro fallos provocados |
| **v0.6** | 2026-11-26 | Proyecto final | Backend FastAPI autenticado, cliente MEXC, motor de riesgo, Paper, SQLite, modelo provisional v0 en CPU |
| **v0.7** | 2026-11-26 | Proyecto final | Gateway y cliente Ollama local con validación JSON; estrategias con y sin LLM |
| **v0.8** | 2026-11-26 | Proyecto final | Panel React + TypeScript: precio en vivo, pronóstico, operaciones, PnL, comparación y bitácora |
| **v0.9** | 2026-11-26 | Proyecto final | Inyección de fallos, 16 pruebas automatizadas, recuperación tras reinicio |
| **v0.9.1** | 2026-11-26 | Proyecto final | Ollama local integrado (llama3.2:3b), prompt v2 con definiciones de indicadores y `model_action`, comparación sin LLM, con LLM y pasiva; notebook probado de punta a punta y exporta `lstm_best` |
| **v1.1** | 2026-11-26 | Rúbrica de la app | **Entry/Long/Short** en Paper (SHORT/COVER, sin apalancamiento), **WebSocket** `/ws` con ticker, decisiones y eventos en vivo, **gráfico de velas** con SMA20, Bollinger y entradas/salidas, estados de carga y error, diagrama REST + WebSocket, **stop-loss** por operación, **Sharpe/Sortino/VaR/CVaR** por corrida, 23 pruebas, autoevaluación `docs/RUBRICA_AUTOEVALUACION.md` |
| **v1.0** | 2026-11-26 | Entrega | Notebook ejecutado en Colab **Tesla T4**; modelo MLP de la T4 integrado al agente (`models/current`), evidencia en `reports/colab_t4/`, README y presentación con cifras reales |

## 11. Correspondencia con las rúbricas

> Autoevaluación detallada contra la **Rúbrica · Proyecto de App de Trading con IA** (5 criterios, 100 pts): [`docs/RUBRICA_AUTOEVALUACION.md`](docs/RUBRICA_AUTOEVALUACION.md).

**Práctica (100 pts):** Datos (§5.2 sec. 1) · TR/VA/TE (sec. 2 y 8) · Referencias y modelos (sec. 2 y 3) · T4 (sec. 0 y 3) · Validación y estudio técnico (sec. 3, 4 y 6) · Estimaciones (sec. 8) · Manejo de errores (sec. 7 y bitácora) · Agente y simulación (sec. 5 y 8) · Reproducibilidad (sec. 9 y 10).

**Proyecto final (100 pts):**

| Criterio | Evidencia en el repositorio |
|---|---|
| Integración y modelo | `models/current/meta.json` versionado con su escalador; Ollama local (`/api/llm/info`); UI React + TS |
| API privada y secretos | `require_key` en `main.py`, pruebas 401; `.env` fuera del repo, `.env.example` sin valores |
| Datos MEXC y Paper | `mexc.py` (velas cerradas y fechadas), `paper.py` (saldo, costos, precisión), `/api/real/*` → 403 |
| Ciclo del agente | `agent.py` (7 pasos), `llm.py` (JSON validado), bucle live con `LIVE_POLL_SECONDS` |
| Riesgo y control humano | `risk.json` + `check()`; botones Iniciar, Pausar, Reanudar, Detener y Liquidar |
| Errores y persistencia | 9 escenarios (§8), SQLite idempotente, tabla `events` |
| Evaluación | tabla de corridas: modelo sin LLM, agente con LLM y comprar y mantener |
| Entrega reproducible | este README, `.env.example`, `requirements.txt`, `package.json` |

## 12. Seguridad y secretos
- `AGENT_API_KEY` vive solo en `backend/.env`, que está en `.gitignore`. Nunca se registra en logs ni se devuelve en `/api/status`.
- El frontend no contiene secretos ni variables `VITE_*` con claves.
- Iniciar, detener, liquidar y cambiar la configuración también requieren la clave.
- No se usa ninguna clave de MEXC. Si en el futuro se usara una de solo lectura, no debe tener permisos de trading ni de retiro.
- Los datos externos (incluida la respuesta del LLM) se tratan como **datos**: el LLM no puede cambiar reglas ni pedir secretos.

## 13. Limitaciones y trabajo futuro
- Un solo activo (BTC/USDT) y un solo periodo de prueba; los costos y el deslizamiento son supuestos académicos.
- La frecuencia es diaria: en modo live solo hay una decisión nueva por día. Los 20 o más ciclos de la demostración se obtienen en **replay**, identificado como tal.
- Los cortos son simulados: en Paper no se modelan el costo del préstamo ni el *funding* de un margen real. Con 0.10% por lado el Long/Short sigue negativo en TE.
- Siguiente paso: convertir `TradingEnv` en el entorno de entrenamiento y entrenar **DQN/DDQN** con una prueba independiente nueva.

*Simulación educativa. No es asesoría de inversión.*
