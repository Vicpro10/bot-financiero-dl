# Autoevaluación: Rúbrica · Proyecto de App de Trading con IA

**Proyecto:** Bot Financiero DL · Vicente Rafael Reyes Vieyra · versión **v1.1** (2026-11-26)
**Rúbrica:** 5 criterios, 100 puntos. Cada indicador se evalúa en 4 niveles (4 Excelente = 1.00, 3 Competente = 0.75, 2 En desarrollo = 0.50, 1 Insuficiente = 0.25). El puntaje del criterio es el promedio de sus indicadores × su peso.

> **Regla de exclusión de la rúbrica:** "un proyecto que opere sin control de riesgo o que pueda enviar órdenes reales queda en nivel 1 en los criterios 2 y 5". **Este proyecto no cae en ella.**
> - **No puede enviar órdenes reales.** No existe código que firme ni envíe órdenes a MEXC, no se usa ninguna clave de trading y cualquier ruta `/api/real/*` responde `403` (probado en `test_api_rejects_missing_or_wrong_key`).
> - **Sí tiene control de riesgo determinista** (`backend/app/paper.py`, `config/risk.json`), que el LLM no puede modificar.

Solo se publicaron los indicadores del criterio 1; para los criterios 2 a 5 se evalúa contra el título del criterio y lo que una app de este tipo suele implicar, como indica la rúbrica ("es un indicador, no una lista obligatoria"). Cada fila cita la evidencia verificable.

---

## 1 · App: FrontEnd y BackEnd (20 pts)

| Indicador | Nivel | Evidencia |
|---|---|---|
| **Arquitectura** | **4** | Capas separadas: **FrontEnd** (React + TS, `frontend/`) → **BackEnd/gateway** (FastAPI, `backend/app/main.py`, autenticado con `X-API-Key`; también es gateway del LLM) → **broker** (cliente de solo lectura de MEXC `mexc.py` + motor Paper propio `paper.py`). El navegador **nunca** habla con MEXC ni con Ollama. Diagrama en README §2 con los flujos **REST** (`/api/*`) y **WebSocket** (`/ws`: ticker cada 5 s, decisiones, estado y errores). |
| **FrontEnd** | **4** | Datos **en vivo** de MEXC (ticker por WebSocket; indicador "● WebSocket en vivo" con reconexión automática y respaldo REST). **Gráfico de velas** OHLC con **SMA20, Bandas de Bollinger (20, 2σ)** y las **entradas y salidas** de la corrida (BUY, SELL, SHORT, COVER). **Estados de carga** (skeleton "Cargando datos de MEXC…") y **de error** (caja con botón *Reintentar*, avisos de dato no actualizado, banner cuando el agente se pausa). **Responsive** (grids a 1 columna en pantallas angostas). |
| Seguridad de claves | 4 | `AGENT_API_KEY` solo en `backend/.env` (en `.gitignore`); el frontend la pide al entrar y la guarda en `sessionStorage`, nunca en el código ni en `VITE_*`. El WebSocket se autentica con el primer mensaje, así la clave no viaja en la URL. Comparación de tiempo constante (`hmac.compare_digest`). Probado: 401 sin clave o con clave incorrecta, WebSocket rechazado (`test_websocket_requires_key`). |
| Solo Paper | 4 | Ver la regla de exclusión arriba. |

**Estimado: 4/4 → 20 / 20.**

## 2 · Consideraciones financieras (25 pts)

| Aspecto | Nivel | Evidencia (con números) |
|---|---|---|
| Costos y fricción | 4 | Costo por lado configurable (0.10% en la práctica, 0% MEXC en el agente) y **deslizamiento de 2 pb** documentado. Se cobra solo al cambiar de posición. El impacto está medido: en TE la MLP solo largo da −22.7% con 0.10% y −6.2% con 0%; costos de 15,909 USDT en 97 operaciones. |
| Tamaño de posición y restricciones | 4 | Exposición máxima del 95% del patrimonio. **Sin apalancamiento**: un corto queda 100% respaldado por efectivo. Respeta el tamaño mínimo y la precisión del par leídos de `exchangeInfo` de MEXC. |
| Límites de riesgo | 4 | Pérdida diaria máxima de 5% (bloquea entradas), **drawdown que detiene** de 25%, antigüedad máxima del dato de 30 h y **stop-loss por operación** configurable. Con stop de 2% se activó 2 veces en el replay de 90 días. |
| Métricas de desempeño y riesgo | 4 | Por corrida: rendimiento neto, **PnL realizado y total**, max drawdown, Win Rate, operaciones largas y cortas, **Sharpe, Sortino, volatilidad anual, VaR y CVaR históricos al 95%** (fórmulas de los apuntes de clase). Ejemplo del Long/Short de 90 días: Sharpe 2.11, Sortino 4.25, VaR 1.66%, CVaR 2.11%. |
| Referencias financieras | 4 | Toda corrida se compara con **comprar y mantener** en el mismo periodo y con los mismos costos; en la práctica también con **efectivo**. |
| Momento de ejecución realista | 4 | Señal al cierre de *t* → ejecución en la apertura de *t+1* (replay) o al precio vivo tras el cierre (Live). No se usa información futura. Se excluye la vela abierta. |

**Estimado: 4/4 → 25 / 25.**

## 3 · Aprendizaje (ML / DL) y predicciones (20 pts)

| Aspecto | Nivel | Evidencia |
|---|---|---|
| Datos y partición sin fuga | 4 | 2,521 velas diarias de MEXC, auditadas. Partición TR/VA/TE 70/15/15 **por fecha del objetivo**, con verificación. Escalador ajustado solo con TR. Fila de control de aprendizaje. |
| Modelos de referencia y profundos | 4 | Pronóstico cero, regresión lineal, **MLP y LSTM entrenadas en Tesla T4** (`nvidia-smi`, `cuda:0` en cada experimento): tres configuraciones por red, tres semillas, parada temprana, curvas TR/VA. |
| Validación | 4 | Tres ventanas expansivas, cinco umbrales en VA, configuración congelada con hash y **TE evaluado una sola vez**. |
| Predicciones y métricas | 4 | MAE, RMSE y precisión direccional en TR, VA y TE. `predicciones_TE.csv` trazable y `prediccion_futura.csv` con fecha. RMSE TE: MLP 0.023351 y LSTM 0.023060, frente a regresión 0.023403 y cero 0.023122. |
| Integración al producto | 4 | El modelo de la T4 se exporta con su contrato de variables y se **verifica al cargar** (la inferencia en NumPy reproduce las predicciones de PyTorch). El panel muestra el pronóstico del día siguiente. |

**Estimado: 4/4 → 20 / 20.** Límite declarado: solo la LSTM con indicadores supera al pronóstico cero en TE.

## 4 · IA como cerebro de operaciones (15 pts)

| Aspecto | Nivel | Evidencia |
|---|---|---|
| LLM local y privado | 4 | Ollama 0.40.18 con `llama3.2:3b` en la propia computadora, sin nube. Gateway autenticado (`/api/llm/query`, `/api/llm/info`). Latencia medida: 6.1 s por decisión. |
| Decisión estructurada | 4 | El LLM recibe datos estructurados (precio, SMA20, RSI, volatilidad, pronóstico, señal, posición, entrada, acciones permitidas) y debe responder un **JSON** `{action, symbol, candle_date, reason}` con un *schema* forzado. Se valida tipo, acción, símbolo y fecha; si falla hay un reintento y luego se rechaza. |
| Gobierno y seguridad | 4 | El LLM **no puede saltarse el riesgo** ni cambiar límites. Los datos externos se tratan como datos y no como instrucciones. Los fallos de Ollama pausan el agente. |
| Valor aportado | 3 | **Evaluado con números:** en los mismos 60 días, la regla sin LLM da +2.55% y el agente con LLM −10.68% (prompt v2). El LLM aporta explicación (motivo por decisión) pero no rendimiento; se reporta así, sin maquillar. |

**Estimado: (4+4+4+3)/4 = 3.75 → 14.1 / 15.**

## 5 · Trading: Entry, Long, Short (20 pts)

| Aspecto | Nivel | Evidencia |
|---|---|---|
| **Entry** (reglas de entrada) | 4 | La entrada se decide con el pronóstico de la T4 contra un umbral δ = 0.20% elegido en VA (con filtro SMA20 evaluado). Se registra el **precio de entrada**, se muestra en el panel y se usa para el stop-loss y el PnL. |
| **Long** | 4 | `BUY` → `SELL`, con PnL por operación. Solo largo, 90 días: **+7.03%** (PnL +3,513.67 USDT, 51 operaciones, Win Rate 52%, max DD 4.9%), contra −24.66% de comprar y mantener. |
| **Short** | 4 | `SHORT` → `COVER` en Paper, sin apalancamiento. Long/Short, 90 días: **+11.08%** (PnL +5,539.64; los cortos aportan +902.17), Sharpe 2.11, max DD 4.5%. En el año completo de TE con 0% de costo: **+6.07%** (cortos +2,903), contra −6.17% solo largo y −35.25% de comprar y mantener. |
| Salidas y gestión | 4 | Cierre por señal, stop-loss por operación, *Detener* (bloquea nuevas órdenes sin liquidar) y *Liquidar posición* (cierra un largo o un corto al precio vivo). Idempotencia: una sola decisión por vela. |
| Pruebas | 4 | 23 pruebas automatizadas: abrir y cerrar cortos, PnL de un corto, sin apalancamiento, sin doble posición ni giro directo, corto bloqueado en modo solo largo, transiciones de señal y stop-loss. |

**Estimado: 4/4 → 20 / 20.**

---

## Total estimado: **≈ 99 / 100**

## Qué falta, por qué y qué riesgos corre

| Tema | Decisión | Por qué |
|---|---|---|
| Órdenes reales | **Excluidas a propósito** | El protocolo y la rúbrica lo exigen; un sistema que puede enviar órdenes reales queda en nivel 1. |
| Short en la práctica (capítulos 7 a 11) | Solo largo por defecto | El protocolo de la práctica pide "sin ventas en corto". El Short se agregó como modo opcional del agente por el criterio 5 de esta rúbrica. |
| Costo del préstamo en cortos | No modelado | En Paper no hay *funding* ni interés de préstamo; con 0.10% por lado el Long/Short en TE sigue negativo (−14.87%). |
| Contaminación de TE | Declarada | El análisis Long/Short en TE se hizo después de la evaluación oficial solo largo. La regla es simétrica y sin parámetros nuevos, pero hace falta un periodo nuevo para confirmarla. |
| Un activo y frecuencia diaria | Limitación | Solo BTC/USDT; una decisión al día. Los 20 o más ciclos se obtienen en replay (identificado) y se probó un ciclo Live real. |
| Rentabilidad del +20% | No alcanzada en TE | En los últimos 90 días: +11.08% (Long/Short). En el año de TE: +6.07% (Long/Short, 0%) o −6.2% (solo largo). Se reporta tal cual. |
| Valor del LLM | Bajo | El LLM de 3B no mejora a la regla; la siguiente mejora sería un modelo mayor o un agente DQN/DDQN. |

**Riesgos de una app así en producción:** sobreajuste a un régimen de mercado, costos y deslizamiento reales mayores, caídas de la API o datos atrasados (mitigado con reintentos, límites de antigüedad y pausa), errores de un LLM (mitigado con JSON validado y riesgo determinista) y gestión de secretos (mitigada con la clave solo en el servidor).
