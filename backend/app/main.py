"""API del agente (FastAPI). Documentación interactiva en /docs.

Todas las rutas /api/* excepto /api/health exigen la cabecera `X-API-Key`.
El backend también es el GATEWAY de Ollama: nadie habla con el LLM sin pasar por aquí.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import features as F
from .agent import FAULTS, Agent, run_metrics
from .config import RiskConfig, load_risk, load_settings, save_risk
from .db import DB
from .llm import LLMUnavailable, OllamaClient
from .mexc import MarketDataError, MexcClient
from .model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

S = load_settings()
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not S.api_key or len(S.api_key) < 16:
        raise RuntimeError("Defina AGENT_API_KEY (>=16 caracteres) en backend/.env. Vea .env.example.")
    db = DB(S.db_path)
    risk = load_risk(S.risk_path)
    model = load_model(S.model_dir)
    STATE.update(db=db, risk=risk, model=model, mexc=MexcClient(),
                 llm=OllamaClient(S.ollama_url, S.ollama_model, S.ollama_timeout))
    STATE["agent"] = Agent(S, risk, db, model, STATE["mexc"], STATE["llm"])
    db.event("boot", f"Backend iniciado. Modelo {model.version} ({model.kind}). Órdenes reales: DESHABILITADAS.")
    yield
    await STATE["agent"].stop() if STATE["agent"].status == "running" else None


app = FastAPI(title="Bot Financiero DL – Agente Paper MEXC", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=S.cors_origins, allow_methods=["*"], allow_headers=["X-API-Key", "Content-Type"])


def require_key(request: Request, x_api_key: str | None = Header(default=None)):
    if not x_api_key or not hmac.compare_digest(x_api_key, S.api_key):
        STATE["db"].event("auth", f"Acceso rechazado a {request.url.path}: clave {'ausente' if not x_api_key else 'incorrecta'}.", "WARN")
        raise HTTPException(status_code=401, detail="API key ausente o incorrecta.")


auth = [Depends(require_key)]


@app.exception_handler(MarketDataError)
async def _mde(_, e: MarketDataError):
    STATE["db"].event("mexc", str(e), "ERROR")
    return JSONResponse(status_code=503, content={"detail": str(e)})


@app.exception_handler(F.DataValidationError)
async def _dve(_, e):
    STATE["db"].event("data", str(e), "ERROR")
    return JSONResponse(status_code=422, content={"detail": str(e)})


# ---------------------------------------------------------------- públicas
@app.get("/api/health")
async def health():
    return {"ok": True, "real_orders": "disabled", "paper_only": True}


# ---------------------------------------------------------------- estado y datos
@app.get("/api/status", dependencies=auth)
async def status():
    ag: Agent = STATE["agent"]
    m = STATE["model"]
    p = ag.portfolio()
    run = STATE["db"].one("SELECT * FROM runs WHERE id=?", (ag.run_id,)) if ag.run_id else None
    last_dec = STATE["db"].one("SELECT * FROM decisions WHERE run_id=? ORDER BY candle_date DESC LIMIT 1", (ag.run_id,)) if ag.run_id else None
    return {"status": ag.status, "status_reason": ag.status_reason, "run": run, "symbol": S.symbol,
            "portfolio": asdict(p) if p else None, "last_decision": last_dec,
            "model": {"version": m.version, "kind": m.kind, "signal_rule": m.meta["signal_rule"], "trained_on": m.meta.get("trained_on"),
                      "features": m.meta.get("features") or m.meta.get("seq_vars"), "splits": m.meta.get("splits")},
            "risk": asdict(STATE["risk"]), "faults_pending": sorted(ag.faults), "real_orders": "disabled"}


@app.get("/api/market", dependencies=auth)
async def market(days: int = 90):
    mx: MexcClient = STATE["mexc"]
    tick = await mx.ticker(S.symbol)
    df = await mx.recent_candles(S.symbol, days=max(days, 60))
    df = F.validate_candles(df)
    feats = F.build_features(df)
    sd = feats["close"].rolling(20).std(ddof=0)            # Bandas de Bollinger: SMA20 ± 2σ de los cierres
    feats["bb_up"], feats["bb_dn"] = feats["sma20"] + 2 * sd, feats["sma20"] - 2 * sd
    i = len(feats) - 1
    m = STATE["model"]
    fc = m.predict_row(feats, i)
    last = feats.iloc[i]
    return {
        "ticker": tick, "source": "MEXC spot v3 (datos públicos)",
        "last_closed_candle": feats.index[i].strftime("%Y-%m-%d"),
        "indicators": {k: float(last[k]) for k in ("close", "sma20", "dist_sma20", "vol20", "rsi14")},
        "forecast": {"cutoff": feats.index[i].strftime("%Y-%m-%d"), "next_return": fc,
                     "signal": m.signal(fc, float(last["close"]), float(last["sma20"])),
                     "signal_long_short": m.signal(fc, float(last["close"]), float(last["sma20"]), allow_short=True)},
        "candles": [{"date": d.strftime("%Y-%m-%d"), "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                     "sma20": _num(r.sma20), "bb_up": _num(r.bb_up), "bb_dn": _num(r.bb_dn), "rsi14": _num(r.rsi14)}
                    for d, r in feats.tail(days).iterrows()],
    }


@app.get("/api/runs", dependencies=auth)
async def runs():
    db = STATE["db"]
    return [run_metrics(db, r["id"]) for r in db.q("SELECT id FROM runs ORDER BY id DESC LIMIT 30")]


@app.get("/api/runs/{run_id}", dependencies=auth)
async def run_detail(run_id: int):
    db = STATE["db"]
    if not db.one("SELECT id FROM runs WHERE id=?", (run_id,)):
        raise HTTPException(404, "Corrida inexistente")
    return {"metrics": run_metrics(db, run_id),
            "equity": db.q("SELECT * FROM equity WHERE run_id=? ORDER BY date", (run_id,)),
            "decisions": db.q("SELECT * FROM decisions WHERE run_id=? ORDER BY candle_date DESC", (run_id,)),
            "orders": db.q("SELECT * FROM orders WHERE run_id=? ORDER BY id DESC", (run_id,))}


@app.get("/api/events", dependencies=auth)
async def events(limit: int = 100):
    return STATE["db"].q("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


# ---------------------------------------------------------------- control humano (protegido)
class StartReq(BaseModel):
    mode: str = Field("replay", pattern="^(replay|live)$")
    strategy: str = Field("model", pattern="^(llm|model)$")
    replay_days: int = Field(60, ge=20, le=365)
    stop_loss_pct: float | None = Field(None, ge=0, le=0.5)   # stop-loss por operación (None = el de risk.json)
    allow_short: bool = False   # Long/Short en Paper (sin apalancamiento); por defecto solo largo, como el protocolo


@app.post("/api/agent/start", dependencies=auth)
async def start(req: StartReq):
    try:
        rid = await STATE["agent"].start(req.mode, req.strategy, req.replay_days, req.allow_short, req.stop_loss_pct)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e))
    return {"run_id": rid}


@app.post("/api/agent/pause", dependencies=auth)
async def pause():
    STATE["agent"].pause()
    return {"status": STATE["agent"].status}


@app.post("/api/agent/resume", dependencies=auth)
async def resume():
    await STATE["agent"].resume()
    return {"status": STATE["agent"].status}


@app.post("/api/agent/stop", dependencies=auth)
async def stop():
    await STATE["agent"].stop()
    return {"status": STATE["agent"].status}


@app.post("/api/agent/liquidate", dependencies=auth)
async def liquidate():
    try:
        return {"order": await STATE["agent"].liquidate()}
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.put("/api/config/risk", dependencies=auth)
async def set_risk(cfg: dict):
    ag: Agent = STATE["agent"]
    if ag.status == "running":
        raise HTTPException(409, "No se puede cambiar el riesgo con el agente en marcha. Deténgalo primero.")
    new = RiskConfig(**{**asdict(STATE["risk"]), **cfg})
    save_risk(S.risk_path, new)
    STATE["risk"] = ag.risk = new
    STATE["db"].event("config", "Configuración de riesgo actualizada por el usuario.", detail=asdict(new))
    return asdict(new)


# ---------------------------------------------------------------- gateway del LLM (protegido)
class LLMQuery(BaseModel):
    prompt: str = Field(..., max_length=2000)


@app.get("/api/llm/info", dependencies=auth)
async def llm_info():
    try:
        return await STATE["llm"].info()
    except LLMUnavailable as e:
        raise HTTPException(503, str(e))


@app.post("/api/llm/query", dependencies=auth)
async def llm_query(q: LLMQuery):
    """Consulta directa al LLM local (demostración de inferencia sin nube y medición de latencia)."""
    try:
        text, ms = await STATE["llm"].chat([{"role": "user", "content": q.prompt}])
    except LLMUnavailable as e:
        STATE["db"].event("ollama_down", str(e), "ERROR")
        raise HTTPException(503, str(e))
    return {"model": S.ollama_model, "latency_ms": ms, "answer": text, "inference": "local (Ollama)"}


# ---------------------------------------------------------------- demostración de errores
@app.post("/api/faults/{kind}", dependencies=auth)
async def inject(kind: str):
    if not S.enable_fault_injection:
        raise HTTPException(403, "Inyección de fallos deshabilitada.")
    if kind not in FAULTS:
        raise HTTPException(400, f"Fallo desconocido; use uno de {sorted(FAULTS)}")
    STATE["agent"].faults.add(kind)
    STATE["db"].event("fault", f"Fallo '{kind}' programado para el próximo ciclo (demostración).", "WARN")
    return {"pending": sorted(STATE["agent"].faults)}


# Explícitamente: no existe ninguna ruta de órdenes reales.
@app.api_route("/api/real/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def real_orders_blocked(path: str):
    raise HTTPException(403, "Órdenes reales deshabilitadas: este sistema solo opera en modo Paper.")


def _num(v):
    return None if v is None or v != v else float(v)


# ---------------------------------------------------------------- WebSocket (eventos en vivo)
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Canal en vivo para la UI. Autenticación: el primer mensaje debe ser {"type":"auth","key":"..."}
    (la clave no viaja en la URL, así no queda en logs). Luego el servidor empuja:
    ticker de MEXC cada 5 s, decisiones, cambios de estado y eventos/errores del agente."""
    await websocket.accept()
    try:
        first = json.loads(await asyncio.wait_for(websocket.receive_text(), timeout=10))
    except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
        await websocket.close(code=4401)
        return
    if not isinstance(first, dict) or not hmac.compare_digest(str(first.get("key", "")), S.api_key):
        STATE["db"].event("auth", "WebSocket rechazado: clave ausente o incorrecta.", "WARN")
        await websocket.send_text(json.dumps({"type": "error", "message": "API key ausente o incorrecta."}))
        await websocket.close(code=4401)
        return
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    loop = asyncio.get_running_loop()

    def push(msg: dict):
        loop.call_soon_threadsafe(lambda: q.full() or q.put_nowait(msg))

    STATE["db"].listeners.append(push)
    await websocket.send_text(json.dumps({"type": "hello", "status": STATE["agent"].status, "symbol": S.symbol}))

    async def ticker_loop():
        while True:
            try:
                t = await STATE["mexc"].ticker(S.symbol)
                push({"type": "ticker", **t})
            except MarketDataError as e:
                push({"type": "error", "message": f"MEXC: {e}"})
            await asyncio.sleep(5)

    tk = asyncio.create_task(ticker_loop())
    try:
        while True:
            msg = await q.get()
            await websocket.send_text(json.dumps(msg, default=str))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        tk.cancel()
        if push in STATE["db"].listeners:
            STATE["db"].listeners.remove(push)
