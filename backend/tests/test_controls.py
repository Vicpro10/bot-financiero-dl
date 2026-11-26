"""Pruebas automatizadas de los controles del agente (sin red: MEXC y Ollama se simulan).

Cubren los nueve escenarios del protocolo (sección 15): API caída / 429, dato atrasado, Ollama detenido,
JSON inválido, clave incorrecta, saldo insuficiente, evento duplicado, reinicio del backend y detención.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.mkdtemp()
os.environ["AGENT_API_KEY"] = "test-key-0123456789abcdef"
os.environ["DB_PATH"] = str(Path(TMP) / "test.db")
os.environ["REPLAY_STEP_SECONDS"] = "0"

from app import features as F  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.config import RiskConfig, Settings  # noqa: E402
from app.db import DB  # noqa: E402
from app.llm import LLMInvalidResponse, LLMUnavailable, Proposal, validate_proposal  # noqa: E402
from app.mexc import MarketDataError, MexcClient, SymbolRules, candles_to_frame  # noqa: E402
from app.model import load_model  # noqa: E402
from app.paper import Portfolio, check, execute  # noqa: E402

RULES = SymbolRules("BTCUSDT", "BTC", "USDT", qty_step=1e-6, price_decimals=2, min_notional=1.0, taker_fee=0.0)
CFG = RiskConfig()


def synthetic_candles(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 60000 * np.cumprod(1 + rng.normal(0, 0.02, n))
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"date": dates, "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 100.0, "close_time": (dates + pd.Timedelta(days=1)).astype("int64") // 10**6})


class FakeMexc:
    def __init__(self, fail=False):
        self.fail = fail

    async def symbol_rules(self, s):
        return RULES

    async def daily_candles(self, s, start=None, now_ms=None):
        if self.fail:
            raise MarketDataError("MEXC sin respuesta válida tras 3 intentos")
        return synthetic_candles()

    async def ticker(self, s):
        return {"last": 61000.0}


class FakeLLM:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.model = "fake"

    async def propose(self, ctx):
        if self.mode == "down":
            raise LLMUnavailable("Ollama no responde en http://127.0.0.1:11434")
        if self.mode == "bad":
            validate_proposal("no es json", ctx["symbol"], ctx["candle_date"], ctx["allowed_actions"])
        return Proposal(action=ctx["allowed_actions"][0], symbol=ctx["symbol"], candle_date=ctx["candle_date"],
                        reason="prueba", latency_ms=5.0, raw="{}")


def make_agent(llm_mode="ok", mexc_fail=False, db_name=None):
    s = Settings(api_key="x" * 20, db_path=str(Path(TMP) / (db_name or f"a{np.random.randint(1e9)}.db")), replay_step_seconds=0)
    db = DB(s.db_path)
    model = load_model(s.model_dir)
    return Agent(s, RiskConfig(), db, model, FakeMexc(mexc_fail), FakeLLM(llm_mode)), db


def new_run(agent, strategy="model"):
    agent.rules = RULES
    agent.run_id = agent.db.ex(
        "INSERT INTO runs(mode,strategy,symbol,model_version,initial_capital,fee_rate,status,cash,qty,entry_cost,peak_equity,day_start_equity,created_at,updated_at) "
        "VALUES ('replay',?,?,?,?,?,'running',?,0,0,?,?,'t','t')",
        (strategy, "BTCUSDT", agent.model.version, 50000.0, 0.0, 50000.0, 50000.0, 50000.0))
    return agent.run_id


FEATS = F.build_features(F.validate_candles(synthetic_candles()))
ROW = len(FEATS) - 2


# ---------------------------------------------------------------- datos y modelo
def test_data_validation_catches_controlled_faults():
    df = synthetic_candles()
    with pytest.raises(F.DataValidationError, match="vacío"):
        F.validate_candles(df.iloc[0:0])
    with pytest.raises(F.DataValidationError, match="duplicadas"):
        F.validate_candles(pd.concat([df, df.tail(1)]))
    bad = df.copy()
    bad.loc[5, "close"] = np.nan
    with pytest.raises(F.DataValidationError, match="no finitos"):
        F.validate_candles(bad)
    with pytest.raises(F.DataValidationError, match="insuficiente"):
        F.validate_candles(df.head(5))


def test_open_candle_is_excluded():
    rows = [[0, "1", "1", "1", "1", "1", 86_400_000], [86_400_000, "1", "1", "1", "1", "1", 2 * 86_400_000]]
    assert len(candles_to_frame(rows, now_ms=86_400_000 + 5)) == 1


def test_model_contract_and_finite_prediction():
    m = load_model(ROOT / "models" / "current")
    y = m.predict_row(FEATS, ROW)
    assert np.isfinite(y)
    with pytest.raises(F.DataValidationError):
        m.predict_row(FEATS, 3)  # sin ventana completa


# ---------------------------------------------------------------- riesgo y Paper
def test_insufficient_balance_rejected():
    p = Portfolio(cash=0.5, peak_equity=0.5, day_start_equity=0.5, day_key="d")
    r = check("BUY", "BTCUSDT", p, 60000, CFG, RULES, None, "d")
    assert r.status == "REJECTED" and "insuficiente" in r.reason


def test_stale_data_blocks_orders():
    p = Portfolio(cash=50000, peak_equity=50000, day_start_equity=50000, day_key="d")
    r = check("BUY", "BTCUSDT", p, 60000, CFG, RULES, data_age_hours=72, day_key="d")
    assert r.status == "BLOCKED" and r.final_action == "HOLD"


def test_no_short_no_leverage_no_rebuy():
    p = Portfolio(cash=50000, peak_equity=50000, day_start_equity=50000, day_key="d")
    assert check("SELL", "BTCUSDT", p, 60000, CFG, RULES, None, "d").status == "REJECTED"  # sin posición -> no corto
    o = execute("BUY", p, 60000, CFG, RULES)
    assert o["notional"] + o["fee"] <= 50000 and p.cash >= 0 and p.qty * 60000 <= 0.951 * 50000
    assert check("BUY", "BTCUSDT", p, 60000, CFG, RULES, None, "d").status == "REJECTED"  # sin recompra


def test_symbol_not_allowed_and_stop_blocks():
    p = Portfolio(cash=50000, peak_equity=50000, day_start_equity=50000, day_key="d")
    assert check("BUY", "ETHUSDT", p, 3000, CFG, RULES, None, "d").status == "REJECTED"
    assert check("BUY", "BTCUSDT", p, 60000, CFG, RULES, None, "d", stopped=True).status == "BLOCKED"


def test_daily_loss_limit_blocks_buys():
    p = Portfolio(cash=47000, peak_equity=50000, day_start_equity=50000, day_key="d")
    r = check("BUY", "BTCUSDT", p, 60000, CFG, RULES, None, "d")
    assert r.status == "BLOCKED" and "diaria" in r.reason


# ---------------------------------------------------------------- LLM
def test_invalid_json_and_disallowed_values():
    with pytest.raises(LLMInvalidResponse):
        validate_proposal("{acción: COMPRAR", "BTCUSDT", "2026-01-01", ["BUY", "HOLD"])
    with pytest.raises(LLMInvalidResponse, match="no permitida"):
        validate_proposal('{"action":"SELL","symbol":"BTCUSDT","candle_date":"2026-01-01","reason":"x"}', "BTCUSDT", "2026-01-01", ["BUY", "HOLD"])
    with pytest.raises(LLMInvalidResponse, match="Símbolo"):
        validate_proposal('{"action":"BUY","symbol":"DOGEUSDT","candle_date":"2026-01-01","reason":"x"}', "BTCUSDT", "2026-01-01", ["BUY", "HOLD"])
    ok = validate_proposal('{"action":"buy","symbol":"BTCUSDT","candle_date":"2026-01-01","reason":"ok"}', "BTCUSDT", "2026-01-01", ["BUY", "HOLD"])
    assert ok["action"] == "BUY"


def test_agent_invalid_json_does_not_trade():
    ag, db = make_agent("bad")
    rid = new_run(ag, "llm")
    asyncio.run(ag.cycle(FEATS, ROW, 61000, "2026-04-30", 61000, None))
    d = db.one("SELECT * FROM decisions WHERE run_id=?", (rid,))
    assert d["final_action"] == "HOLD" and d["risk_status"] == "REJECTED"
    assert not db.q("SELECT * FROM orders WHERE run_id=? AND status='FILLED'", (rid,))


def test_agent_ollama_down_pauses():
    ag, db = make_agent("down")
    rid = new_run(ag, "llm")
    with pytest.raises(LLMUnavailable):
        asyncio.run(ag.cycle(FEATS, ROW, 61000, "2026-04-30", 61000, None))
    assert ag.status == "halted"
    assert db.one("SELECT * FROM events WHERE kind='ollama_down'")
    assert not db.q("SELECT * FROM orders WHERE run_id=?", (rid,))


# ---------------------------------------------------------------- idempotencia, reinicio, detención
def test_duplicate_candle_processed_once():
    ag, db = make_agent()
    rid = new_run(ag)
    asyncio.run(ag.cycle(FEATS, ROW, 61000, "2026-04-30", 61000, None))
    r2 = asyncio.run(ag.cycle(FEATS, ROW, 61000, "2026-04-30", 61000, None))
    assert "duplicate" in r2
    assert len(db.q("SELECT * FROM decisions WHERE run_id=?", (rid,))) == 1


def test_restart_recovers_state_without_duplicates():
    ag, db = make_agent(db_name="restart.db")
    rid = new_run(ag)
    asyncio.run(ag.cycle(FEATS, ROW, 61000, "2026-04-30", 61000, None))
    before = db.one("SELECT cash, qty, last_candle FROM runs WHERE id=?", (rid,))
    ag2, db2 = make_agent(db_name="restart.db")  # "reinicio" del backend con la misma base
    assert ag2.status == "paused" and ag2.run_id == rid
    after = db2.one("SELECT cash, qty, last_candle FROM runs WHERE id=?", (rid,))
    assert before == after
    r = asyncio.run(ag2.cycle(FEATS, ROW, 61000, "2026-04-30", 61000, None))
    assert "duplicate" in r


def test_mexc_down_halts_agent():
    ag, db = make_agent(mexc_fail=True)

    async def go():
        await ag.start("replay", "model", 30)
        await ag.task
    asyncio.run(go())
    assert ag.status == "halted" and "MEXC" in ag.status_reason


def test_stop_blocks_new_orders_but_keeps_position():
    ag, db = make_agent()

    async def go():
        await ag.start("replay", "model", 30)
        await asyncio.sleep(0.2)
        await ag.stop()
    asyncio.run(go())
    assert ag.status == "stopped"
    n = len(db.q("SELECT * FROM decisions WHERE run_id=?", (ag.run_id,)))
    asyncio.run(asyncio.sleep(0.1))
    assert len(db.q("SELECT * FROM decisions WHERE run_id=?", (ag.run_id,))) == n  # nada nuevo tras detener


# ---------------------------------------------------------------- API: clave incorrecta y órdenes reales
def test_api_rejects_missing_or_wrong_key():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/status").status_code == 401
        assert c.get("/api/status", headers={"X-API-Key": "incorrecta"}).status_code == 401
        assert c.get("/api/status", headers={"X-API-Key": os.environ["AGENT_API_KEY"]}).status_code == 200
        assert c.post("/api/llm/query", json={"prompt": "hola"}).status_code == 401
        assert c.post("/api/agent/start", json={}).status_code == 401
        assert c.post("/api/real/order", headers={"X-API-Key": os.environ["AGENT_API_KEY"]}).status_code == 403
        st = c.get("/api/status", headers={"X-API-Key": os.environ["AGENT_API_KEY"]}).json()
        assert "api_key" not in str(st).lower() or os.environ["AGENT_API_KEY"] not in str(st)


# ---------------------------------------------------------------- v1.1: Entry / Long / Short
def test_short_open_cover_pnl_and_no_leverage():
    p = Portfolio(cash=50000, peak_equity=50000, day_start_equity=50000, day_key="d")
    assert check("SHORT", "BTCUSDT", p, 60000, CFG, RULES, None, "d", allow_short=True).status == "APPROVED"
    o = execute("SHORT", p, 60000, CFG, RULES)
    assert p.position == "SHORT" and p.qty < 0 and p.entry_price == o["exec_price"]
    assert abs(p.qty) * 60000 <= 0.951 * 50000          # exposición ≤ 95%: sin apalancamiento
    assert abs(p.equity(60000) - 50000) < 50            # al abrir, el patrimonio casi no cambia (solo deslizamiento)
    assert p.equity(54000) > 50000 and p.equity(66000) < 50000  # el corto gana si baja y pierde si sube
    assert check("SHORT", "BTCUSDT", p, 60000, CFG, RULES, None, "d", allow_short=True).status == "REJECTED"  # sin doble corto
    assert check("BUY", "BTCUSDT", p, 60000, CFG, RULES, None, "d", allow_short=True).status == "REJECTED"    # sin giro directo
    c = execute("COVER", p, 54000, CFG, RULES)
    assert p.qty == 0 and c["pnl"] > 0 and abs(p.cash - (50000 + c["pnl"])) < 1e-6


def test_short_rejected_when_run_is_long_only():
    p = Portfolio(cash=50000, peak_equity=50000, day_start_equity=50000, day_key="d")
    r = check("SHORT", "BTCUSDT", p, 60000, CFG, RULES, None, "d", allow_short=False)
    assert r.status == "REJECTED" and "corto" in r.reason


def test_signal_to_action_transitions():
    from app.agent import signal_to_action
    cash, lng, sht = Portfolio(cash=1), Portfolio(cash=0, qty=1), Portfolio(cash=2, qty=-1)
    assert [signal_to_action(s, cash) for s in ("LONG", "SHORT", "CASH")] == ["BUY", "SHORT", "HOLD"]
    assert [signal_to_action(s, lng) for s in ("LONG", "SHORT", "CASH")] == ["HOLD", "SELL", "SELL"]
    assert [signal_to_action(s, sht) for s in ("LONG", "SHORT", "CASH")] == ["COVER", "HOLD", "COVER"]


def test_model_signal_symmetric_short():
    m = load_model(ROOT / "models" / "current")
    d = m.meta["signal_rule"]["delta"]
    assert m.signal(-2 * d, 100, 101, allow_short=True) == "SHORT"
    assert m.signal(-2 * d, 100, 101, allow_short=False) == "CASH"
    assert m.signal(2 * d, 102, 101, allow_short=True) == "LONG"


def test_agent_long_short_cycle_records_short():
    ag, db = make_agent()
    rid = new_run(ag)
    db.ex("UPDATE runs SET allow_short=1 WHERE id=?", (rid,))
    for i in range(30, len(FEATS) - 1):
        asyncio.run(ag.cycle(FEATS, i, float(FEATS["open"].iloc[i + 1]), FEATS.index[i + 1].strftime("%Y-%m-%d"),
                             float(FEATS["close"].iloc[i + 1]), None))
    sides = {o["side"] for o in db.q("SELECT side FROM orders WHERE run_id=? AND status='FILLED'", (rid,))}
    runs = db.one("SELECT qty FROM runs WHERE id=?", (rid,))
    assert sides <= {"BUY", "SELL", "SHORT", "COVER"} and sides  # opera y solo con acciones válidas
    assert runs["qty"] is not None


def test_websocket_requires_key():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        with c.websocket_connect("/ws") as w:
            w.send_text('{"type":"auth","key":"mala"}')
            assert "incorrecta" in w.receive_text()
        with c.websocket_connect("/ws") as w:
            w.send_text('{"type":"auth","key":"%s"}' % os.environ["AGENT_API_KEY"])
            assert '"hello"' in w.receive_text()


def test_stop_loss_closes_position_and_risk_metrics():
    from app.agent import risk_metrics
    ag, db = make_agent()
    rid = new_run(ag)
    db.ex("UPDATE runs SET qty=0.8, cash=2000, entry_price=70000, entry_cost=56000, stop_loss_pct=0.05 WHERE id=?", (rid,))
    feats = FEATS.copy()
    feats.iloc[ROW, feats.columns.get_loc("close")] = 60000.0   # -14% desde la entrada
    r = asyncio.run(ag.cycle(feats, ROW, 60000, "2026-04-30", 60000, None))
    assert r["final_action"] == "SELL" and "STOP-LOSS" in r["reason"]
    assert db.one("SELECT qty FROM runs WHERE id=?", (rid,))["qty"] == 0
    m = risk_metrics(pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.03, 0.01]))
    assert m["var95_diario"] > 0 and m["cvar95_diario"] >= m["var95_diario"] and m["sharpe_anual"] is not None
