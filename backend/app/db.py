"""Persistencia SQLite: corridas, velas procesadas, decisiones, órdenes Paper, patrimonio y eventos/errores."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mode TEXT NOT NULL,              -- replay | live
  strategy TEXT NOT NULL,          -- llm | model (sin LLM)
  symbol TEXT NOT NULL,
  model_version TEXT NOT NULL,
  llm_model TEXT,
  initial_capital REAL NOT NULL,
  fee_rate REAL NOT NULL,
  replay_start TEXT, replay_end TEXT,
  status TEXT NOT NULL,            -- running | paused | stopped | halted | finished
  cash REAL NOT NULL, qty REAL NOT NULL, entry_cost REAL NOT NULL DEFAULT 0,
  day_start_equity REAL, day_key TEXT, peak_equity REAL,
  last_candle TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candles (
  symbol TEXT NOT NULL, date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  source TEXT NOT NULL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  candle_date TEXT NOT NULL,
  close REAL, sma20 REAL, dist_sma20 REAL, vol20 REAL, rsi14 REAL,
  forecast REAL, model_signal TEXT,
  proposed_action TEXT, proposal_source TEXT, reason TEXT,
  final_action TEXT, risk_status TEXT, risk_reason TEXT,
  llm_latency_ms REAL, llm_raw TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (run_id, candle_date)      -- idempotencia: una decisión por vela y corrida
);
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL, decision_id INTEGER,
  side TEXT NOT NULL, status TEXT NOT NULL,   -- FILLED | REJECTED
  ref_price REAL, exec_price REAL, qty REAL, notional REAL, fee REAL,
  pnl REAL, reason TEXT, exec_date TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
  run_id INTEGER NOT NULL, date TEXT NOT NULL,
  price REAL, cash REAL, qty REAL, equity REAL,
  PRIMARY KEY (run_id, date)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, time TEXT NOT NULL, level TEXT NOT NULL,
  kind TEXT NOT NULL, message TEXT NOT NULL, detail TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        # migraciones v1.1 (bases creadas con versiones anteriores)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(runs)").fetchall()}
        for col, ddl in (("allow_short", "INTEGER NOT NULL DEFAULT 0"), ("entry_price", "REAL NOT NULL DEFAULT 0"), ("stop_loss_pct", "REAL NOT NULL DEFAULT 0")):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")
        self.lock = threading.RLock()
        self.listeners: list = []   # callbacks del WebSocket (se notifican eventos en vivo)

    def notify(self, msg: dict):
        for cb in list(self.listeners):
            try:
                cb(msg)
            except Exception:  # un cliente caído no debe detener al agente
                self.listeners.remove(cb)

    def q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict | None:
        r = self.q(sql, args)
        return r[0] if r else None

    def ex(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            return cur.lastrowid

    # ---------- eventos / bitácora ----------
    def event(self, kind: str, message: str, level: str = "INFO", run_id: int | None = None, detail: dict | None = None):
        # Nunca registrar secretos: detail solo contiene datos de diagnóstico.
        self.ex("INSERT INTO events(run_id,time,level,kind,message,detail) VALUES (?,?,?,?,?,?)",
                (run_id, now_iso(), level, kind, message, json.dumps(detail, default=str) if detail else None))
        self.notify({"type": "event", "kind": kind, "level": level, "message": message, "run_id": run_id})

    def save_candles(self, symbol: str, df, source: str = "MEXC"):
        ts = now_iso()
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?,?)",
                [(symbol, d.strftime("%Y-%m-%d"), o, h, l, c, v, source, ts)
                 for d, o, h, l, c, v in zip(df["date"], df["open"], df["high"], df["low"], df["close"], df["volume"])],
            )
