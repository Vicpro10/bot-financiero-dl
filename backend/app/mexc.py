"""Cliente de datos PÚBLICOS de MEXC (spot v3). Solo lectura.

No existe ninguna función para enviar órdenes reales: el bot opera solo en Paper.
Documentación: https://www.mexc.com/api-docs/spot-v3/market-data-endpoints
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
import pandas as pd

log = logging.getLogger("mexc")

BASE_URL = "https://api.mexc.com"
DAY_MS = 86_400_000
MAX_PER_REQUEST = 500  # MEXC devuelve como máximo ~500 velas por consulta


class MarketDataError(RuntimeError):
    """La API no respondió, limitó solicitudes o devolvió datos inválidos."""


@dataclass
class SymbolRules:
    symbol: str
    base_asset: str
    quote_asset: str
    qty_step: float  # baseSizePrecision
    price_decimals: int
    min_notional: float
    taker_fee: float


class MexcClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0, retries: int = 3, backoff: float = 1.0):
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    async def _get(self, path: str, params: dict) -> object:
        """GET con reintentos limitados y espera creciente (1s, 2s, 4s...)."""
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as cli:
                    r = await cli.get(path, params=params)
                if r.status_code == 429:
                    raise MarketDataError("MEXC respondió 429 (límite de solicitudes).")
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, MarketDataError, ValueError) as e:
                last_err = e
                wait = self.backoff * (2**attempt)
                log.warning("MEXC %s intento %d/%d falló: %s; reintento en %.1fs", path, attempt + 1, self.retries, e, wait)
                if attempt < self.retries - 1:
                    await asyncio.sleep(wait)
        raise MarketDataError(f"MEXC sin respuesta válida tras {self.retries} intentos ({path}): {last_err}")

    async def klines(self, symbol: str, start_ms: int, end_ms: int) -> list[list]:
        data = await self._get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": 1000},
        )
        if not isinstance(data, list):
            raise MarketDataError(f"Respuesta inesperada de klines: {str(data)[:200]}")
        return data

    async def daily_candles(self, symbol: str, start: str = "2020-01-01", now_ms: int | None = None) -> pd.DataFrame:
        """Velas diarias CERRADAS (UTC) desde `start`. La vela del día en curso se excluye."""
        now_ms = now_ms or int(time.time() * 1000)
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        rows: list[list] = []
        cur = start_ms
        while cur < now_ms:
            end = min(cur + MAX_PER_REQUEST * DAY_MS - 1, now_ms)
            chunk = await self.klines(symbol, cur, end)
            rows.extend(chunk)
            cur = end + 1
        return candles_to_frame(rows, now_ms)

    async def recent_candles(self, symbol: str, days: int = 120) -> pd.DataFrame:
        now_ms = int(time.time() * 1000)
        start = pd.Timestamp(now_ms - days * DAY_MS, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        return await self.daily_candles(symbol, start=start, now_ms=now_ms)

    async def ticker(self, symbol: str) -> dict:
        d = await self._get("/api/v3/ticker/24hr", {"symbol": symbol})
        if isinstance(d, list):
            d = d[0]
        return {
            "symbol": d["symbol"],
            "last": float(d["lastPrice"]),
            "change_pct": float(d["priceChangePercent"]),
            "high": float(d["highPrice"]),
            "low": float(d["lowPrice"]),
            "volume": float(d["volume"]),
            "time": int(d.get("closeTime") or time.time() * 1000),
        }

    async def symbol_rules(self, symbol: str) -> SymbolRules:
        d = await self._get("/api/v3/exchangeInfo", {"symbol": symbol})
        s = d["symbols"][0]
        return SymbolRules(
            symbol=s["symbol"],
            base_asset=s["baseAsset"],
            quote_asset=s["quoteAsset"],
            qty_step=float(s.get("baseSizePrecision") or 10 ** -int(s["baseAssetPrecision"])),
            price_decimals=int(s.get("quotePrecision", 2)),
            min_notional=max(float(s.get("quoteAmountPrecision") or 1), 1.0),
            taker_fee=float(s.get("takerCommission", 0)),
        )


def candles_to_frame(rows: list[list], now_ms: int) -> pd.DataFrame:
    """[openTime, open, high, low, close, volume, closeTime, quoteVolume] -> DataFrame de velas cerradas."""
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "close_time"])
    df = pd.DataFrame([r[:7] for r in rows], columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df["close_time"].astype("int64") <= now_ms]  # solo velas cerradas
    df["date"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume", "close_time"]].reset_index(drop=True)
