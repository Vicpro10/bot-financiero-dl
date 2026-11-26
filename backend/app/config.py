"""Configuración del servidor. Los secretos se leen SOLO de variables de entorno / .env del backend."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv(BACKEND_DIR / ".env")


@dataclass
class RiskConfig:
    """Límites deterministas definidos ANTES de la prueba. El LLM no puede modificarlos."""
    initial_capital: float = 50_000.0
    fee_rate: float = 0.0             # 0.0% por compra o venta (protocolo, sección 15; MEXC 0 fees maker)
    slippage_bps: float = 2.0         # supuesto: ejecución 2 pb peor que el precio de referencia
    max_exposure: float = 0.95        # fracción máxima del patrimonio invertida en el activo
    max_daily_loss: float = 0.05      # si el patrimonio cae 5% desde el inicio del día: bloquear compras
    max_drawdown_halt: float = 0.25   # si el drawdown supera 25%: detener el agente
    max_data_age_hours: float = 30.0  # la última vela cerrada debe tener menos de 30 h
    allowed_symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    allowed_actions: list[str] = field(default_factory=lambda: ["BUY", "HOLD", "SELL", "SHORT", "COVER"])
    stop_loss_pct: float = 0.0       # stop-loss por operación sobre el precio de entrada (0 = apagado; p. ej. 0.05 = 5%)
    allow_short: bool = True          # interruptor general; cada corrida elige además si permite cortos (Paper, sin apalancamiento)


@dataclass
class Settings:
    api_key: str = ""
    symbol: str = "BTCUSDT"
    db_path: str = str(BACKEND_DIR / "data" / "agent.db")
    model_dir: str = str(BACKEND_DIR / "models" / "current")
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 60.0
    live_poll_seconds: int = 300
    replay_step_seconds: float = 0.3
    enable_fault_injection: bool = True
    risk_path: str = str(BACKEND_DIR / "config" / "risk.json")
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    def public(self) -> dict:
        d = asdict(self)
        d.pop("api_key")  # nunca exponer el secreto
        d["api_key_configured"] = bool(self.api_key)
        return d


def load_settings() -> Settings:
    s = Settings(
        api_key=os.environ.get("AGENT_API_KEY", ""),
        symbol=os.environ.get("SYMBOL", "BTCUSDT"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        live_poll_seconds=int(os.environ.get("LIVE_POLL_SECONDS", "300")),
        replay_step_seconds=float(os.environ.get("REPLAY_STEP_SECONDS", "0.3")),
        enable_fault_injection=os.environ.get("ENABLE_FAULT_INJECTION", "true").lower() == "true",
    )
    if os.environ.get("DB_PATH"):
        s.db_path = os.environ["DB_PATH"]
    if os.environ.get("MODEL_DIR"):
        s.model_dir = os.environ["MODEL_DIR"]
    return s


def load_risk(path: str) -> RiskConfig:
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return RiskConfig(**data)
    cfg = RiskConfig()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return cfg


def save_risk(path: str, cfg: RiskConfig) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
