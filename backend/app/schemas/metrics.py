from pydantic import BaseModel, Field


class TokenTelemetry(BaseModel):
    token: str
    accepted: bool
    probability: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    position: int = Field(ge=0)


class EngineMetrics(BaseModel):
    tokens_per_second: float = Field(ge=0.0)
    memory_bandwidth_gbps: float = Field(ge=0.0)
