from typing import Literal

from pydantic import BaseModel, Field

TokenStatus = Literal["accepted", "rejected", "correction", "bonus"]


class TokenTelemetry(BaseModel):
    token: str
    token_id: int
    status: TokenStatus
    position: int = Field(ge=0)
    draft_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    target_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    accept_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    timestamp: float


class RunMetrics(BaseModel):
    run_id: str
    is_final: bool
    elapsed_s: float = Field(ge=0.0)
    total_tokens: int = Field(ge=0)
    draft_tokens_per_second: float = Field(ge=0.0)
    target_tokens_per_second: float = Field(ge=0.0)
    effective_tokens_per_second: float = Field(ge=0.0)
    speedup_ratio: float = Field(ge=0.0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)
    memory_bandwidth_gbps: float = Field(ge=0.0)
    kv_cache_mb: float = Field(ge=0.0)
    current_k_lookahead: int = Field(ge=1)
    actual_temperature: float = Field(ge=0.0, le=1.0)
    top_p: float = Field(ge=0.0, le=1.0)
    system_role: str
    ended_naturally: bool


class RunRequest(BaseModel):
    prompt: str


class ModelPair(BaseModel):
    draft_model: str
    target_model: str
