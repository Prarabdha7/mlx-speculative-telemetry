from pydantic import BaseModel, Field

from app.db.session import get_session

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    draft_model TEXT NOT NULL,
    target_model TEXT NOT NULL,
    k_lookahead INTEGER NOT NULL,
    temperature REAL NOT NULL,
    prompt TEXT NOT NULL,
    total_tokens INTEGER NOT NULL,
    acceptance_rate REAL NOT NULL,
    speedup_ratio REAL NOT NULL,
    effective_tokens_per_second REAL NOT NULL,
    elapsed_s REAL NOT NULL
)
"""


class BenchmarkRun(BaseModel):
    id: str
    timestamp: float
    draft_model: str
    target_model: str
    k_lookahead: int = Field(ge=1)
    temperature: float = Field(ge=0.0)
    prompt: str
    total_tokens: int = Field(ge=0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)
    speedup_ratio: float = Field(ge=0.0)
    effective_tokens_per_second: float = Field(ge=0.0)
    elapsed_s: float = Field(ge=0.0)


def init_db() -> None:
    with get_session() as conn:
        conn.execute(SCHEMA)


def insert_benchmark_run(run: BenchmarkRun) -> None:
    with get_session() as conn:
        conn.execute(
            """
            INSERT INTO benchmark_runs (
                id, timestamp, draft_model, target_model, k_lookahead,
                temperature, prompt, total_tokens, acceptance_rate,
                speedup_ratio, effective_tokens_per_second, elapsed_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.timestamp,
                run.draft_model,
                run.target_model,
                run.k_lookahead,
                run.temperature,
                run.prompt,
                run.total_tokens,
                run.acceptance_rate,
                run.speedup_ratio,
                run.effective_tokens_per_second,
                run.elapsed_s,
            ),
        )


def list_benchmark_runs() -> list[BenchmarkRun]:
    with get_session() as conn:
        rows = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY timestamp DESC"
        ).fetchall()
    return [BenchmarkRun(**dict(row)) for row in rows]
