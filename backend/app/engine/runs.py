import asyncio
import time
import uuid
from dataclasses import dataclass, field

from app.db.models import BenchmarkRun, insert_benchmark_run
from app.engine.mlx_speculative import SpeculativeEngine
from app.schemas.metrics import RunMetrics, RunRequest, TokenTelemetry

QUEUE_DONE = object()


@dataclass
class RunHandle:
    task: asyncio.Task
    queue: "asyncio.Queue[TokenTelemetry | RunMetrics | object]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=256)
    )


class RunRegistry:
    """Tracks the background generation task and event queue for every run
    started via the REST API, so a WebSocket client can attach to a run by id
    independently of when it started."""

    def __init__(self, engine: SpeculativeEngine) -> None:
        self._engine = engine
        self._runs: dict[str, RunHandle] = {}

    def start(self, request: RunRequest) -> str:
        run_id = uuid.uuid4().hex[:12]
        handle = RunHandle(task=None)
        handle.task = asyncio.create_task(self._drive(handle, request, run_id))
        self._runs[run_id] = handle
        return run_id

    def stop(self, run_id: str) -> bool:
        handle = self._runs.get(run_id)
        if handle is None or handle.task.done():
            return False
        handle.task.cancel()
        return True

    def get(self, run_id: str) -> RunHandle | None:
        return self._runs.get(run_id)

    async def _drive(self, handle: RunHandle, request: RunRequest, run_id: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            async for event in self._engine.generate_stream(
                prompt=request.prompt,
                k_lookahead=request.k_lookahead,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                run_id=run_id,
            ):
                if isinstance(event, RunMetrics) and event.is_final:
                    # Persisted before the event reaches the WebSocket client so a
                    # client that reacts to `is_final` by immediately refetching
                    # /api/benchmarks is guaranteed to see this run's row.
                    benchmark = BenchmarkRun(
                        id=run_id,
                        timestamp=time.time(),
                        draft_model=self._engine.draft_model_path,
                        target_model=self._engine.target_model_path,
                        k_lookahead=request.k_lookahead,
                        temperature=request.temperature,
                        prompt=request.prompt,
                        total_tokens=event.total_tokens,
                        acceptance_rate=event.acceptance_rate,
                        speedup_ratio=event.speedup_ratio,
                        effective_tokens_per_second=event.effective_tokens_per_second,
                        elapsed_s=event.elapsed_s,
                    )
                    await loop.run_in_executor(None, insert_benchmark_run, benchmark)
                await handle.queue.put(event)
        except asyncio.CancelledError:
            pass
        finally:
            await handle.queue.put(QUEUE_DONE)
