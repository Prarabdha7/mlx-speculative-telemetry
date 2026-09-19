import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.db.models import BenchmarkRun, init_db, list_benchmark_runs
from app.engine.mlx_speculative import SpeculativeEngine
from app.engine.runs import QUEUE_DONE, RunRegistry
from app.schemas.metrics import ModelPair, RunRequest, TokenTelemetry

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit")
TARGET_MODEL = os.getenv("TARGET_MODEL", "mlx-community/Llama-3.1-8B-Instruct-4bit")

PRESET_PROMPTS = [
    "Write one sentence about the ocean.",
    "Explain what a hash table is in two sentences.",
    "List three benefits of unit testing.",
    "Describe Apple Silicon's unified memory architecture in one paragraph.",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="mlx-speculative-telemetry", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SpeculativeEngine(draft_model=DRAFT_MODEL, target_model=TARGET_MODEL)
runs = RunRegistry(engine)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def list_models() -> ModelPair:
    return ModelPair(draft_model=DRAFT_MODEL, target_model=TARGET_MODEL)


@app.get("/api/prompts")
def list_prompts() -> list[str]:
    return PRESET_PROMPTS


@app.get("/api/benchmarks")
def list_benchmarks() -> list[BenchmarkRun]:
    return list_benchmark_runs()


@app.post("/api/runs")
async def start_run(request: RunRequest) -> dict[str, str]:
    run_id = runs.start(request)
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, bool]:
    stopped = runs.stop(run_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"No active run with id '{run_id}'")
    return {"stopped": True}


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        run_id = websocket.query_params.get("run_id")
        handle = runs.get(run_id) if run_id else None
        if handle is None:
            await websocket.close(code=4004, reason=f"No active run with id '{run_id}'")
            return

        while True:
            event = await handle.queue.get()
            if event is QUEUE_DONE:
                break
            event_type = "token" if isinstance(event, TokenTelemetry) else "metrics"
            await websocket.send_json({"type": event_type, "data": event.model_dump()})
        await websocket.close()
    except WebSocketDisconnect:
        pass
