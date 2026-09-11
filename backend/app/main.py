import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.engine.mlx_speculative import SpeculativeEngine
from app.schemas.metrics import TokenTelemetry

# Next.js dev server origin; override with a comma-separated list in .env
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(title="mlx-speculative-telemetry", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SpeculativeEngine()


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            request = await websocket.receive_json()
            stream = engine.generate_stream(
                prompt=request.get("prompt", ""),
                k_lookahead=int(request.get("k_lookahead", 4)),
            )
            async for event in stream:
                event_type = "token" if isinstance(event, TokenTelemetry) else "metrics"
                await websocket.send_json({"type": event_type, "data": event.model_dump()})
    except WebSocketDisconnect:
        pass
