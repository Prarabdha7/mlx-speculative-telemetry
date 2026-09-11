import asyncio
import random
import time
from collections.abc import AsyncIterator

from app.schemas.metrics import EngineMetrics, TokenTelemetry

DRAFT_STEP_LATENCY_S = 0.010
TARGET_VERIFY_LATENCY_S = 0.040
DRAFT_ACCEPT_PROBABILITY = 0.72

# Placeholder weight-read volume for one target forward pass, used to derive a
# plausible bandwidth figure until real MLX memory counters are wired in.
TARGET_PASS_BYTES = 4.3e9

_MOCK_TOKENS = (
    "Apple",
    " Silicon",
    " runs",
    " the",
    " draft",
    " and",
    " target",
    " models",
    " against",
    " a",
    " single",
    " unified",
    " memory",
    " pool",
    ",",
    " so",
    " speculative",
    " decoding",
    " avoids",
    " host",
    " transfers",
    ".",
)


class SpeculativeEngine:
    def __init__(
        self,
        draft_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        target_model: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model

    async def generate_stream(
        self,
        prompt: str,
        k_lookahead: int = 4,
        max_tokens: int = 48,
    ) -> AsyncIterator[TokenTelemetry | EngineMetrics]:
        run_start = time.perf_counter()
        position = 0
        target_passes = 0

        while position < max_tokens:
            window_start = time.perf_counter()
            await asyncio.sleep(DRAFT_STEP_LATENCY_S * k_lookahead)
            await asyncio.sleep(TARGET_VERIFY_LATENCY_S)
            target_passes += 1

            accepted_run = 0
            while accepted_run < k_lookahead and random.random() < DRAFT_ACCEPT_PROBABILITY:
                accepted_run += 1

            # The target pass always contributes one extra token: a resampled
            # correction when a draft token was rejected, or a free bonus token
            # when the whole look-ahead window was accepted.
            window_size = min(accepted_run + 1, max_tokens - position)
            window_latency_ms = (time.perf_counter() - window_start) * 1000 / window_size

            for offset in range(window_size):
                is_draft_token = offset < accepted_run
                yield TokenTelemetry(
                    token=_MOCK_TOKENS[position % len(_MOCK_TOKENS)],
                    accepted=is_draft_token or accepted_run == k_lookahead,
                    probability=random.uniform(0.6, 0.99) if is_draft_token else random.uniform(0.2, 0.6),
                    latency_ms=window_latency_ms,
                    position=position,
                )
                position += 1

        elapsed_s = time.perf_counter() - run_start
        yield EngineMetrics(
            tokens_per_second=position / elapsed_s,
            memory_bandwidth_gbps=(target_passes * TARGET_PASS_BYTES) / elapsed_s / 1e9,
        )
