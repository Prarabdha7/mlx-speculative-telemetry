import time
import uuid
from dataclasses import dataclass

from app.schemas.metrics import RunMetrics, TokenStatus


@dataclass(frozen=True)
class KvCacheArchParams:
    """Target-model architecture constants needed to size its KV cache.

    `num_kv_heads` is deliberately the model's key/value head count, not its
    total attention-head count: grouped-query-attention architectures like
    Llama 3 project far fewer KV heads than query heads (Llama-3.1-8B: 8 vs
    32), and sizing off the query-head count would overstate the real cache
    footprint by that same factor.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    precision_bytes: int


def kv_cache_size_bytes(arch: KvCacheArchParams, sequence_length: int, batch_size: int = 1) -> int:
    """Theoretical KV-cache footprint across every transformer layer:
    2 (key + value) * layers * batch * seq_len * kv_heads * head_dim * dtype_size.
    """
    return (
        2
        * batch_size
        * sequence_length
        * arch.num_kv_heads
        * arch.head_dim
        * arch.precision_bytes
        * arch.num_layers
    )


class TelemetryTracker:
    """Accumulates per-token and per-forward-pass stats for a single run and
    derives the aggregate figures the dashboard gauges display."""

    def __init__(
        self,
        draft_weight_bytes: int,
        target_weight_bytes: int,
        kv_cache_arch: KvCacheArchParams,
        actual_temperature: float,
        top_p: float,
        system_role: str,
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.draft_weight_bytes = draft_weight_bytes
        self.target_weight_bytes = target_weight_bytes
        self.kv_cache_arch = kv_cache_arch
        self.actual_temperature = actual_temperature
        self.top_p = top_p
        self.system_role = system_role

        self._start = time.perf_counter()
        self._status_counts: dict[TokenStatus, int] = {
            "accepted": 0,
            "rejected": 0,
            "correction": 0,
            "bonus": 0,
        }
        self._draft_forward_calls = 0
        self._draft_time_s = 0.0
        self._target_forward_calls = 0
        self._target_time_s = 0.0

    def record_draft_step(self, latency_s: float) -> None:
        self._draft_forward_calls += 1
        self._draft_time_s += latency_s

    def record_target_pass(self, latency_s: float) -> None:
        self._target_forward_calls += 1
        self._target_time_s += latency_s

    def record_token(self, status: TokenStatus) -> None:
        self._status_counts[status] += 1

    @property
    def committed_tokens(self) -> int:
        # every status except "rejected" corresponds to a token that actually
        # lands in the output sequence; a rejection only discards a proposal.
        return (
            self._status_counts["accepted"]
            + self._status_counts["correction"]
            + self._status_counts["bonus"]
        )

    def snapshot(
        self,
        sequence_length: int,
        current_k_lookahead: int,
        ended_naturally: bool = False,
        is_final: bool = False,
    ) -> RunMetrics:
        elapsed_s = max(time.perf_counter() - self._start, 1e-9)
        committed = self.committed_tokens

        draft_tps = self._draft_forward_calls / max(self._draft_time_s, 1e-9)
        # A target-only baseline calls the target model once per emitted token;
        # our verify pass covers several draft positions per call, but on
        # Apple Silicon's unified memory the per-call latency is dominated by
        # the one-time weight read rather than the extra sequence positions,
        # so treating "per verify-pass latency" as a stand-in for "per solo
        # decode step latency" is a reasonable approximation of that baseline.
        target_tps = self._target_forward_calls / max(self._target_time_s, 1e-9)
        effective_tps = committed / elapsed_s
        speedup_ratio = effective_tps / target_tps if target_tps > 0 else 0.0

        proposed = self._status_counts["accepted"] + self._status_counts["rejected"]
        acceptance_rate = self._status_counts["accepted"] / proposed if proposed else 0.0

        bandwidth_bytes = (
            self._draft_forward_calls * self.draft_weight_bytes
            + self._target_forward_calls * self.target_weight_bytes
        )
        memory_bandwidth_gbps = bandwidth_bytes / elapsed_s / 1e9
        kv_cache_mb = kv_cache_size_bytes(self.kv_cache_arch, sequence_length) / 1e6

        return RunMetrics(
            run_id=self.run_id,
            is_final=is_final,
            elapsed_s=elapsed_s,
            total_tokens=committed,
            draft_tokens_per_second=draft_tps,
            target_tokens_per_second=target_tps,
            effective_tokens_per_second=effective_tps,
            speedup_ratio=speedup_ratio,
            acceptance_rate=acceptance_rate,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            kv_cache_mb=kv_cache_mb,
            current_k_lookahead=current_k_lookahead,
            actual_temperature=self.actual_temperature,
            top_p=self.top_p,
            system_role=self.system_role,
            ended_naturally=ended_naturally,
        )
