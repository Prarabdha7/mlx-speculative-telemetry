import asyncio
import random
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Generator

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from app.engine.telemetry import KvCacheArchParams, TelemetryTracker
from app.schemas.metrics import RunMetrics, TokenTelemetry

PREFILL_STEP_SIZE = 512
GREEDY_TEMPERATURE_THRESHOLD = 1e-4
# Floor added before taking log() of a residual/target distribution so that
# zero-probability vocab entries don't produce -inf logits for mx.random.categorical.
RESIDUAL_EPSILON = 1e-12
# Fallback KV-cache dtype width (float16, MLX's default compute/activation
# dtype) used only when a cache has no allocated buffer yet to introspect.
DEFAULT_KV_PRECISION_BYTES = 2

# Adaptive lookahead: rolling window of the last N rounds' acceptance rates,
# and the acceptance-rate thresholds that raise/lower the active K.
ADAPTIVE_K_WINDOW = 5
ADAPTIVE_K_RAISE_THRESHOLD = 0.85
ADAPTIVE_K_LOWER_THRESHOLD = 0.40
ADAPTIVE_K_MIN = 1
ADAPTIVE_K_MAX = 8

_SENTINEL = object()


def _weight_nbytes(model) -> int:
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def _format_prompt(tokenizer, prompt: str) -> str:
    """Wraps a raw user prompt for an instruction-tuned model instead of
    handing it to the tokenizer as raw completion text — without this, the
    model continues the prompt as prose (e.g. narrating tutorial steps for a
    "write a function" request) rather than treating it as an instruction to
    follow. Uses the tokenizer's own chat template when the model ships one
    (every configured Llama-3.x Instruct model does), falling back to a
    manual instruction wrapper only for a tokenizer with no template at all.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "You are a coding assistant. Output ONLY the raw Python code requested. "
        "Do not output steps, markdown, or explanations.\n\n"
        f"User: {prompt}\nAssistant:"
    )


def _prefill(model, cache, tokens: mx.array, step_size: int = PREFILL_STEP_SIZE) -> mx.array:
    """Feeds all but the last token of `tokens` through `model`, leaving the
    final token unconsumed so the caller's next forward call produces its
    first real prediction."""
    while tokens.size > 1:
        n = min(step_size, tokens.size - 1)
        model(tokens[:n][None], cache=cache)
        tokens = tokens[n:]
    return tokens


class SpeculativeEngine:
    def __init__(
        self,
        draft_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        target_model: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
    ) -> None:
        self.draft_model_path = draft_model
        self.target_model_path = target_model

        self._draft = None
        self._target = None
        self._tokenizer = None
        self._eos_ids: set[int] = set()
        self._draft_weight_bytes = 0
        self._target_weight_bytes = 0
        self._target_num_layers = 0
        self._target_num_kv_heads = 0
        self._target_head_dim = 0
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._draft is not None:
            return
        with self._load_lock:
            if self._draft is not None:
                return
            self._draft, draft_tokenizer = load(self.draft_model_path)
            self._target, self._tokenizer = load(self.target_model_path)
            if draft_tokenizer.vocab_size != self._tokenizer.vocab_size:
                raise ValueError(
                    "Draft and target models must share a tokenizer/vocabulary "
                    "for token-level speculative verification to be valid "
                    f"(draft vocab={draft_tokenizer.vocab_size}, "
                    f"target vocab={self._tokenizer.vocab_size})."
                )
            self._eos_ids = set(self._tokenizer.eos_token_ids)
            self._draft_weight_bytes = _weight_nbytes(self._draft)
            self._target_weight_bytes = _weight_nbytes(self._target)

            target_args = self._target.args
            self._target_num_layers = target_args.num_hidden_layers
            self._target_num_kv_heads = target_args.num_key_value_heads or target_args.num_attention_heads
            self._target_head_dim = target_args.head_dim or (
                target_args.hidden_size // target_args.num_attention_heads
            )

    async def generate_stream(
        self,
        prompt: str,
        k_lookahead: int = 4,
        max_tokens: int = 128,
        temperature: float = 0.0,
        run_id: str | None = None,
    ) -> AsyncIterator[TokenTelemetry | RunMetrics]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_loaded)

        sync_gen = self._generate_sync(prompt, k_lookahead, max_tokens, temperature, run_id)
        while True:
            item = await loop.run_in_executor(None, next, sync_gen, _SENTINEL)
            if item is _SENTINEL:
                break
            yield item

    def _generate_sync(
        self,
        prompt: str,
        k_lookahead: int,
        max_tokens: int,
        temperature: float,
        run_id: str | None = None,
    ) -> Generator[TokenTelemetry | RunMetrics, None, None]:
        is_greedy = temperature <= GREEDY_TEMPERATURE_THRESHOLD

        formatted_prompt = _format_prompt(self._tokenizer, prompt)
        prompt_ids = mx.array(self._tokenizer.encode(formatted_prompt), mx.uint32)
        draft_cache = make_prompt_cache(self._draft)
        target_cache = make_prompt_cache(self._target)

        draft_y = _prefill(self._draft, draft_cache, prompt_ids)
        y = _prefill(self._target, target_cache, prompt_ids)

        # The cache's own buffer dtype reflects MLX's real runtime activation
        # precision (independent of the model's quantized weight bits), so
        # read it directly rather than assuming a fixed width; an empty cache
        # (single-token prompt, no prefill step run) falls back to fp16.
        first_cache_layer = target_cache[0] if target_cache else None
        kv_precision_bytes = (
            first_cache_layer.keys.dtype.size
            if first_cache_layer is not None and first_cache_layer.keys is not None
            else DEFAULT_KV_PRECISION_BYTES
        )
        kv_cache_arch = KvCacheArchParams(
            num_layers=self._target_num_layers,
            num_kv_heads=self._target_num_kv_heads,
            head_dim=self._target_head_dim,
            precision_bytes=kv_precision_bytes,
        )
        tracker = TelemetryTracker(self._draft_weight_bytes, self._target_weight_bytes, kv_cache_arch, run_id)

        # Adaptive tuning operates within [ADAPTIVE_K_MIN, ADAPTIVE_K_MAX]
        # regardless of the requested starting k_lookahead, so a request above
        # the ceiling doesn't get silently snapped down the first time the
        # raise condition fires.
        current_k = max(min(k_lookahead, ADAPTIVE_K_MAX), ADAPTIVE_K_MIN)
        recent_acceptance_rates: deque[float] = deque(maxlen=ADAPTIVE_K_WINDOW)

        position = 0
        while position < max_tokens:
            num_draft = min(current_k, max_tokens - position)

            draft_ids: list[int] = []
            draft_probs: list[mx.array] = []
            dy = draft_y
            for _ in range(num_draft):
                step_start = time.perf_counter()
                logits = self._draft(dy[None], cache=draft_cache)[0, -1, :]
                scaled = logits if is_greedy else logits / temperature
                probs = mx.softmax(scaled, axis=-1)
                token = mx.argmax(scaled, axis=-1) if is_greedy else mx.random.categorical(scaled)
                mx.eval(token, probs)
                tracker.record_draft_step(time.perf_counter() - step_start)

                token_id = int(token.item())
                draft_ids.append(token_id)
                draft_probs.append(probs)
                dy = token.reshape(1)

            candidate = mx.concatenate([y, mx.array(draft_ids, mx.uint32)])
            verify_start = time.perf_counter()
            target_logits = self._target(candidate[None], cache=target_cache)[0]
            scaled_target = target_logits if is_greedy else target_logits / temperature
            target_probs = mx.softmax(scaled_target, axis=-1)
            mx.eval(target_probs)
            verify_latency_s = time.perf_counter() - verify_start
            tracker.record_target_pass(verify_latency_s)
            verify_latency_ms = verify_latency_s * 1000

            accepted = 0
            commit_id = None
            commit_status = None
            commit_draft_p = None
            commit_target_p = None
            reached_limit = False

            for i in range(num_draft):
                draft_token_id = draft_ids[i]
                draft_p = float(draft_probs[i][draft_token_id].item())
                target_p = float(target_probs[i, draft_token_id].item())

                if is_greedy:
                    accept_probability = None
                    is_accepted = int(mx.argmax(target_logits[i]).item()) == draft_token_id
                else:
                    accept_probability = min(1.0, target_p / max(draft_p, RESIDUAL_EPSILON))
                    is_accepted = random.random() < accept_probability

                if is_accepted:
                    tracker.record_token("accepted")
                    yield TokenTelemetry(
                        token=self._tokenizer.decode([draft_token_id]),
                        token_id=draft_token_id,
                        status="accepted",
                        position=position,
                        draft_probability=draft_p,
                        target_probability=target_p,
                        accept_probability=accept_probability,
                        latency_ms=verify_latency_ms / num_draft,
                        timestamp=time.time(),
                    )
                    position += 1
                    accepted += 1
                    if position == max_tokens:
                        reached_limit = True
                        break
                    continue

                tracker.record_token("rejected")
                yield TokenTelemetry(
                    token=self._tokenizer.decode([draft_token_id]),
                    token_id=draft_token_id,
                    status="rejected",
                    position=position,
                    draft_probability=draft_p,
                    target_probability=target_p,
                    accept_probability=accept_probability,
                    latency_ms=verify_latency_ms / num_draft,
                    timestamp=time.time(),
                )

                if is_greedy:
                    commit_id = int(mx.argmax(target_logits[i]).item())
                else:
                    # Resample the correction from the residual distribution
                    # max(0, p_target - q_draft): the probability mass the
                    # draft model under-weighted relative to the target,
                    # renormalized by the implicit softmax inside categorical().
                    residual = mx.maximum(target_probs[i] - draft_probs[i], 0.0)
                    commit_id = int(mx.random.categorical(mx.log(residual + RESIDUAL_EPSILON)).item())
                commit_status = "correction"
                commit_draft_p = draft_p
                commit_target_p = float(target_probs[i, commit_id].item())
                break

            if not reached_limit and commit_id is None and accepted == num_draft:
                # Every drafted token was accepted: the target model's pass
                # already scored one position past the last draft token, so
                # that bonus token is free — no extra forward pass needed.
                bonus_logits = target_logits[num_draft]
                commit_id = (
                    int(mx.argmax(bonus_logits).item())
                    if is_greedy
                    else int(mx.random.categorical(bonus_logits / temperature).item())
                )
                commit_status = "bonus"
                commit_target_p = float(target_probs[num_draft, commit_id].item())

            if commit_id is not None:
                tracker.record_token(commit_status)
                yield TokenTelemetry(
                    token=self._tokenizer.decode([commit_id]),
                    token_id=commit_id,
                    status=commit_status,
                    position=position,
                    draft_probability=commit_draft_p,
                    target_probability=commit_target_p,
                    accept_probability=None,
                    latency_ms=verify_latency_ms / num_draft,
                    timestamp=time.time(),
                )
                position += 1

            trim_prompt_cache(target_cache, num_draft - accepted)
            trim_prompt_cache(draft_cache, max(num_draft - accepted - 1, 0))

            # Self-tuning lookahead: a rolling average (not a single round's
            # rate) over the last few rounds absorbs per-round noise before
            # nudging K, so one lucky/unlucky round doesn't whipsaw it.
            recent_acceptance_rates.append(accepted / num_draft)
            rolling_acceptance_rate = sum(recent_acceptance_rates) / len(recent_acceptance_rates)
            if rolling_acceptance_rate > ADAPTIVE_K_RAISE_THRESHOLD:
                current_k = min(current_k + 1, ADAPTIVE_K_MAX)
            elif rolling_acceptance_rate < ADAPTIVE_K_LOWER_THRESHOLD:
                current_k = max(current_k - 1, ADAPTIVE_K_MIN)

            yield tracker.snapshot(sequence_length=position, current_k_lookahead=current_k, is_final=False)

            if reached_limit or commit_id in self._eos_ids:
                break

            y = mx.array([commit_id], mx.uint32)
            draft_y = y
            if accepted == num_draft:
                # The draft model never fed its own last proposal back into
                # itself, so the next round's first draft step must catch the
                # draft cache up with it before producing a new token.
                draft_y = mx.concatenate([mx.array([draft_ids[-1]], mx.uint32), draft_y])

        yield tracker.snapshot(sequence_length=position, current_k_lookahead=current_k, is_final=True)
