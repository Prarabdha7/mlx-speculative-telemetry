import asyncio
import random
import re
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

DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate, and concise AI assistant."
# Named stop tokens across common chat-model families, unioned with the
# tokenizer's own configured eos ids: a model's tokenizer_config.json doesn't
# always list every terminator its template can emit (e.g. a Qwen-family
# model whose eos_token_id config predates <|im_end|> being added), so this
# widens coverage defensively rather than trusting a single config field.
NAMED_STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"]

# Autonomous parameter engine: a feature-scoring heuristic (not a fixed
# keyword-to-bucket lookup) that derives continuous generation parameters
# from a prompt's estimated determinism vs. creativity, rather than sorting
# every prompt into one of three hardcoded parameter tuples.
MAX_ALLOWED_TEMPERATURE = 1.0
_TEMPERATURE_FLOOR = 0.1  # non-code deterministic prompts asymptote here, never all the way to 0.0
_TEMPERATURE_CEILING = 0.8

# Deliberately excludes bare "(" / ")" from the syntax class: parenthetical
# asides are common in ordinary prose ("(briefly)") and would otherwise
# false-trigger code mode; "{" / "}" / ";" / "+" are rare enough outside
# code and math to be reliable signals on their own.
_CODE_SYNTAX_PATTERN = re.compile(r"def\s|```|[{};]|\+|\bimport\b|\bclass\b|\bfunction\b|\bsql\b")
_DETERMINISM_WORDS = re.compile(r"\ball\b|\bdetails?\b|\blist\b|\bexact\b|\bprecise\b|\bcalculate\b|\bsolve\b")
_FACTUAL_WORDS = re.compile(r"\bwhat is\b|\bwho is\b|\bwhen\b|\bhistory\b|\bdefinition\b|\bexplain\b|\bsummarize\b")
_CREATIVITY_WORDS = re.compile(
    r"\bwrite\b|\bstory\b|\bimagine\b|\bcreative\b|\bpoem\b|\bfiction\b|\bbrainstorm\b|\binvent\b|\bdream\b|\bdesign\b"
)
_STRONG_LENGTH_SIGNALS = re.compile(r"\blist all\b|\ball the\b|\bcomprehensive\b|\bin depth\b|\bevery\b")
_MODERATE_LENGTH_SIGNALS = re.compile(r"\bexplain\b|\bdetails?\b|\bsummarize\b|\bdescribe\b|\bhistory of\b")


def calculate_dynamic_parameters(prompt: str) -> dict:
    """Scores a prompt's determinism vs. creativity and its expected output
    length, deriving continuous generation parameters rather than sorting
    it into one of a few fixed buckets.

    Determinism signals: code/math syntax, digit density, and exhaustive or
    precision-demanding language ("all", "details", "exact"). Creativity
    signals: open-ended, generative language ("write", "imagine", "story").
    Pure code prompts are special-cased to temperature 0.0 (fully greedy) —
    every other prompt's temperature asymptotes toward, but never reaches,
    _TEMPERATURE_FLOOR as determinism dominates, and toward _TEMPERATURE_CEILING
    as creativity dominates, on a continuous scale between the two.
    """
    lowered = prompt.lower()

    is_code = bool(_CODE_SYNTAX_PATTERN.search(lowered))
    numeral_hits = len(re.findall(r"\d", lowered))
    determinism_word_hits = len(_DETERMINISM_WORDS.findall(lowered))
    factual_word_hits = len(_FACTUAL_WORDS.findall(lowered))
    creativity_word_hits = len(_CREATIVITY_WORDS.findall(lowered))

    determinism_score = numeral_hits + determinism_word_hits * 2 + factual_word_hits * 2
    creativity_score = creativity_word_hits * 3

    strong_length = bool(_STRONG_LENGTH_SIGNALS.search(lowered))
    moderate_length = bool(_MODERATE_LENGTH_SIGNALS.search(lowered))
    max_tokens = 2048 if strong_length else 1024 if moderate_length else 256

    if is_code:
        return {
            "temperature": 0.0,
            "lookahead_k": 5,
            "max_tokens": max_tokens,
            "detected_intent": "Deterministic Code / Math Generation",
        }

    total_signal = determinism_score + creativity_score
    # net in [-1, 1]: -1 = purely deterministic, +1 = purely creative, 0 = no signal either way.
    net = 0.0 if total_signal == 0 else (creativity_score - determinism_score) / total_signal
    temperature = round(max(_TEMPERATURE_FLOOR, min(_TEMPERATURE_CEILING, 0.45 + net * 0.35)), 2)

    if determinism_score >= creativity_score:
        lookahead_k = 4
        if strong_length:
            detected_intent = "Strict Factual / Historical Listing"
        elif factual_word_hits > 0:
            detected_intent = "Explanatory Factual Response"
        else:
            detected_intent = "Deterministic / Analytical Query"
    else:
        lookahead_k = 3
        detected_intent = "Open-Ended Creative Generation" if net > 0.6 else "General Conversational Response"

    return {
        "temperature": temperature,
        "lookahead_k": lookahead_k,
        "max_tokens": max_tokens,
        "detected_intent": detected_intent,
    }

_SENTINEL = object()


def _weight_nbytes(model) -> int:
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def _format_prompt(tokenizer, prompt: str) -> str:
    """Wraps a raw user prompt for an instruction-tuned model instead of
    handing it to the tokenizer as raw completion text — without this, the
    model continues the prompt as prose (e.g. narrating tutorial steps for a
    "write a function" request, or drifting into unrelated content on an
    open-ended question) rather than treating it as an instruction to follow.
    A general system prompt keeps this consistent across every prompt type
    (coding, knowledge, reasoning), not just one category. Uses the
    tokenizer's own chat template when the model ships one (every configured
    Llama-3.x Instruct model does); falls back to a manual instruction
    wrapper only for a tokenizer whose template application fails or is
    absent entirely.
    """
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {DEFAULT_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"


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
        self._stop_token_ids: set[int] = set()
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
            self._stop_token_ids = set(self._tokenizer.eos_token_ids)
            vocab = self._tokenizer.get_vocab()
            for token_str in NAMED_STOP_TOKENS:
                if token_str in vocab:
                    self._stop_token_ids.add(vocab[token_str])
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
        auto_tune: bool = True,
    ) -> AsyncIterator[TokenTelemetry | RunMetrics]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_loaded)

        sync_gen = self._generate_sync(prompt, k_lookahead, max_tokens, temperature, run_id, auto_tune)
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
        auto_tune: bool = True,
    ) -> Generator[TokenTelemetry | RunMetrics, None, None]:
        # Guardrail: applies regardless of auto_tune, so a manually-supplied
        # temperature outside the model's sane sampling range never reaches
        # generation even if a client bypasses the frontend slider's own cap.
        temperature = max(0.0, min(temperature, MAX_ALLOWED_TEMPERATURE))
        if auto_tune:
            params = calculate_dynamic_parameters(prompt)
            temperature = params["temperature"]
            k_lookahead = params["lookahead_k"]
            max_tokens = params["max_tokens"]

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
        tracker = TelemetryTracker(
            self._draft_weight_bytes, self._target_weight_bytes, kv_cache_arch, temperature, run_id
        )

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
            hit_stop = False

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
                    if draft_token_id in self._stop_token_ids:
                        # End the turn here without emitting the stop token
                        # itself, so its control tag never reaches the
                        # WebSocket stream or renders in the UI.
                        hit_stop = True
                        break
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

            if not reached_limit and not hit_stop and commit_id is None and accepted == num_draft:
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
                if commit_id in self._stop_token_ids:
                    # Same sanitization as the accepted-draft-token case above:
                    # a correction/bonus token that turns out to be a stop
                    # token ends the turn without ever being yielded.
                    hit_stop = True
                else:
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

            if reached_limit or hit_stop:
                break

            y = mx.array([commit_id], mx.uint32)
            draft_y = y
            if accepted == num_draft:
                # The draft model never fed its own last proposal back into
                # itself, so the next round's first draft step must catch the
                # draft cache up with it before producing a new token.
                draft_y = mx.concatenate([mx.array([draft_ids[-1]], mx.uint32), draft_y])

        yield tracker.snapshot(sequence_length=position, current_k_lookahead=current_k, is_final=True)
