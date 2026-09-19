import asyncio
import json
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

# Named stop tokens across common chat-model families, unioned with the
# tokenizer's own configured eos ids: a model's tokenizer_config.json doesn't
# always list every terminator its template can emit (e.g. a Qwen-family
# model whose eos_token_id config predates <|im_end|> being added), so this
# widens coverage defensively rather than trusting a single config field.
NAMED_STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"]

# Generation is bounded only by natural EOS/EOT termination now — this is a
# safety net against a pathological run that never produces a stop token,
# not a per-category budget.
SAFETY_MAX_TOKENS_CEILING = 8192

# Neural meta-planner: instead of a keyword/regex heuristic, the 1B draft
# model classifies its own prompt and self-selects a generation profile.
#
# A 1B instruct model asked cold to "classify, don't answer" reliably just
# answers the user's request instead (verified directly: asking it to
# classify a coding prompt produced a working Fibonacci function, not JSON,
# even with a very forceful system prompt). Two things fixed this, both
# verified empirically before shipping: (1) few-shot examples spanning code,
# math, factual, and creative prompts — a bare instruction wasn't enough,
# concrete examples of the exact behavior wanted were; (2) forcing the
# completion to start mid-JSON (`{"temperature":`) rather than hoping the
# model opens the object itself — a well-known guided-generation technique
# that turned "usually forgets the opening brace" into reliable, parseable
# output across every category tested.
PLANNER_MAX_NEW_TOKENS = 100
PLANNER_SYSTEM_PROMPT = (
    "You are a JSON classifier for a text-generation router. You output ONLY a JSON object, "
    "never code, stories, or direct answers to the request."
)
PLANNER_FEW_SHOT_EXAMPLES = [
    (
        "Classify: Write a Python function that returns the nth Fibonacci number using memoization.",
        '{"temperature": 0.0, "top_p": 0.9, "initial_k": 5, "system_role": "You are a concise software engineer."}',
    ),
    (
        "Classify: Solve for x: 2x + 5 = 15",
        '{"temperature": 0.0, "top_p": 0.9, "initial_k": 5, "system_role": "You are a precise mathematician."}',
    ),
    (
        "Classify: What year did the Berlin Wall fall?",
        '{"temperature": 0.1, "top_p": 0.9, "initial_k": 4, "system_role": "You are a precise, factual historian."}',
    ),
    (
        "Classify: Write a short story about a lighthouse keeper.",
        '{"temperature": 0.7, "top_p": 0.95, "initial_k": 3, "system_role": "You are a creative fiction writer."}',
    ),
]
PLANNER_FORCED_PREFIX = '{"temperature":'
PLANNER_FALLBACK_PROFILE = {
    "temperature": 0.1,
    "top_p": 0.9,
    "initial_k": 4,
    "system_role": "You are a helpful AI assistant.",
}
_JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


def _run_completion(model, tokenizer, formatted_prompt: str, stop_token_ids: set[int], max_new_tokens: int) -> str:
    """Runs a plain greedy completion (no speculative decoding, no
    telemetry) — used for the planner's own cheap self-classification pass,
    never for real user-facing generation."""
    prompt_ids = mx.array(tokenizer.encode(formatted_prompt), mx.uint32)
    cache = make_prompt_cache(model)
    y = _prefill(model, cache, prompt_ids)

    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(y[None], cache=cache)[0, -1, :]
        token_id = int(mx.argmax(logits).item())
        if token_id in stop_token_ids:
            break
        generated_ids.append(token_id)
        y = mx.array([token_id], mx.uint32)
    return tokenizer.decode(generated_ids)


def plan_execution_profile(prompt: str, draft_model, tokenizer, stop_token_ids: set[int]) -> dict:
    """Asks the small draft model to classify the prompt and self-select a
    generation profile, in place of a fixed keyword/regex heuristic. Runs
    entirely on the 1B draft model (never the target) since this is a cheap
    routing decision, not real generation — a few hundred milliseconds of
    extra latency per run, traded for genuinely prompt-derived parameters
    instead of a hand-tuned scoring formula.

    Missing or invalid individual fields fall back one at a time (a 1B model
    asked for four-key structured output won't always get all four right),
    and the whole profile falls back to a fixed safe default only if the
    model's output can't be parsed as JSON at all — this must never crash a run.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
    for example_prompt, example_response in PLANNER_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_prompt})
        messages.append({"role": "assistant", "content": example_response})
    messages.append({"role": "user", "content": f"Classify: {prompt}"})

    try:
        planner_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        planner_prompt += PLANNER_FORCED_PREFIX
        raw_output = _run_completion(draft_model, tokenizer, planner_prompt, stop_token_ids, PLANNER_MAX_NEW_TOKENS)

        match = _JSON_OBJECT_PATTERN.search(PLANNER_FORCED_PREFIX + raw_output)
        if not match:
            return dict(PLANNER_FALLBACK_PROFILE)
        parsed = json.loads(match.group(0))
    except Exception:
        return dict(PLANNER_FALLBACK_PROFILE)

    fallback = PLANNER_FALLBACK_PROFILE

    def _numeric_field(key: str, lo: float, hi: float, cast: type) -> float | int:
        try:
            return max(lo, min(cast(parsed[key]), hi))
        except (KeyError, TypeError, ValueError):
            return fallback[key]

    system_role = str(parsed.get("system_role") or "").strip() or fallback["system_role"]

    return {
        "temperature": _numeric_field("temperature", 0.0, 0.8, float),
        "top_p": _numeric_field("top_p", 0.85, 1.0, float),
        "initial_k": int(_numeric_field("initial_k", 2, 6, int)),
        "system_role": system_role,
    }


def _sample_with_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Nucleus-samples a token id from `logits`. Only the smallest prefix of
    tokens (in descending probability order) whose cumulative probability
    reaches `top_p` is eligible; everything else is masked to -inf before
    sampling, which keeps `mx.random.categorical`'s implicit softmax
    correctly renormalized over just that nucleus.
    """
    if top_p >= 1.0:
        return mx.random.categorical(logits)
    sorted_indices = mx.argsort(-logits)
    sorted_logits = logits[sorted_indices]
    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    # Shifted by one so the token that CROSSES the top_p threshold is kept
    # (otherwise the nucleus could end up empty when the top token alone
    # already exceeds top_p).
    keep_mask = mx.concatenate([mx.array([True]), cumulative_probs[:-1] <= top_p])
    masked_sorted_logits = mx.where(keep_mask, sorted_logits, mx.array(float("-inf")))
    unsorted_logits = masked_sorted_logits[mx.argsort(sorted_indices)]
    return mx.random.categorical(unsorted_logits)


_SENTINEL = object()


def _weight_nbytes(model) -> int:
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def _format_prompt(tokenizer, prompt: str, system_role: str) -> str:
    """Wraps a raw user prompt for an instruction-tuned model instead of
    handing it to the tokenizer as raw completion text — without this, the
    model continues the prompt as prose (e.g. narrating tutorial steps for a
    "write a function" request, or drifting into unrelated content on an
    open-ended question) rather than treating it as an instruction to follow.
    `system_role` is the meta-planner's own per-prompt system instruction
    rather than one fixed string, so a math question and a creative-writing
    request get differently-tailored framing. Uses the tokenizer's own chat
    template when the model ships one (every configured Llama-3.x Instruct
    model does); falls back to a manual instruction wrapper only for a
    tokenizer whose template application fails or is absent entirely.
    """
    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system_role}\n\nUser: {prompt}\n\nAssistant:"


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
        run_id: str | None = None,
    ) -> AsyncIterator[TokenTelemetry | RunMetrics]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_loaded)

        sync_gen = self._generate_sync(prompt, run_id)
        while True:
            item = await loop.run_in_executor(None, next, sync_gen, _SENTINEL)
            if item is _SENTINEL:
                break
            yield item

    def _generate_sync(
        self,
        prompt: str,
        run_id: str | None = None,
    ) -> Generator[TokenTelemetry | RunMetrics, None, None]:
        # Every generation criterion (temperature, top_p, starting lookahead,
        # system persona) is inferred from the prompt itself by the 1B draft
        # model — there is no manual/heuristic parameter path left to fall
        # back to.
        profile = plan_execution_profile(prompt, self._draft, self._tokenizer, self._stop_token_ids)
        temperature = profile["temperature"]
        top_p = profile["top_p"]
        k_lookahead = profile["initial_k"]
        system_role = profile["system_role"]
        max_tokens = SAFETY_MAX_TOKENS_CEILING

        is_greedy = temperature <= GREEDY_TEMPERATURE_THRESHOLD

        formatted_prompt = _format_prompt(self._tokenizer, prompt, system_role)
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
            self._draft_weight_bytes,
            self._target_weight_bytes,
            kv_cache_arch,
            temperature,
            top_p,
            system_role,
            run_id,
        )

        # Adaptive tuning operates within [ADAPTIVE_K_MIN, ADAPTIVE_K_MAX];
        # the planner's own initial_k is already clamped to [2, 6] (a subset
        # of this range), so this is a defensive invariant rather than a
        # live constraint on typical planner output.
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
                # top_p only ever narrows which token gets PROPOSED here; the
                # accept/reject probabilities below are always computed from
                # the full, untruncated `probs`/`target_probs` distributions,
                # so nucleus sampling can't perturb the speculative-sampling
                # math itself (it only changes what draft_ids[i] ends up being).
                token = mx.argmax(scaled, axis=-1) if is_greedy else _sample_with_top_p(scaled, top_p)
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
                    commit_id = int(_sample_with_top_p(mx.log(residual + RESIDUAL_EPSILON), top_p).item())
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
                    else int(_sample_with_top_p(bonus_logits / temperature, top_p).item())
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

            yield tracker.snapshot(
                sequence_length=position, current_k_lookahead=current_k, ended_naturally=False, is_final=False
            )

            if reached_limit or hit_stop:
                break

            y = mx.array([commit_id], mx.uint32)
            draft_y = y
            if accepted == num_draft:
                # The draft model never fed its own last proposal back into
                # itself, so the next round's first draft step must catch the
                # draft cache up with it before producing a new token.
                draft_y = mx.concatenate([mx.array([draft_ids[-1]], mx.uint32), draft_y])

        # "Natural" means the run ended because the model itself produced a
        # stop token, not because the 8192-token safety ceiling was hit —
        # the ceiling exists only as a backstop against a run that never
        # terminates on its own.
        yield tracker.snapshot(
            sequence_length=position, current_k_lookahead=current_k, ended_naturally=hit_stop, is_final=True
        )
