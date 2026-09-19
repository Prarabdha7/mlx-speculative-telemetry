"use client";

import { useEffect, useRef, useState } from "react";
import type { AutoTuneParams, RunRequest } from "@/types/telemetry";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const CLASSIFY_DEBOUNCE_MS = 350;

interface ControlsSandboxProps {
  isRunning: boolean;
  onStart: (request: RunRequest) => void;
  onStop: () => void;
}

const PROMPT_PRESETS: { label: string; prompt: string }[] = [
  {
    label: "Code",
    prompt: "Write a Python function that returns the nth Fibonacci number using memoization.",
  },
  {
    label: "Math",
    prompt: "Solve for x: 3x^2 - 12x + 9 = 0. Show each step of the derivation.",
  },
  {
    label: "Reasoning",
    prompt:
      "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain your reasoning step by step.",
  },
  {
    label: "Creative",
    prompt: "Write a short story about a lighthouse keeper who receives a letter carried in by the tide.",
  },
];

function PromptEditor({
  prompt,
  onChange,
  disabled,
}: {
  prompt: string;
  onChange: (v: string) => void;
  disabled: boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 320)}px`;
  }, [prompt]);

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs text-zinc-500 font-sans h-4 leading-4">Prompt</label>
      <div className="flex flex-wrap gap-1.5">
        {PROMPT_PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => onChange(preset.prompt)}
            disabled={disabled}
            className="text-xs font-sans text-zinc-600 dark:text-zinc-400 border border-border rounded-sm px-2 py-1 hover:text-zinc-900 dark:hover:text-zinc-100 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors disabled:opacity-50"
          >
            {preset.label}
          </button>
        ))}
      </div>
      <textarea
        ref={textareaRef}
        value={prompt}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        rows={3}
        className="bg-zinc-100 dark:bg-black/40 border border-border rounded-sm px-2 py-1.5 text-sm font-mono text-zinc-800 dark:text-zinc-200 disabled:opacity-50 resize-none min-h-[4.5rem] max-h-80 overflow-y-auto"
        placeholder="Enter a prompt, or pick a preset above…"
      />
    </div>
  );
}

function ToggleSwitch({
  enabled,
  onChange,
  disabled,
}: {
  enabled: boolean;
  onChange: (v: boolean) => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      disabled={disabled}
      onClick={() => onChange(!enabled)}
      className={cn(
        "relative w-9 h-5 rounded-full transition-colors shrink-0 disabled:opacity-50",
        enabled ? "bg-accepted" : "bg-zinc-300 dark:bg-zinc-700",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform",
          enabled && "translate-x-4",
        )}
      />
    </button>
  );
}

function AutonomousCard({ params, loading }: { params: AutoTuneParams | null; loading: boolean }) {
  return (
    <div className="border border-border rounded-md bg-zinc-100 dark:bg-black/40 p-3 flex flex-col gap-1.5">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">
        {loading ? "Analyzing…" : "Detected Intent"}
      </div>
      <div className="text-sm font-mono text-zinc-800 dark:text-zinc-200">{params?.detected_intent ?? "—"}</div>
      <div className="flex gap-4 text-xs font-mono text-zinc-600 dark:text-zinc-400">
        <span>Temp: {params ? params.temperature.toFixed(2) : "—"}</span>
        <span>K: {params?.lookahead_k ?? "—"}</span>
        <span>Max: {params?.max_tokens ?? "—"}</span>
      </div>
    </div>
  );
}

export default function ControlsSandbox({ isRunning, onStart, onStop }: ControlsSandboxProps) {
  const [prompt, setPrompt] = useState(PROMPT_PRESETS[0].prompt);
  const [manualOverride, setManualOverride] = useState(false);
  const [autoParams, setAutoParams] = useState<AutoTuneParams | null>(null);
  const [autoParamsLoading, setAutoParamsLoading] = useState(false);
  const [kLookahead, setKLookahead] = useState(4);
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(128);

  useEffect(() => {
    if (!prompt.trim()) {
      setAutoParams(null);
      return;
    }
    setAutoParamsLoading(true);
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/api/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      })
        .then((r) => r.json())
        .then((data: AutoTuneParams) => setAutoParams(data))
        .catch(() => setAutoParams(null))
        .finally(() => setAutoParamsLoading(false));
    }, CLASSIFY_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [prompt]);

  const effectiveK = manualOverride ? kLookahead : autoParams?.lookahead_k ?? 3;
  const effectiveTemperature = manualOverride ? temperature : autoParams?.temperature ?? 0.7;
  const effectiveMaxTokens = manualOverride ? maxTokens : autoParams?.max_tokens ?? 512;

  return (
    <div className="border border-border rounded-md bg-surface p-4 flex flex-col gap-5">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Run Controls</div>

      <PromptEditor prompt={prompt} onChange={setPrompt} disabled={isRunning} />

      <div className="flex flex-col gap-2">
        <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Autonomous Neural Controller</div>
        {!manualOverride && <AutonomousCard params={autoParams} loading={autoParamsLoading} />}
        <div className="flex items-center justify-between">
          <label className="text-xs text-zinc-500 font-sans">Advanced Manual Override</label>
          <ToggleSwitch enabled={manualOverride} onChange={setManualOverride} disabled={isRunning} />
        </div>
      </div>

      {manualOverride && (
        <>
          <Slider
            label="Lookahead (k)"
            value={kLookahead}
            min={1}
            max={16}
            step={1}
            disabled={isRunning}
            onChange={setKLookahead}
          />
          <Slider
            label="Temperature"
            value={temperature}
            min={0}
            max={1}
            step={0.1}
            decimals={1}
            disabled={isRunning}
            onChange={setTemperature}
          />
          <Slider
            label="Max Tokens"
            value={maxTokens}
            min={16}
            max={512}
            step={16}
            disabled={isRunning}
            onChange={setMaxTokens}
          />
        </>
      )}

      {isRunning ? (
        <button
          onClick={onStop}
          className="mt-2 bg-rejected/10 border border-rejected text-rejected rounded-sm py-2 text-sm font-sans hover:bg-rejected/20 transition-colors"
        >
          Stop
        </button>
      ) : (
        <button
          onClick={() =>
            onStart({
              prompt,
              k_lookahead: effectiveK,
              temperature: effectiveTemperature,
              max_tokens: effectiveMaxTokens,
              auto_tune: !manualOverride,
            })
          }
          disabled={!prompt.trim()}
          className="mt-2 bg-accepted/10 border border-accepted text-accepted rounded-sm py-2 text-sm font-sans hover:bg-accepted/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Start Run
        </button>
      )}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  decimals = 0,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  disabled: boolean;
  onChange: (v: number) => void;
}) {
  const [display, setDisplay] = useState(value);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => setDisplay(value), [value]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleInput = (v: number) => {
    setDisplay(v);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => onChange(v), 60);
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between h-4">
        <span className="text-xs text-zinc-500 font-sans leading-4">{label}</span>
        <span className="min-w-[3rem] text-center text-xs font-mono text-zinc-700 dark:text-zinc-300 bg-zinc-100 dark:bg-black/40 border border-border rounded-sm px-1.5 py-0.5 leading-4">
          {display.toFixed(decimals)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={display}
        disabled={disabled}
        onChange={(e) => handleInput(Number(e.target.value))}
        className="accent-[#3b82f6] disabled:opacity-50"
      />
    </div>
  );
}
