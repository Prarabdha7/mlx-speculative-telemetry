"use client";

import { useEffect, useRef, useState } from "react";
import type { RunMetrics, RunRequest } from "@/types/telemetry";

interface ControlsSandboxProps {
  isRunning: boolean;
  metrics: RunMetrics | null;
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

function eosStatus(isRunning: boolean, metrics: RunMetrics | null): { label: string; color: string } {
  if (isRunning) return { label: "Running", color: "#eab308" };
  if (!metrics) return { label: "Idle", color: "#71717a" };
  return metrics.ended_naturally
    ? { label: "Complete (EOS)", color: "#22c55e" }
    : { label: "Stopped (Safety Ceiling)", color: "#ef4444" };
}

function NeuralTelemetryCard({ isRunning, metrics }: { isRunning: boolean; metrics: RunMetrics | null }) {
  const status = eosStatus(isRunning, metrics);
  return (
    <div className="border border-border rounded-md bg-zinc-100 dark:bg-black/40 p-3 flex flex-col gap-2">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Neural Autonomous Telemetry</div>

      <div className="flex flex-col gap-0.5">
        <span className="text-xs text-zinc-500 dark:text-zinc-600 font-sans">System Persona</span>
        <span className="text-sm font-mono text-zinc-800 dark:text-zinc-200">
          {metrics?.system_role ?? "—"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="flex flex-col gap-0.5">
          <span className="text-zinc-500 dark:text-zinc-600 font-sans">Temperature</span>
          <span className="text-zinc-800 dark:text-zinc-200">
            {metrics ? metrics.actual_temperature.toFixed(2) : "—"}
          </span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-zinc-500 dark:text-zinc-600 font-sans">Top-P</span>
          <span className="text-zinc-800 dark:text-zinc-200">{metrics ? metrics.top_p.toFixed(2) : "—"}</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-zinc-500 dark:text-zinc-600 font-sans">Lookahead K</span>
          <span className="text-zinc-800 dark:text-zinc-200">{metrics?.current_k_lookahead ?? "—"}</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-zinc-500 dark:text-zinc-600 font-sans">EOS Status</span>
          <span style={{ color: status.color }}>{status.label}</span>
        </div>
      </div>
    </div>
  );
}

export default function ControlsSandbox({ isRunning, metrics, onStart, onStop }: ControlsSandboxProps) {
  const [prompt, setPrompt] = useState(PROMPT_PRESETS[0].prompt);

  return (
    <div className="border border-border rounded-md bg-surface p-4 flex flex-col gap-5">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Run Controls</div>

      <PromptEditor prompt={prompt} onChange={setPrompt} disabled={isRunning} />

      <NeuralTelemetryCard isRunning={isRunning} metrics={metrics} />

      {isRunning ? (
        <button
          onClick={onStop}
          className="mt-2 bg-rejected/10 border border-rejected text-rejected rounded-sm py-2 text-sm font-sans hover:bg-rejected/20 transition-colors"
        >
          Stop
        </button>
      ) : (
        <button
          onClick={() => onStart({ prompt })}
          disabled={!prompt.trim()}
          className="mt-2 bg-accepted/10 border border-accepted text-accepted rounded-sm py-2 text-sm font-sans hover:bg-accepted/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Start Run
        </button>
      )}
    </div>
  );
}
