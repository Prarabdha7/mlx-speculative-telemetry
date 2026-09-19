"use client";

import { useEffect, useRef, useState } from "react";
import type { RunRequest } from "@/types/telemetry";

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

export default function ControlsSandbox({ isRunning, onStart, onStop }: ControlsSandboxProps) {
  const [prompt, setPrompt] = useState(PROMPT_PRESETS[0].prompt);
  const [kLookahead, setKLookahead] = useState(4);
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(128);

  return (
    <div className="border border-border rounded-md bg-surface p-4 flex flex-col gap-5">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Run Controls</div>

      <PromptEditor prompt={prompt} onChange={setPrompt} disabled={isRunning} />

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
        max={2}
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
            onStart({ prompt, k_lookahead: kLookahead, temperature, max_tokens: maxTokens })
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
