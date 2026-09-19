"use client";

import { useEffect, useRef, useState } from "react";
import type { RunRequest } from "@/types/telemetry";

interface ControlsSandboxProps {
  prompts: string[];
  isRunning: boolean;
  onStart: (request: RunRequest) => void;
  onStop: () => void;
}

export default function ControlsSandbox({ prompts, isRunning, onStart, onStop }: ControlsSandboxProps) {
  const [prompt, setPrompt] = useState(prompts[0] ?? "");

  useEffect(() => {
    if (!prompt && prompts.length > 0) setPrompt(prompts[0]);
  }, [prompts, prompt]);
  const [customPrompt, setCustomPrompt] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [kLookahead, setKLookahead] = useState(4);
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(128);

  const activePrompt = useCustom ? customPrompt : prompt;

  return (
    <div className="border border-border rounded-md bg-surface p-4 flex flex-col gap-5">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Run Controls</div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs text-zinc-500 font-sans h-4 leading-4">Prompt</label>
        {!useCustom ? (
          <select
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isRunning}
            className="bg-zinc-100 dark:bg-black/40 border border-border rounded-sm px-2 py-1.5 text-sm font-mono text-zinc-800 dark:text-zinc-200 disabled:opacity-50"
          >
            {prompts.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        ) : (
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            disabled={isRunning}
            rows={3}
            className="bg-zinc-100 dark:bg-black/40 border border-border rounded-sm px-2 py-1.5 text-sm font-mono text-zinc-800 dark:text-zinc-200 disabled:opacity-50 resize-none"
            placeholder="Enter a custom prompt…"
          />
        )}
        <button
          onClick={() => setUseCustom((v) => !v)}
          disabled={isRunning}
          className="self-start text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-300 font-sans disabled:opacity-50"
        >
          {useCustom ? "use preset prompts" : "use custom prompt"}
        </button>
      </div>

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
            onStart({ prompt: activePrompt, k_lookahead: kLookahead, temperature, max_tokens: maxTokens })
          }
          disabled={!activePrompt.trim()}
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
