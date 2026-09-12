"use client";

import { useState } from "react";
import type { RunRequest } from "@/types/telemetry";

interface ControlsSandboxProps {
  prompts: string[];
  isRunning: boolean;
  onStart: (request: RunRequest) => void;
  onStop: () => void;
}

export default function ControlsSandbox({ prompts, isRunning, onStart, onStop }: ControlsSandboxProps) {
  const [prompt, setPrompt] = useState(prompts[0] ?? "");
  const [customPrompt, setCustomPrompt] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [kLookahead, setKLookahead] = useState(4);
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(128);

  const activePrompt = useCustom ? customPrompt : prompt;

  return (
    <div className="border border-border rounded-md bg-surface p-4 flex flex-col gap-4">
      <div className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Run Controls</div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs text-zinc-500 font-sans">Prompt</label>
        {!useCustom ? (
          <select
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isRunning}
            className="bg-black/40 border border-border rounded-sm px-2 py-1.5 text-sm font-mono text-zinc-200 disabled:opacity-50"
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
            className="bg-black/40 border border-border rounded-sm px-2 py-1.5 text-sm font-mono text-zinc-200 disabled:opacity-50 resize-none"
            placeholder="Enter a custom prompt…"
          />
        )}
        <button
          onClick={() => setUseCustom((v) => !v)}
          disabled={isRunning}
          className="self-start text-xs text-zinc-500 hover:text-zinc-300 font-sans disabled:opacity-50"
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
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between text-xs text-zinc-500 font-sans">
        <span>{label}</span>
        <span className="font-mono text-zinc-300">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-[#3b82f6] disabled:opacity-50"
      />
    </div>
  );
}
