"use client";

import { useState } from "react";
import type { TokenTelemetry } from "@/types/telemetry";
import { cn, STATUS_COLOR } from "@/lib/utils";

const STATUS_LABEL: Record<TokenTelemetry["status"], string> = {
  accepted: "Accepted",
  rejected: "Rejected",
  correction: "Correction",
  bonus: "Bonus",
};

const FINAL_DECISION: Record<TokenTelemetry["status"], string> = {
  accepted: "Accepted",
  rejected: "Rejected",
  correction: "Resampled",
  bonus: "Accepted (bonus)",
};

function TokenChip({ token, onSelect }: { token: TokenTelemetry; onSelect: (t: TokenTelemetry) => void }) {
  const color = STATUS_COLOR[token.status];
  return (
    <button
      onClick={() => onSelect(token)}
      className={cn(
        "font-mono text-sm px-1 py-0.5 rounded-sm border border-transparent hover:border-zinc-600 transition-colors",
        token.status === "rejected" && "line-through decoration-2",
      )}
      style={{ color }}
      title={`${STATUS_LABEL[token.status]} @ position ${token.position}`}
    >
      {token.token || "·"}
    </button>
  );
}

function TokenModal({ token, onClose }: { token: TokenTelemetry; onClose: () => void }) {
  const reasoning =
    token.status === "accepted"
      ? "Draft token matched the target model's distribution and was committed directly."
      : token.status === "rejected"
        ? "Draft token diverged from the target model; discarded in favor of a resampled correction."
        : token.status === "correction"
          ? "Resampled from the target model's residual distribution after the preceding draft token was rejected."
          : "Free token emitted by the target model after the full draft window was accepted.";

  const isGreedy = token.status !== "correction" && token.status !== "bonus" && token.accept_probability === null;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-surface border border-border rounded-md w-full max-w-md p-4 font-mono text-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <span className="text-base" style={{ color: STATUS_COLOR[token.status] }}>
            &quot;{token.token}&quot;
          </span>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 font-sans">
            close
          </button>
        </div>

        <dl className="space-y-1.5 text-zinc-600 dark:text-zinc-400">
          <Row label="position" value={String(token.position)} />
          <Row label="token_id" value={String(token.token_id)} />
          <Row label="latency_ms" value={token.latency_ms.toFixed(2)} />
        </dl>

        <div className="mt-3 pt-3 border-t border-border space-y-1.5">
          <Row label="Draft Probability Q(x)" value={formatProb(token.draft_probability)} />
          <Row label="Target Probability P(x)" value={formatProb(token.target_probability)} />
          <div className="flex flex-col gap-1 pt-1">
            <span className="text-zinc-500 dark:text-zinc-600 text-xs">
              Acceptance Condition: min(1, P(x)/Q(x))
            </span>
            <span className="text-zinc-800 dark:text-zinc-200 text-right">
              {isGreedy ? "argmax match (greedy)" : formatProb(token.accept_probability)}
            </span>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-border flex items-center justify-between">
          <span className="text-xs text-zinc-500 dark:text-zinc-600 font-sans uppercase tracking-wide">
            Final Decision
          </span>
          <span className="font-mono text-sm" style={{ color: STATUS_COLOR[token.status] }}>
            {FINAL_DECISION[token.status]}
          </span>
        </div>

        <p className="mt-3 text-zinc-500 font-sans text-xs leading-relaxed">{reasoning}</p>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-zinc-500 dark:text-zinc-600">{label}</dt>
      <dd className="text-zinc-800 dark:text-zinc-200">{value}</dd>
    </div>
  );
}

function formatProb(value: number | null): string {
  return value === null ? "—" : value.toFixed(4);
}

export default function TokenInspector({ tokens }: { tokens: TokenTelemetry[] }) {
  const [selected, setSelected] = useState<TokenTelemetry | null>(null);

  return (
    <div className="border border-border rounded-md bg-surface flex flex-col h-full">
      <div className="px-3 py-2 border-b border-border text-xs text-zinc-500 font-sans uppercase tracking-wide">
        Token Stream
      </div>
      <div className="flex-1 overflow-y-auto p-3 flex flex-wrap content-start gap-x-0.5 gap-y-1 min-h-[200px]">
        {tokens.length === 0 && <span className="text-zinc-600 font-mono text-sm">Waiting for a run to start…</span>}
        {tokens.map((t, i) => (
          <TokenChip key={i} token={t} onSelect={setSelected} />
        ))}
      </div>
      {selected && <TokenModal token={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
