import type { TokenTelemetry } from "@/types/telemetry";
import { STATUS_COLOR } from "@/lib/utils";

interface DiffSegment {
  text: string;
  diverged: boolean;
}

function draftOnlySegments(tokens: TokenTelemetry[]): DiffSegment[] {
  return tokens
    .filter((t) => t.status === "accepted" || t.status === "rejected")
    .map((t) => ({ text: t.token, diverged: t.status === "rejected" }));
}

function correctedSegments(tokens: TokenTelemetry[]): DiffSegment[] {
  return tokens
    .filter((t) => t.status !== "rejected")
    .map((t) => ({ text: t.token, diverged: t.status === "correction" || t.status === "bonus" }));
}

function Terminal({ title, segments }: { title: string; segments: DiffSegment[] }) {
  return (
    <div className="border border-border rounded-md bg-surface flex flex-col h-full min-w-0">
      <div className="px-3 py-2 border-b border-border text-xs text-zinc-500 font-sans uppercase tracking-wide">
        {title}
      </div>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-sm leading-relaxed whitespace-pre-wrap break-words">
        {segments.length === 0 && <span className="text-zinc-600">—</span>}
        {segments.map((s, i) => (
          <span key={i} style={s.diverged ? { color: STATUS_COLOR.correction } : undefined}>
            {s.text}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function ExecutionDiff({ tokens }: { tokens: TokenTelemetry[] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 h-full">
      <Terminal title="Draft-Only Output" segments={draftOnlySegments(tokens)} />
      <Terminal title="Speculative-Corrected Output" segments={correctedSegments(tokens)} />
    </div>
  );
}
