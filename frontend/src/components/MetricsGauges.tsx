import type { RunMetrics } from "@/types/telemetry";
import { formatPercent, formatTps } from "@/lib/utils";

function Gauge({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="border border-border rounded-md bg-surface px-4 py-3 flex flex-col gap-1">
      <span className="text-xs text-zinc-500 font-sans uppercase tracking-wide">{label}</span>
      <span className="text-2xl font-mono" style={accent ? { color: accent } : undefined}>
        {value}
      </span>
    </div>
  );
}

export default function MetricsGauges({ metrics }: { metrics: RunMetrics | null }) {
  const acceptanceColor = metrics
    ? metrics.acceptance_rate >= 0.7
      ? "#22c55e"
      : metrics.acceptance_rate >= 0.4
        ? "#eab308"
        : "#ef4444"
    : undefined;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      <Gauge label="Speedup" value={metrics ? `${metrics.speedup_ratio.toFixed(2)}x` : "—"} accent="#3b82f6" />
      <Gauge
        label="Acceptance Rate"
        value={metrics ? formatPercent(metrics.acceptance_rate) : "—"}
        accent={acceptanceColor}
      />
      <Gauge label="Effective TPS" value={metrics ? formatTps(metrics.effective_tokens_per_second) : "—"} />
      <Gauge label="Draft TPS" value={metrics ? formatTps(metrics.draft_tokens_per_second) : "—"} />
      <Gauge label="Target TPS" value={metrics ? formatTps(metrics.target_tokens_per_second) : "—"} />
      <Gauge
        label="Memory Bandwidth"
        value={metrics ? `${metrics.memory_bandwidth_gbps.toFixed(1)} GB/s` : "—"}
      />
    </div>
  );
}
