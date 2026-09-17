"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ControlsSandbox from "@/components/ControlsSandbox";
import MetricsGauges from "@/components/MetricsGauges";
import TokenInspector from "@/components/TokenInspector";
import ExecutionDiff from "@/components/ExecutionDiff";
import BenchmarkTable from "@/components/BenchmarkTable";
import { TelemetryClient } from "@/lib/websocket";
import type { BenchmarkRun, ModelPair, RunMetrics, RunRequest, TokenTelemetry } from "@/types/telemetry";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [models, setModels] = useState<ModelPair | null>(null);
  const [prompts, setPrompts] = useState<string[]>([]);
  const [tokens, setTokens] = useState<TokenTelemetry[]>([]);
  const [metrics, setMetrics] = useState<RunMetrics | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRun[]>([]);
  const client = useRef<TelemetryClient | null>(null);

  const refreshBenchmarks = useCallback(() => {
    fetch(`${API_BASE}/api/benchmarks`)
      .then((r) => r.json())
      .then(setBenchmarks)
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then((r) => r.json())
      .then(setModels)
      .catch(() => setModels(null));
    fetch(`${API_BASE}/api/prompts`)
      .then((r) => r.json())
      .then(setPrompts)
      .catch(() => setPrompts([]));
    refreshBenchmarks();
  }, [refreshBenchmarks]);

  useEffect(() => {
    return () => client.current?.close();
  }, []);

  const handleStart = useCallback(async (request: RunRequest) => {
    setTokens([]);
    setMetrics(null);

    const res = await fetch(`${API_BASE}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!res.ok) return;
    const { run_id } = (await res.json()) as { run_id: string };
    setRunId(run_id);

    client.current?.close();
    client.current = new TelemetryClient({
      onToken: (t) => setTokens((prev) => [...prev, t]),
      onMetrics: (m) => {
        setMetrics(m);
        if (m.is_final) {
          setRunId(null);
          refreshBenchmarks();
        }
      },
      onClose: () => setRunId(null),
    });
    client.current.connect(run_id);
  }, [refreshBenchmarks]);

  const handleStop = useCallback(async () => {
    if (!runId) return;
    await fetch(`${API_BASE}/api/runs/${runId}/stop`, { method: "POST" });
    client.current?.close();
    setRunId(null);
  }, [runId]);

  return (
    <main className="min-h-screen p-4 md:p-6 flex flex-col gap-4 max-w-[1600px] mx-auto">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-sm font-mono text-zinc-200">mlx-speculative-telemetry</h1>
          {models && (
            <p className="text-xs text-zinc-500 font-mono mt-0.5">
              {models.draft_model} → {models.target_model}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-zinc-500">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: runId ? "#22c55e" : "#3f3f46" }}
          />
          {runId ? "streaming" : "idle"}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-4">
        <ControlsSandbox prompts={prompts} isRunning={runId !== null} onStart={handleStart} onStop={handleStop} />

        <div className="flex flex-col gap-4 min-w-0">
          <MetricsGauges metrics={metrics} />
          <TokenInspector tokens={tokens} />
          <div className="h-64">
            <ExecutionDiff tokens={tokens} />
          </div>
        </div>
      </div>

      <BenchmarkTable runs={benchmarks} />
    </main>
  );
}
