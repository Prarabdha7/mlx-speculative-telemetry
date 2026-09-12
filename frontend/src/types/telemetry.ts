export type TokenStatus = "accepted" | "rejected" | "correction" | "bonus";

export interface TokenTelemetry {
  token: string;
  token_id: number;
  status: TokenStatus;
  position: number;
  draft_probability: number | null;
  target_probability: number | null;
  accept_probability: number | null;
  latency_ms: number;
  timestamp: number;
}

export interface RunMetrics {
  run_id: string;
  is_final: boolean;
  elapsed_s: number;
  total_tokens: number;
  draft_tokens_per_second: number;
  target_tokens_per_second: number;
  effective_tokens_per_second: number;
  speedup_ratio: number;
  acceptance_rate: number;
  memory_bandwidth_gbps: number;
}

export interface RunRequest {
  prompt: string;
  k_lookahead: number;
  temperature: number;
  max_tokens: number;
}

export interface ModelPair {
  draft_model: string;
  target_model: string;
}

export type TelemetryEvent =
  | { type: "token"; data: TokenTelemetry }
  | { type: "metrics"; data: RunMetrics };

export interface BenchmarkRun {
  id: string;
  timestamp: number;
  draft_model: string;
  target_model: string;
  k_lookahead: number;
  temperature: number;
  prompt: string;
  total_tokens: number;
  acceptance_rate: number;
  speedup_ratio: number;
  effective_tokens_per_second: number;
}
