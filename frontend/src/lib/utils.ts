import type { TokenStatus } from "@/types/telemetry";

export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export const STATUS_COLOR: Record<TokenStatus, string> = {
  accepted: "#22c55e",
  rejected: "#ef4444",
  correction: "#3b82f6",
  bonus: "#3b82f6",
};

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatTps(value: number): string {
  return `${value.toFixed(1)} tok/s`;
}

export function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

export function formatDuration(seconds: number): string {
  return `${seconds.toFixed(2)}s`;
}

export function toCsv<T extends object>(rows: T[]): string {
  if (rows.length === 0) return "";
  const headers = Object.keys(rows[0]) as (keyof T)[];
  const escape = (value: unknown) => {
    const s = String(value ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => escape(row[h])).join(","));
  }
  return lines.join("\n");
}

export function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export interface PromptIntent {
  label: string;
  temperature: number;
  k: number;
}

// Mirrors backend/app/engine/mlx_speculative.py's classify_prompt_intent —
// this is only a live preview shown before Start Run; the server classifies
// authoritatively and is the source of truth for what a run actually used.
const CODE_MATH_PATTERNS = [
  /def\s/,
  /```/,
  ...["function", "import", "class", "calculate", "solve", "code", "sql"].map((kw) => new RegExp(`\\b${kw}\\b`)),
];
const FACTUAL_PATTERNS = ["what is", "who is", "explain", "summarize", "history", "definition"].map(
  (kw) => new RegExp(`\\b${kw}\\b`),
);

export function classifyPromptIntent(prompt: string): PromptIntent {
  const lowered = prompt.toLowerCase();
  if (CODE_MATH_PATTERNS.some((re) => re.test(lowered))) {
    return { label: "Code Mode", temperature: 0.0, k: 5 };
  }
  if (FACTUAL_PATTERNS.some((re) => re.test(lowered))) {
    return { label: "Factual Mode", temperature: 0.2, k: 4 };
  }
  return { label: "Creative Mode", temperature: 0.7, k: 3 };
}
