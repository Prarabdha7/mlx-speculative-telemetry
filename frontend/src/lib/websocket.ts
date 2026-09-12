import type { RunMetrics, TelemetryEvent, TokenTelemetry } from "@/types/telemetry";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_BASE_DELAY_MS = 500;

export interface TelemetryClientHandlers {
  onToken?: (event: TokenTelemetry) => void;
  onMetrics?: (event: RunMetrics) => void;
  onError?: (error: Event) => void;
  onClose?: (reason: string) => void;
}

function isTelemetryEvent(value: unknown): value is TelemetryEvent {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as { type?: unknown; data?: unknown };
  return (candidate.type === "token" || candidate.type === "metrics") && typeof candidate.data === "object";
}

export class TelemetryClient {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private intentionalClose = false;
  private runId: string | null = null;

  constructor(private readonly handlers: TelemetryClientHandlers) {}

  connect(runId: string): void {
    this.runId = runId;
    this.intentionalClose = false;
    this.reconnectAttempts = 0;
    this.open();
  }

  private open(): void {
    if (!this.runId) return;
    const socket = new WebSocket(`${WS_BASE}/ws/telemetry?run_id=${encodeURIComponent(this.runId)}`);

    socket.onmessage = (message: MessageEvent<string>) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(message.data);
      } catch {
        return;
      }
      if (!isTelemetryEvent(parsed)) return;
      if (parsed.type === "token") {
        this.handlers.onToken?.(parsed.data);
      } else {
        this.handlers.onMetrics?.(parsed.data);
      }
    };

    socket.onerror = (event) => {
      this.handlers.onError?.(event);
    };

    socket.onclose = (event) => {
      // Normal completion (1000) or an unknown/finished run_id (4004) — the run
      // is over either way, so don't retry against a queue that no longer exists.
      const terminal = this.intentionalClose || event.code === 1000 || event.code === 4004;
      if (terminal || this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        this.handlers.onClose?.(event.reason || `closed (code ${event.code})`);
        return;
      }
      const delay = RECONNECT_BASE_DELAY_MS * 2 ** this.reconnectAttempts;
      this.reconnectAttempts += 1;
      setTimeout(() => this.open(), delay);
    };

    this.socket = socket;
  }

  close(): void {
    this.intentionalClose = true;
    this.socket?.close();
    this.socket = null;
  }
}
