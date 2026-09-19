"use client";

import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";
import type { BenchmarkRun } from "@/types/telemetry";
import { downloadCsv, formatDuration, formatPercent, formatTimestamp, formatTps, toCsv } from "@/lib/utils";

function ModelCell({ value }: { value: string }) {
  return (
    <span className="block max-w-[12rem] truncate" title={value}>
      {value}
    </span>
  );
}

const columns: ColumnDef<BenchmarkRun>[] = [
  { header: "Timestamp", accessorKey: "timestamp", cell: (c) => formatTimestamp(c.getValue<number>()) },
  { header: "Draft Model", accessorKey: "draft_model", cell: (c) => <ModelCell value={c.getValue<string>()} /> },
  { header: "Target Model", accessorKey: "target_model", cell: (c) => <ModelCell value={c.getValue<string>()} /> },
  { header: "k", accessorKey: "k_lookahead" },
  { header: "Temp", accessorKey: "temperature" },
  { header: "Tokens", accessorKey: "total_tokens" },
  { header: "Acceptance", accessorKey: "acceptance_rate", cell: (c) => formatPercent(c.getValue<number>()) },
  { header: "Speedup", accessorKey: "speedup_ratio", cell: (c) => `${c.getValue<number>().toFixed(2)}x` },
  {
    header: "Effective TPS",
    accessorKey: "effective_tokens_per_second",
    cell: (c) => formatTps(c.getValue<number>()),
  },
  { header: "Wall Clock", accessorKey: "elapsed_s", cell: (c) => formatDuration(c.getValue<number>()) },
];

export default function BenchmarkTable({ runs }: { runs: BenchmarkRun[] }) {
  const table = useReactTable({ data: runs, columns, getCoreRowModel: getCoreRowModel() });

  return (
    <div className="border border-border rounded-md bg-surface flex flex-col">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between">
        <span className="text-xs text-zinc-500 font-sans uppercase tracking-wide">Benchmark History</span>
        <button
          onClick={() => downloadCsv(`benchmarks-${Date.now()}.csv`, toCsv(runs))}
          disabled={runs.length === 0}
          className="text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 font-sans disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Export CSV
        </button>
      </div>
      <div className="overflow-auto max-h-[420px]">
        <table className="w-full text-sm font-mono">
          <thead className="sticky top-0 z-10 bg-surface">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-border">
                {hg.headers.map((header) => (
                  <th key={header.id} className="text-left px-3 py-2 text-zinc-500 font-sans font-normal text-xs whitespace-nowrap bg-surface">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-border last:border-0 hover:bg-black/[0.03] dark:hover:bg-white/5">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3 py-2 text-zinc-700 dark:text-zinc-300 whitespace-nowrap">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
            {runs.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-3 py-4 text-center text-zinc-600 font-sans">
                  No completed runs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
