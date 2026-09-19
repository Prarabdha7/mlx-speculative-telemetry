"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  return (
    <button
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      aria-label="Toggle theme"
      className="w-7 h-7 flex items-center justify-center rounded-sm border border-border text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors"
    >
      {mounted && resolvedTheme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
    </button>
  );
}
