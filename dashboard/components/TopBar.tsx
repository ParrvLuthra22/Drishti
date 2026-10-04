"use client";

import { useEffect, useState } from "react";

export function TopBar({ connected }: { connected: boolean }) {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    const tick = () => setNow(new Date());
    const first = setTimeout(tick, 0);
    const timer = setInterval(tick, 1000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, []);

  return (
    <header className="fixed inset-x-0 top-0 z-10 grid h-14 grid-cols-3 items-center border-b border-line bg-bg/95 px-5 backdrop-blur">
      <div className="flex items-center gap-3">
        <span className="h-2.5 w-2.5 rounded-[2px] bg-signal shadow-[0_0_10px_var(--color-signal)]" />
        <span className="font-mono text-lg font-bold tracking-[0.3em] text-signal">DRISHTI</span>
      </div>

      <div className="text-center font-mono text-xs tracking-[0.45em] text-muted uppercase">
        Operation Frontier
      </div>

      <div className="flex items-center justify-end gap-5">
        <time className="font-mono text-sm tabular-nums text-ink" suppressHydrationWarning>
          {now ? now.toLocaleTimeString([], { hour12: false }) : "--:--:--"}
        </time>
        <div className="flex items-center gap-2" title={connected ? "API reachable" : "API unreachable"}>
          <span
            className={`h-2.5 w-2.5 rounded-full ${connected ? "bg-green-500 shadow-[0_0_8px_#22C55E]" : "bg-red-500 shadow-[0_0_8px_#EF4444]"}`}
          />
          <span className="font-mono text-[11px] tracking-widest text-muted uppercase">
            {connected ? "Online" : "Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
