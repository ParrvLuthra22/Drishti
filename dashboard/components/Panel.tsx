import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  icon?: ReactNode;
  right?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Panel({ title, icon, right, className = "", children }: PanelProps) {
  return (
    <section className={`flex min-h-0 flex-col rounded-lg border border-line bg-panel ${className}`}>
      <header className="flex shrink-0 items-center justify-between border-b border-line px-4 py-2.5">
        <h2 className="flex items-center gap-2 font-mono text-[11px] font-medium tracking-[0.18em] text-muted uppercase">
          {icon}
          {title}
        </h2>
        {right}
      </header>
      <div className="no-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto p-4">{children}</div>
    </section>
  );
}

export function Bar({ value, color }: { value: number; color: string }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  );
}

export function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center rounded border px-2 py-0.5 font-mono text-[11px] font-semibold tracking-widest uppercase"
      style={{ color, borderColor: `${color}66`, backgroundColor: `${color}1A` }}
    >
      {label}
    </span>
  );
}
