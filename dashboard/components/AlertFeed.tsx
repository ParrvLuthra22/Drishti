import { Siren, Trash2 } from "lucide-react";
import { formatTime, riskColor } from "@/lib/risk";
import type { Alert } from "@/lib/types";
import { Panel } from "./Panel";

function AlertCard({ alert }: { alert: Alert }) {
  const color = riskColor(alert.risk_score);
  return (
    <article
      className="flex w-[340px] shrink-0 flex-col gap-2 rounded-md border border-line border-l-[3px] bg-bg/60 p-3"
      style={{ borderLeftColor: color }}
    >
      <div className="flex items-center justify-between font-mono text-[11px] tracking-widest uppercase">
        <span className="text-muted tabular-nums">{formatTime(alert.timestamp)}</span>
        <span className="text-muted">{alert.camera_id}</span>
        <span
          className="rounded px-1.5 py-0.5 font-semibold tabular-nums"
          style={{ color, backgroundColor: `${color}1F` }}
        >
          {alert.risk_score.toFixed(1)} · {alert.risk_level}
        </span>
      </div>
      <p className="line-clamp-2 text-sm leading-snug text-ink">{alert.summary || "No summary available."}</p>
      <p className="truncate text-[11px] text-muted" title={alert.signals.join(" · ")}>
        {alert.signals.join(" · ") || "No signals"}
      </p>
    </article>
  );
}

export function AlertFeed({ alerts, onClear }: { alerts: Alert[]; onClear: () => void }) {
  return (
    <Panel
      title={`Alert feed · ${alerts.length}`}
      icon={<Siren size={14} />}
      className="h-[210px] shrink-0"
      right={
        <button
          onClick={onClear}
          disabled={alerts.length === 0}
          className="flex items-center gap-1.5 rounded border border-line px-2 py-1 font-mono text-[11px] tracking-widest text-muted uppercase transition-colors enabled:hover:border-muted enabled:hover:text-ink disabled:opacity-40"
        >
          <Trash2 size={12} /> Clear
        </button>
      }
    >
      {alerts.length === 0 ? (
        <div className="flex flex-1 items-center justify-center font-mono text-xs tracking-widest text-muted uppercase">
          No alerts
        </div>
      ) : (
        <div className="no-scrollbar flex min-h-0 flex-1 gap-3 overflow-x-auto">
          {alerts.map((alert, i) => (
            <AlertCard key={`${alert.timestamp}-${i}`} alert={alert} />
          ))}
        </div>
      )}
    </Panel>
  );
}
