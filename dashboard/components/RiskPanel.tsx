import { ArrowRight, ShieldAlert, Siren } from "lucide-react";
import { RISK_BANDS, riskColor, severityColor } from "@/lib/risk";
import type { Status } from "@/lib/types";
import { Badge, Panel } from "./Panel";

function pipelineState(status: Status | null, connected: boolean): { label: string; live: boolean } {
  if (!connected) return { label: "API offline", live: false };
  if (!status || status.age_s === null) return { label: "Awaiting pipeline", live: false };
  if (status.age_s > 5) return { label: `Stale ${Math.round(status.age_s)}s`, live: false };
  return { label: "Live", live: true };
}

export function RiskPanel({ status, connected }: { status: Status | null; connected: boolean }) {
  const score = status?.risk_score ?? 0;
  const color = riskColor(score);
  const pipeline = pipelineState(status, connected);

  return (
    <Panel
      title="Risk assessment"
      icon={<ShieldAlert size={14} />}
      right={
        <span className="flex items-center gap-2 font-mono text-[11px] tracking-widest uppercase">
          <span className={`h-1.5 w-1.5 rounded-full ${pipeline.live ? "animate-pulse bg-green-500" : "bg-muted"}`} />
          <span className={pipeline.live ? "text-green-500" : "text-muted"}>{pipeline.label}</span>
        </span>
      }
    >
      <div className="flex items-end gap-3">
        <span
          className="font-mono text-[clamp(4.5rem,15vh,8.5rem)] leading-[0.85] font-bold tabular-nums transition-colors duration-500"
          style={{ color, textShadow: `0 0 40px ${color}55` }}
        >
          {Math.round(score)}
        </span>
        <span className="pb-2 font-mono text-2xl text-muted">/100</span>
      </div>

      <div className="mt-4 flex items-center gap-3">
        <Badge label={status?.risk_level ?? "low"} color={color} />
        {status?.alert_required && (
          <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold tracking-widest text-red-400 uppercase">
            <Siren size={14} className="animate-pulse" /> Alert
          </span>
        )}
        {status?.camera_id && (
          <span className="ml-auto font-mono text-[11px] tracking-widest text-muted uppercase">
            {status.camera_id} · frame {status.frame_id}
          </span>
        )}
      </div>

      <div className="relative mt-3 flex h-1.5 gap-0.5">
        {RISK_BANDS.map((band) => (
          <div key={band.from} className="flex-1 rounded-full opacity-35" style={{ backgroundColor: band.color }} />
        ))}
        <div
          className="absolute -top-1 h-3.5 w-0.5 -translate-x-1/2 rounded bg-white transition-[left] duration-500"
          style={{ left: `${Math.min(score, 100)}%` }}
        />
      </div>

      <div className="no-scrollbar mt-4 flex min-h-0 flex-1 flex-col gap-3.5 overflow-y-auto">
        <div>
          <h3 className="mb-1.5 font-mono text-[11px] tracking-[0.18em] text-muted uppercase">Signals</h3>
          {status && status.signals.length > 0 ? (
            <ul className="space-y-1">
              {status.signals.map((signal) => (
                <li key={signal} className="flex items-start gap-2 text-sm text-ink">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: color }} />
                  {signal}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted">No active signals.</p>
          )}
        </div>

        <div>
          <h3 className="mb-1.5 font-mono text-[11px] tracking-[0.18em] text-muted uppercase">Incident summary</h3>
          <p className="text-sm leading-relaxed text-ink">
            {status?.incident_summary || <span className="text-muted">No incident.</span>}
          </p>
        </div>

      </div>

      <div className="mt-3 shrink-0 border-t border-line pt-3">
          <h3 className="mb-1.5 font-mono text-[11px] tracking-[0.18em] text-muted uppercase">Recommended action</h3>
          <p className="flex items-start gap-2 text-sm leading-relaxed text-ink">
            <ArrowRight size={15} className="mt-0.5 shrink-0" style={{ color: severityColor(status?.risk_level ?? "low") }} />
            {status?.recommended_action || <span className="text-muted">Continue monitoring.</span>}
          </p>
      </div>
    </Panel>
  );
}
