import { Activity, Gauge, Layers, ScanEye, Swords, Users, Wind } from "lucide-react";
import type { ReactNode } from "react";
import { COLORS, severityColor } from "@/lib/risk";
import type { Status } from "@/lib/types";
import { Bar, Badge, Panel } from "./Panel";

function StatCard({ label, icon, children }: { label: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div className="flex flex-col justify-between gap-3 rounded-md border border-line bg-bg/60 p-4">
      <div className="flex items-center gap-2 font-mono text-[11px] tracking-[0.18em] text-muted uppercase">
        {icon}
        {label}
      </div>
      {children}
    </div>
  );
}

export function MetricsPanel({ status }: { status: Status | null }) {
  const densityColor = severityColor(status?.density_level ?? "low");
  const flowColor = severityColor(status?.flow_direction ?? "calm");
  const fight = status?.fight_probability ?? 0;

  return (
    <Panel
      title="Live metrics"
      icon={<Activity size={14} />}
      right={status?.bottleneck_detected ? <Badge label="Bottleneck" color={COLORS.orange} /> : undefined}
    >
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Tracks" icon={<Users size={13} />}>
          <span className="font-mono text-5xl leading-none font-semibold tabular-nums">{status?.track_count ?? 0}</span>
        </StatCard>

        <StatCard label="FPS" icon={<Gauge size={13} />}>
          <span className="font-mono text-5xl leading-none font-semibold tabular-nums">
            {(status?.fps ?? 0).toFixed(1)}
          </span>
        </StatCard>

        <StatCard label="Density" icon={<Layers size={13} />}>
          <div className="space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-xl font-semibold uppercase" style={{ color: densityColor }}>
                {status?.density_level ?? "low"}
              </span>
              <span className="font-mono text-xs text-muted tabular-nums">{(status?.density_score ?? 0).toFixed(2)}</span>
            </div>
            <Bar value={status?.density_score ?? 0} color={densityColor} />
          </div>
        </StatCard>

        <StatCard label="Flow" icon={<Wind size={13} />}>
          <div className="space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-xl font-semibold uppercase" style={{ color: flowColor }}>
                {status?.flow_direction ?? "calm"}
              </span>
              <span className="font-mono text-xs text-muted tabular-nums">{(status?.flow_score ?? 0).toFixed(2)}</span>
            </div>
            <Bar value={status?.flow_score ?? 0} color={flowColor} />
          </div>
        </StatCard>
      </div>

      <div className="mt-4 rounded-md border border-line bg-bg/60 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="flex items-center gap-2 font-mono text-[11px] tracking-[0.18em] text-muted uppercase">
            <Swords size={13} /> Fight probability
          </span>
          <span className="font-mono text-2xl font-semibold tabular-nums" style={{ color: fight > 0.5 ? COLORS.red : undefined }}>
            {Math.round(fight * 100)}%
          </span>
        </div>
        <Bar value={fight} color={COLORS.red} />
      </div>

      <div className="mt-3 flex items-center justify-between rounded-md border border-line bg-bg/60 p-4">
        <span className="flex items-center gap-2 font-mono text-[11px] tracking-[0.18em] text-muted uppercase">
          <ScanEye size={13} /> Anomaly
        </span>
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-muted tabular-nums">{(status?.anomaly_score ?? 0).toFixed(4)}</span>
          <Badge label={status?.anomaly_level ?? "normal"} color={severityColor(status?.anomaly_level ?? "normal")} />
        </div>
      </div>
    </Panel>
  );
}
