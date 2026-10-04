"use client";

import { LineChart as LineChartIcon } from "lucide-react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, usePlotArea, XAxis, YAxis } from "recharts";
import { COLORS, formatTime, RISK_BANDS, riskColor } from "@/lib/risk";
import type { HistoryPoint } from "@/lib/types";
import { Panel } from "./Panel";

const GRADIENT_ID = "risk-line";

/**
 * Colours the line by absolute risk band (green/yellow/orange/red) rather than by the
 * data's own range, by pinning the gradient to the plot area's pixel coordinates.
 */
function RiskGradient() {
  const area = usePlotArea();
  if (!area) return null;
  const top = area.y;
  const bottom = area.y + area.height;

  // Hard stops at the band edges. Offsets run from the top of the plot (score 100) to the bottom (0).
  const stops = [...RISK_BANDS].reverse().flatMap((band) => {
    const start = 1 - band.to / 100;
    const end = 1 - band.from / 100;
    return [
      <stop key={`${band.from}-a`} offset={start} stopColor={band.color} />,
      <stop key={`${band.from}-b`} offset={end} stopColor={band.color} />,
    ];
  });

  return (
    <defs>
      <linearGradient id={GRADIENT_ID} gradientUnits="userSpaceOnUse" x1={0} y1={top} x2={0} y2={bottom}>
        {stops}
      </linearGradient>
    </defs>
  );
}

export function HistoryChart({ history }: { history: HistoryPoint[] }) {
  const latest = history.length ? history[history.length - 1].score : 0;

  return (
    <Panel
      title="Risk history"
      icon={<LineChartIcon size={14} />}
      right={<span className="font-mono text-[11px] tracking-widest text-muted uppercase">Last {history.length || 0} readings</span>}
    >
      <div className="min-h-0 flex-1">
        {history.length === 0 ? (
          <div className="flex h-full items-center justify-center font-mono text-xs tracking-widest text-muted uppercase">
            No readings yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 8, right: 30, bottom: 0, left: -12 }}>
              <RiskGradient />
              <CartesianGrid stroke="#1c2a45" strokeDasharray="2 4" vertical={false} />
              <XAxis
                dataKey="t"
                tickFormatter={formatTime}
                stroke="#7f8ea8"
                tick={{ fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
                tickLine={false}
                axisLine={{ stroke: "#1c2a45" }}
                minTickGap={48}
              />
              <YAxis
                domain={[0, 100]}
                ticks={[0, 25, 50, 75, 100]}
                stroke="#7f8ea8"
                tick={{ fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
                tickLine={false}
                axisLine={false}
              />
              <ReferenceLine y={25} stroke={COLORS.yellow} strokeDasharray="4 4" strokeOpacity={0.45} />
              <ReferenceLine y={50} stroke={COLORS.orange} strokeDasharray="4 4" strokeOpacity={0.45} />
              <ReferenceLine y={75} stroke={COLORS.red} strokeDasharray="4 4" strokeOpacity={0.45} />
              <Tooltip
                contentStyle={{ background: "#0b1220", border: "1px solid #1c2a45", borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: "#7f8ea8" }}
                labelFormatter={(t) => formatTime(Number(t))}
                formatter={(value) => [Number(value).toFixed(1), "Risk"]}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke={`url(#${GRADIENT_ID})`}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, fill: riskColor(latest) }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </Panel>
  );
}
