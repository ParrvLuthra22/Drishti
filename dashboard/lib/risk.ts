export const COLORS = {
  green: "#22C55E",
  yellow: "#EAB308",
  orange: "#F97316",
  red: "#EF4444",
} as const;

/** Risk score bands: green < 25, yellow 25-50, orange 50-75, red 75-100. */
export const RISK_BANDS = [
  { from: 0, to: 25, color: COLORS.green },
  { from: 25, to: 50, color: COLORS.yellow },
  { from: 50, to: 75, color: COLORS.orange },
  { from: 75, to: 100, color: COLORS.red },
] as const;

export function riskColor(score: number): string {
  if (score < 25) return COLORS.green;
  if (score < 50) return COLORS.yellow;
  if (score < 75) return COLORS.orange;
  return COLORS.red;
}

const SEVERITY: Record<string, string> = {
  // risk level and density level
  low: COLORS.green,
  moderate: COLORS.yellow,
  high: COLORS.orange,
  critical: COLORS.red,
  // flow direction
  calm: COLORS.green,
  directional: COLORS.yellow,
  chaotic: COLORS.red,
  // anomaly level
  normal: COLORS.green,
  suspicious: COLORS.yellow,
  anomaly: COLORS.red,
};

export function severityColor(label: string): string {
  return SEVERITY[label] ?? COLORS.green;
}

export function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], { hour12: false });
}
