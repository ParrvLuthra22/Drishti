export type RiskLevel = "low" | "moderate" | "high" | "critical";

export interface Status {
  camera_id: string;
  frame_id: number;
  timestamp: number;
  risk_score: number;
  risk_level: RiskLevel;
  alert_required: boolean;
  incident_summary: string;
  recommended_action: string;
  signals: string[];
  track_count: number;
  density_level: string;
  density_score: number;
  flow_direction: string;
  flow_score: number;
  bottleneck_detected: boolean;
  fight_probability: number;
  anomaly_level: string;
  anomaly_score: number;
  fps: number;
  /** Seconds since the pipeline last pushed an update; null if it never has. */
  age_s: number | null;
}

export interface HistoryPoint {
  t: number;
  score: number;
}

export interface Alert {
  timestamp: number;
  camera_id: string;
  risk_score: number;
  risk_level: RiskLevel;
  summary: string;
  signals: string[];
}
