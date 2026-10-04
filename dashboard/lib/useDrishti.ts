"use client";

import { useCallback, useEffect, useState } from "react";
import type { Alert, HistoryPoint, Status } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

/** Polls the dashboard API: status + history every second, alerts every five. */
export function useDrishti() {
  const [status, setStatus] = useState<Status | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let active = true;

    const pollLive = async () => {
      try {
        const [nextStatus, nextHistory] = await Promise.all([
          getJson<Status>("/api/status"),
          getJson<{ history: HistoryPoint[] }>("/api/history"),
        ]);
        if (!active) return;
        setStatus(nextStatus);
        setHistory(nextHistory.history);
        setConnected(true);
      } catch {
        if (active) setConnected(false);
      }
    };

    const pollAlerts = async () => {
      try {
        const nextAlerts = await getJson<Alert[]>("/api/alerts");
        if (active) setAlerts(nextAlerts);
      } catch {
        // The connection indicator is driven by the status poll.
      }
    };

    pollLive();
    pollAlerts();
    const live = setInterval(pollLive, 1000);
    const alertTimer = setInterval(pollAlerts, 5000);
    return () => {
      active = false;
      clearInterval(live);
      clearInterval(alertTimer);
    };
  }, []);

  const clearAlerts = useCallback(async () => {
    try {
      await fetch(`${API_URL}/api/reset_alerts`, { method: "POST" });
      setAlerts([]);
    } catch {
      // Leave the feed as it is if the API is unreachable.
    }
  }, []);

  return { status, history, alerts, connected, clearAlerts };
}
