"use client";

import { AlertFeed } from "@/components/AlertFeed";
import { HistoryChart } from "@/components/HistoryChart";
import { MetricsPanel } from "@/components/MetricsPanel";
import { RiskPanel } from "@/components/RiskPanel";
import { TopBar } from "@/components/TopBar";
import { useDrishti } from "@/lib/useDrishti";

export default function Home() {
  const { status, history, alerts, connected, clearAlerts } = useDrishti();

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-bg pt-14">
      <TopBar connected={connected} />
      <main className="flex min-h-0 flex-1 flex-col gap-4 p-4">
        <div className="grid min-h-0 flex-1 grid-cols-[1.1fr_1fr_1.4fr] gap-4">
          <RiskPanel status={status} connected={connected} />
          <MetricsPanel status={status} />
          <HistoryChart history={history} />
        </div>
        <AlertFeed alerts={alerts} onClear={clearAlerts} />
      </main>
    </div>
  );
}
