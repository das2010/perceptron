/**
 * Gráficos con Apache ECharts (SPEC §5.2): curvas en vivo, matrices y ROC.
 * Los colores salen de los tokens del tema (Oscuro/Lima/Violeta + neutros, §11.2).
 */
import { BarChart, HeatmapChart, LineChart, ParallelChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  ParallelComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

echarts.use([
  LineChart,
  BarChart,
  HeatmapChart,
  ScatterChart, // historia del HPO y Pareto (RF-HPO-06)
  ParallelChart, // coordenadas paralelas (RF-HPO-06)
  ParallelComponent,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export type EChartOption = echarts.EChartsCoreOption;

function token(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** Paleta de series validada para daltonismo sobre los tokens de marca. */
export function seriesColors(): string[] {
  return [
    token("--color-accent", "#334000"),
    token("--pt-violet-dark", "#a46dcd"),
    "#e0a100",
    "#3b82c4",
    token("--color-text-muted", "#5a6330"),
    "#c2410c",
  ];
}

export function EChart({
  option,
  height = 280,
  label,
}: {
  option: EChartOption;
  height?: number;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    chart.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    const text = token("--color-text", "#263000");
    const line = token("--color-border", "#d1d1c3");
    chart.current?.setOption(
      {
        color: seriesColors(),
        textStyle: { color: text, fontFamily: token("--pt-font", "sans-serif") },
        grid: { left: 48, right: 16, top: 32, bottom: 40 },
        tooltip: { trigger: "axis" },
        ...option,
        xAxis: option.xAxis && {
          axisLine: { lineStyle: { color: line } },
          ...(option.xAxis as object),
        },
        yAxis: option.yAxis && {
          splitLine: { lineStyle: { color: line } },
          ...(option.yAxis as object),
        },
      },
      true,
    );
  }, [option]);

  return <div ref={ref} role="img" aria-label={label} style={{ height, width: "100%" }} />;
}
