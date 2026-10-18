/** Lo que devuelven los modelos del asistente y utilidades para mostrarlo. */
import { useEffect, useMemo } from "react";

export const JUDGED = new Set(["object_detection", "segmentation", "anomaly_detection"]);

export interface ModelPrediction {
  prediction?: unknown;
  confidence?: number;
  details?: {
    objects?: { label: string; score: number; box: [number, number, number, number] }[];
    fraction?: Record<string, number>;
    mask_png?: string;
    series?: string | null;
    times?: string[];
    scores?: number[];
    anomalies?: string[];
    threshold?: number;
  };
}

/** Un archivo elegido como URL para mostrarlo (se libera al cambiar). */
export function useObjectUrl(file: File | null): string | null {
  const url = useMemo(
    () => (file && typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : null),
    [file],
  );
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );
  return url;
}
