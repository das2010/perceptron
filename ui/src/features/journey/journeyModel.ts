/** Lo que devuelven los modelos del asistente y utilidades para mostrarlo. */
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
