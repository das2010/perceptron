/**
 * Lo que devuelve cada tipo de modelo, mostrado para alguien sin conocimientos de ML: recuadros
 * sobre la foto, la zona marcada, el texto leído, el pronóstico o los momentos raros.
 */
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { formatNumber } from "@/lib/format";

import type { ModelPrediction } from "./journeyModel";

const COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#eab308", "#a855f7", "#14b8a6"];

/** La máscara (índice de clase por pixel) pintada con colores semitransparentes. */
function MaskLayer({ png }: { png: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext?.("2d");
    if (!canvas || !ctx) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height);
      for (let i = 0; i < data.data.length; i += 4) {
        const cls = data.data[i] ?? 0;
        const color = cls ? (COLORS[(cls - 1) % COLORS.length] ?? "#ef4444") : null;
        if (!color) {
          data.data[i + 3] = 0;
          continue;
        }
        data.data[i] = parseInt(color.slice(1, 3), 16);
        data.data[i + 1] = parseInt(color.slice(3, 5), 16);
        data.data[i + 2] = parseInt(color.slice(5, 7), 16);
        data.data[i + 3] = 110;
      }
      ctx.putImageData(data, 0, 0);
    };
    img.src = `data:image/png;base64,${png}`;
  }, [png]);
  return <canvas ref={ref} aria-hidden="true" className="absolute inset-0 h-full w-full" />;
}

/** La foto con lo que encontró el modelo encima. */
function ImageResult({ src, p }: { src: string; p: ModelPrediction }) {
  return (
    <div className="relative inline-block max-w-full overflow-hidden rounded-pt border border-line">
      <img src={src} alt="" className="block max-h-80 max-w-full" />
      {p.details?.mask_png && <MaskLayer png={p.details.mask_png} />}
      {(p.details?.objects ?? []).map((o, i) => {
        const [x0, y0, x1, y1] = o.box;
        return (
          <div
            key={i}
            className="absolute border-2"
            style={{
              left: `${x0 * 100}%`,
              top: `${y0 * 100}%`,
              width: `${(x1 - x0) * 100}%`,
              height: `${(y1 - y0) * 100}%`,
              borderColor: COLORS[i % COLORS.length],
            }}
          >
            <span
              className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 text-xs text-white"
              style={{ background: COLORS[i % COLORS.length] }}
            >
              {o.label} {Math.round(o.score * 100)} %
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Una predicción, según la tarea del modelo. */
export function PredictionView({
  task,
  p,
  image,
}: {
  task: string;
  p: ModelPrediction;
  image?: string | null;
}) {
  const { t, i18n } = useTranslation();
  const n = (v: number, d = 1) => formatNumber(v, i18n.language, d);
  if (task === "object_detection") {
    const objects = p.details?.objects ?? [];
    return (
      <div className="space-y-2" role="status">
        <p>{t("journey.tasks.found", { count: objects.length })}</p>
        {image && <ImageResult src={image} p={p} />}
        {objects.length > 0 && (
          <ul className="text-sm text-muted">
            {objects.map((o, i) => (
              <li key={i}>
                {t("journey.tasks.object", { label: o.label, score: n(o.score * 100, 0) })}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  if (task === "segmentation")
    return (
      <div className="space-y-2" role="status">
        <p>{t("journey.tasks.area", { value: n(Number(p.prediction ?? 0)) })}</p>
        {image && <ImageResult src={image} p={p} />}
      </div>
    );
  if (task === "ocr")
    return (
      <p role="status">
        {t("journey.tasks.read")}{" "}
        <strong className="font-mono">«{String(p.prediction ?? "")}»</strong>
      </p>
    );
  return (
    <p role="status">
      {t("journey.use.prediction", {
        value: typeof p.prediction === "number" ? n(p.prediction, 4) : String(p.prediction),
      })}
      {typeof p.confidence === "number" && (
        <span className="ml-2 text-muted">
          ({t("journey.use.confidence", { value: n(p.confidence * 100, 0) })})
        </span>
      )}
    </p>
  );
}

/** Pronóstico por serie, o los momentos raros que encontró. */
export function SeriesView({
  task,
  predictions,
}: {
  task: string;
  predictions: ModelPrediction[];
}) {
  const { t, i18n } = useTranslation();
  if (task === "anomaly_detection") {
    const total = predictions.reduce((a, p) => a + Number(p.prediction ?? 0), 0);
    return (
      <div className="space-y-2 text-sm" role="status">
        <p className="font-semibold">
          {total
            ? t("journey.tasks.anomalies", { count: total })
            : t("journey.tasks.anomaliesNone")}
        </p>
        {predictions.map((p, i) =>
          (p.details?.anomalies ?? []).length ? (
            <div key={i}>
              {p.details?.series && <p className="text-muted">{p.details.series}</p>}
              <ul className="list-disc pl-5">
                {(p.details?.anomalies ?? []).slice(0, 20).map((when) => (
                  <li key={when}>{when.replace("T", " ")}</li>
                ))}
              </ul>
            </div>
          ) : null,
        )}
      </div>
    );
  }
  return (
    <div className="space-y-3 text-sm" role="status">
      {predictions.map((p, i) => {
        const values = Array.isArray(p.prediction) ? (p.prediction as number[]) : [];
        const times = p.details?.times ?? [];
        return (
          <div key={i}>
            {p.details?.series && (
              <p className="font-semibold">
                {t("journey.tasks.series", { name: p.details.series })}
              </p>
            )}
            <table className="mt-1 border-collapse text-xs">
              <thead>
                <tr>
                  <th className="border border-line px-2 py-1 text-left">
                    {t("journey.tasks.when")}
                  </th>
                  <th className="border border-line px-2 py-1 text-right">
                    {t("journey.tasks.value")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {values.map((v, k) => (
                  <tr key={k}>
                    <td className="border border-line px-2 py-1">
                      {(times[k] ?? `+${k + 1}`).replace("T", " ")}
                    </td>
                    <td className="border border-line px-2 py-1 text-right">
                      {formatNumber(v, i18n.language, 4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}
