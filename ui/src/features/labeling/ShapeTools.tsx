/**
 * Formas de etiquetado (RF-LBL-01): polígonos sobre la imagen (máscaras de segmentación) y
 * segmentos temporales sobre el audio (eventos). Coordenadas normalizadas 0–1 y segundos.
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui";
import type { Polygon, Segment } from "@/lib/api/hooks";

type Point = [number, number];

/** Clic agrega un vértice; clic sobre el primero (o «Cerrar») cierra el polígono. */
export function PolygonCanvas({
  url,
  polygons,
  current,
  onAdd,
}: {
  url: string;
  polygons: Polygon[];
  current: string;
  onAdd: (p: Polygon) => void;
}) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<Point[]>([]);
  const close = () => {
    if (draft.length >= 3) onAdd({ points: draft, label: current });
    setDraft([]);
  };
  const pos = (e: React.MouseEvent): Point => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return [0, 0];
    return [
      Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    ];
  };
  const first = draft[0];
  const pts = (p: Point[]) => p.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");
  return (
    <div className="space-y-2">
      <div
        ref={ref}
        className="relative inline-block cursor-crosshair select-none"
        data-testid="polygon-canvas"
        onClick={(e) => {
          const p = pos(e);
          // Cerca del primer vértice (2 % del lado): se cierra.
          if (first && draft.length >= 3 && Math.hypot(p[0] - first[0], p[1] - first[1]) < 0.02)
            close();
          else setDraft((d) => [...d, p]);
        }}
      >
        <img src={url} alt="" className="max-h-96 rounded-pt" draggable={false} />
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {polygons.map((p, i) => (
            <polygon
              key={i}
              points={pts(p.points as Point[])}
              className="fill-pt-lime/30 stroke-pt-lime"
              strokeWidth={0.5}
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {draft.length > 0 && (
            <polyline
              points={pts(draft)}
              className="fill-none stroke-pt-lime"
              strokeWidth={0.5}
              strokeDasharray="1 1"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        <span>{t("labeling.polygonHint")}</span>
        <Button size="sm" variant="secondary" disabled={draft.length < 3} onClick={close}>
          {t("labeling.closePolygon")}
        </Button>
        {draft.length > 0 && (
          <Button size="sm" variant="ghost" onClick={() => setDraft([])}>
            {t("labeling.discard")}
          </Button>
        )}
      </div>
    </div>
  );
}

/** Reproductor con «Inicio» / «Fin» en el tiempo actual; la barra muestra los segmentos. */
export function SegmentMarker({
  url,
  segments,
  current,
  onAdd,
}: {
  url: string;
  segments: Segment[];
  current: string;
  onAdd: (s: Segment) => void;
}) {
  const { t } = useTranslation();
  const audio = useRef<HTMLAudioElement>(null);
  const [start, setStart] = useState<number | null>(null);
  const [duration, setDuration] = useState(0);
  const now = () => audio.current?.currentTime ?? 0;
  const pct = (s: number) => (duration ? `${Math.min(100, (s / duration) * 100)}%` : "0%");
  return (
    <div className="space-y-2">
      <audio
        ref={audio}
        src={url}
        controls
        className="w-full max-w-xl"
        aria-label={t("labeling.sampleAlt")}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
      />
      <div
        className="relative h-4 w-full max-w-xl rounded bg-canvas"
        data-testid="segment-bar"
        aria-hidden="true"
      >
        {segments.map((s, i) => (
          <span
            key={i}
            className="absolute inset-y-0 rounded bg-pt-lime/70"
            style={{ left: pct(s.start_s), width: pct(s.end_s - s.start_s) }}
            title={`${s.label} ${s.start_s.toFixed(2)}–${s.end_s.toFixed(2)} s`}
          />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="secondary" onClick={() => setStart(now())}>
          {t("labeling.markStart")}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          disabled={start === null}
          onClick={() => {
            const end = now();
            if (start !== null && end > start)
              onAdd({ start_s: start, end_s: end, label: current });
            setStart(null);
          }}
        >
          {t("labeling.markEnd")}
        </Button>
        <span className="text-xs text-muted">
          {start === null
            ? t("labeling.segmentHint")
            : t("labeling.segmentFrom", { s: start.toFixed(2) })}
        </span>
      </div>
      {segments.length > 0 && (
        <ul className="text-xs">
          {segments.map((s, i) => (
            <li key={i} className="font-mono">
              {s.label}: {s.start_s.toFixed(2)}–{s.end_s.toFixed(2)} s
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
