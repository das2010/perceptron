/**
 * Pantalla de preparación del desktop (ADR-0026): mientras se aprovisiona el runtime Python
 * embebido (primer arranque o nueva versión) y arranca el Engine, muestra el progreso.
 * En la web no hace nada.
 */
import { Check, RotateCcw } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, CardTitle, Spinner } from "@/components/ui";
import { resetApiClient } from "@/lib/api/client";
import { getPlatform, type RuntimeProgress } from "@/lib/platform/bridge";

const RUNTIME_STEPS = ["python", "venv", "deps", "engine", "hardware", "torch"] as const;

export function RuntimeGate({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const bridge = getPlatform();
  const [phase, setPhase] = useState<"waiting" | "ready" | "failed">(
    bridge.runtime ? "waiting" : "ready",
  );
  const [progress, setProgress] = useState<Record<string, RuntimeProgress>>({});
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const runtime = bridge.runtime;
    if (!runtime) return;
    let alive = true;
    let unlisten: (() => void) | undefined;
    void runtime
      .onProgress((p) => {
        if (alive) setProgress((prev) => ({ ...prev, [p.step]: p }));
      })
      .then((u) => {
        unlisten = u;
      });
    bridge.engine().then(
      () => {
        if (alive) setPhase("ready");
      },
      (e: unknown) => {
        if (!alive) return;
        setError(String(e));
        setPhase("failed");
      },
    );
    return () => {
      alive = false;
      unlisten?.();
    };
  }, [bridge, attempt]);

  if (phase === "ready") return <>{children}</>;

  const retry = () => {
    setPhase("waiting");
    setError(null);
    setProgress({});
    const done = () => {
      resetApiClient();
      setAttempt((a) => a + 1);
    };
    void bridge.runtime?.restart().then(done, done);
  };
  // El paso en curso es el último del que llegó progreso (hasta que llega "done").
  const last = RUNTIME_STEPS.reduce((acc, step, i) => (progress[step] ? i : acc), -1);

  return (
    <main className="flex min-h-screen items-center justify-center bg-canvas p-6">
      <Card className="w-full max-w-lg">
        <CardTitle>{t("desktop.preparing")}</CardTitle>
        <p className="mb-4 text-sm text-muted">{t("desktop.preparingHint")}</p>
        <ol className="space-y-2" aria-label={t("desktop.steps")}>
          {RUNTIME_STEPS.map((step, i) => {
            const seen = Boolean(progress[step]);
            const running = phase === "waiting" && i === last && !progress.done;
            return (
              <li key={step} className="flex items-center gap-2 text-sm">
                {running ? (
                  <Spinner />
                ) : seen ? (
                  <Check className="h-4 w-4 text-ok" aria-hidden="true" />
                ) : (
                  <span className="inline-block h-4 w-4 rounded-full border border-line" />
                )}
                <span className={seen ? "" : "text-muted"}>
                  {progress[step]?.message ?? t(`desktop.step.${step}`)}
                </span>
              </li>
            );
          })}
        </ol>
        {phase === "waiting" && Object.keys(progress).length === 0 && (
          <div className="mt-4">
            <Spinner label={t("desktop.starting")} />
          </div>
        )}
        {phase === "failed" && (
          <div className="mt-4 space-y-3">
            <p
              role="alert"
              className="rounded-pt border border-bad p-2 text-xs whitespace-pre-wrap"
            >
              {error}
            </p>
            <Button onClick={retry}>
              <RotateCcw className="h-4 w-4" aria-hidden="true" />
              {t("desktop.retry")}
            </Button>
          </div>
        )}
      </Card>
    </main>
  );
}
