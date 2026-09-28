/**
 * Modo experto (RF-ARC-06): el modelo como código Python con la interfaz fija
 * `build_model(config) -> nn.Module`. Validación estática en vivo; al guardar, el Engine lo
 * prueba en el sandbox (SPEC §13.2, ADR-0025). Requiere confirmación explícita.
 */
import { useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { ArrowLeft, ShieldAlert } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { CodeMarker } from "@/components/code-marker";
import { Badge, Button, Card, CardTitle, ErrorNote, Field, Input, Spinner } from "@/components/ui";
import {
  lintArchCode,
  useArchCodeStarter,
  useArchSource,
  useArchSpec,
  useCreateCodeArchSpec,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

const CodeView = lazy(() => import("@/components/CodeView"));

function useLint(source: string) {
  const [issues, setIssues] = useState<CodeMarker[] | null>(null);
  useEffect(() => {
    let alive = true;
    const handle = setTimeout(() => {
      lintArchCode(source).then(
        (r) => alive && setIssues(r.issues ?? []),
        () => alive && setIssues(null),
      );
    }, 400);
    return () => {
      alive = false;
      clearTimeout(handle);
    };
  }, [source]);
  return issues;
}

function Editor({
  projectId,
  archspecId,
  initial,
  baseName,
}: {
  projectId: string;
  archspecId: string;
  initial: string;
  baseName: string;
}) {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const [source, setSource] = useState(initial);
  const [name, setName] = useState(`${baseName.replace(/-codigo$/, "")}-codigo`);
  const [ack, setAck] = useState(false);
  const issues = useLint(source);
  const create = useCreateCodeArchSpec(projectId);
  const dark = document.documentElement.dataset.theme === "dark";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("arch.back")}
          onClick={() =>
            void navigate({
              to: "/projects/$projectId/archspecs/$archspecId",
              params: { projectId, archspecId },
            })
          }
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h2 className="font-semibold">{t("expert.title")}</h2>
        {issues && (
          <Badge tone={issues.length ? "bad" : "ok"}>
            {issues.length ? t("expert.issues", { count: issues.length }) : t("expert.lintOk")}
          </Badge>
        )}
      </div>
      <div role="note" className="flex gap-3 rounded-pt border border-warn bg-canvas p-3 text-sm">
        <ShieldAlert className="h-5 w-5 shrink-0 text-warn" aria-hidden="true" />
        <p>{t("expert.warning")}</p>
      </div>
      <Card className="p-0">
        <Suspense fallback={<Spinner />}>
          <CodeView
            code={source}
            onChange={setSource}
            markers={issues ?? []}
            dark={dark}
            height={520}
            label={t("expert.editor")}
          />
        </Suspense>
      </Card>
      {issues && issues.length > 0 && (
        <ul className="space-y-1 text-xs">
          {issues.map((i, k) => (
            <li key={k}>
              <span className="font-mono">{t("expert.line", { line: i.line })}</span> {i.message}
            </li>
          ))}
        </ul>
      )}
      <Card>
        <CardTitle>{t("expert.saveTitle")}</CardTitle>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("arch.name")}>
            <Input value={name} onChange={(e) => setName(e.target.value)} className="w-64" />
          </Field>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} />
            {t("expert.acknowledge")}
          </label>
          <Button
            disabled={!ack || !issues || issues.length > 0}
            loading={create.isPending}
            onClick={() =>
              create.mutate(
                {
                  base_archspec_id: archspecId,
                  source,
                  name: name.trim() || null,
                  acknowledge_risk: ack,
                },
                {
                  onSuccess: (res) =>
                    void navigate({
                      to: "/projects/$projectId/archspecs/$archspecId",
                      params: { projectId, archspecId: res.record.id ?? "" },
                    }),
                },
              )
            }
          >
            {t("expert.save")}
          </Button>
        </div>
        {create.isPending && <p className="mt-2 text-xs text-muted">{t("expert.checking")}</p>}
        {create.data && (
          <p className="mt-2 text-sm">
            {t("expert.checked", {
              params: formatNumber(create.data.check.num_params ?? null, i18n.language, 0),
            })}
          </p>
        )}
        <ErrorNote error={create.error} />
      </Card>
    </div>
  );
}

export function ExpertCodePage() {
  const { projectId, archspecId } = useParams({
    from: "/projects/$projectId/archspecs/$archspecId/code",
  });
  const { from } = useSearch({ from: "/projects/$projectId/archspecs/$archspecId/code" });
  const record = useArchSpec(archspecId);
  const isCode = Boolean(record.data?.code_path);
  const current = useArchSource(archspecId, isCode && from === "current");
  const starter = useArchCodeStarter(
    archspecId,
    Boolean(record.data) && !(isCode && from === "current"),
  );
  const initial = isCode && from === "current" ? current.data : starter.data;
  const error = record.error ?? current.error ?? starter.error;
  if (error) return <ErrorNote error={error} />;
  if (!record.data || initial === undefined) return <Spinner />;
  return (
    <Editor
      key={`${archspecId}:${from ?? "starter"}`}
      projectId={projectId}
      archspecId={archspecId}
      initial={initial}
      baseName={record.data.name}
    />
  );
}
