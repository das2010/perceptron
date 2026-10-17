import { Link, useNavigate } from "@tanstack/react-router";
import { ArrowRight, Cpu, PackageOpen, Plus } from "lucide-react";
import { useRef, useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  Dialog,
  EmptyState,
  ErrorNote,
  Field,
  Input,
  PageHeader,
  Select,
  Spinner,
  Textarea,
} from "@/components/ui";
import { FIRST_MESSAGE } from "@/features/journey/JourneyPage";
import { EngineStatus } from "@/features/system/EngineStatus";
import {
  useCreateProject,
  useHardware,
  useImportProject,
  useProjects,
  useProjectTemplates,
  type ProjectTemplate,
} from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

const PRIVACY = ["L0", "L1", "L2", "L3"] as const;

function NewProjectDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const create = useCreateProject();
  const templates = useProjectTemplates();
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [privacy, setPrivacy] = useState<(typeof PRIVACY)[number]>("L1");
  const [template, setTemplate] = useState<ProjectTemplate | null>(null);

  const choose = (id: string) => {
    const next = templates.data?.find((tp) => tp.id === id) ?? null;
    // El ejemplo de objetivo solo reemplaza lo que no escribió el usuario.
    const example = (tp: ProjectTemplate | null) => (tp ? t(`templates.${tp.id}.goal`) : "");
    if (!goal || goal === example(template)) setGoal(example(next));
    setTemplate(next);
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    create.mutate(
      { name, goal, description: "", privacy_level: privacy, template: template?.id ?? null },
      {
        onSuccess: (p) => {
          onOpenChange(false);
          void navigate({ to: "/projects/$projectId/data", params: { projectId: p.id } });
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={t("home.newProject")}>
      <form onSubmit={submit} className="flex flex-col gap-4">
        <Field
          label={t("project.template")}
          hint={
            template
              ? t("project.templateHint", {
                  task: t(`task.${template.task}`),
                  modalities: template.modalities.map((m) => t(`modality.${m}`)).join(" + "),
                  metric: template.target_metric,
                })
              : t("project.noTemplateHint")
          }
        >
          <Select value={template?.id ?? ""} onChange={(e) => choose(e.target.value)}>
            <option value="">{t("project.noTemplate")}</option>
            {templates.data?.map((tp) => (
              <option key={tp.id} value={tp.id}>
                {`${tp.use_case} · ${t(`templates.${tp.id}.title`)}`}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("project.name")}>
          <Input required value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label={t("project.goal")} hint={t("project.goalHint")}>
          <Textarea value={goal} onChange={(e) => setGoal(e.target.value)} />
        </Field>
        <Field label={t("project.privacy")} hint={t(`privacy.${privacy}`)}>
          <Select
            value={privacy}
            onChange={(e) => setPrivacy(e.target.value as (typeof PRIVACY)[number])}
          >
            {PRIVACY.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </Select>
        </Field>
        <ErrorNote error={create.error} />
        <Button type="submit" loading={create.isPending}>
          {t("common.create")}
        </Button>
      </form>
    </Dialog>
  );
}

function HardwareCard() {
  const { t } = useTranslation();
  const { data, isPending } = useHardware();
  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <Cpu className="h-4 w-4" aria-hidden="true" />
        {t("home.hardware")}
      </CardTitle>
      {isPending && <Spinner />}
      {data && (
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <dt className="text-muted">{t("home.device")}</dt>
          <dd className="font-semibold uppercase">{data.recommended_device}</dd>
          <dt className="text-muted">{t("home.cpu")}</dt>
          <dd>{t("home.cores", { count: data.cpu.logical_cores })}</dd>
          <dt className="text-muted">{t("home.ram")}</dt>
          <dd>{data.ram_total_gb.toFixed(1)} GB</dd>
          <dt className="text-muted">{t("home.gpus")}</dt>
          <dd>{data.gpus?.length ? data.gpus.map((g) => g.name).join(", ") : t("home.noGpu")}</dd>
        </dl>
      )}
    </Card>
  );
}

export function HomePage() {
  const { t, i18n } = useTranslation();
  const [problem, setProblem] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);
  const guidedCreate = useCreateProject();
  const [open, setOpen] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const importer = useImportProject();
  const navigate = useNavigate();
  const { data, isPending, error } = useProjects();
  const all = Array.isArray(data) ? data : [];
  const archivedCount = all.filter((p) => p.status === "archived").length;
  const projects = showArchived ? all : all.filter((p) => p.status !== "archived");

  const start = () => {
    const text = problem.trim();
    if (!text || guidedCreate.isPending) return;
    guidedCreate.mutate(
      { name: text.slice(0, 80), goal: text, privacy_level: "L1", description: "", template: null },
      {
        onSuccess: (p) => {
          // La conversación arranca en el proyecto, con el «escribiendo…» del asistente.
          try {
            sessionStorage.setItem(FIRST_MESSAGE(p.id), text);
          } catch {
            // Sin almacenamiento de sesión: la persona lo vuelve a escribir allá.
          }
          void navigate({ to: "/projects/$projectId", params: { projectId: p.id } });
        },
      },
    );
  };

  return (
    <>
      <PageHeader
        title={t("journey.title")}
        actions={
          <details>
            <summary className="cursor-pointer text-sm text-muted">{t("journey.advanced")}</summary>
            <div className="flex items-center gap-2">
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-pt border border-line bg-card px-4 py-2 text-sm hover:bg-canvas">
                <PackageOpen className="h-4 w-4" aria-hidden="true" />
                {t("home.importProject")}
                <input
                  type="file"
                  accept=".perceptron,.zip"
                  className="sr-only"
                  aria-label={t("home.importProject")}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    importer.mutate(f, {
                      onSuccess: (p) =>
                        void navigate({ to: "/projects/$projectId", params: { projectId: p.id } }),
                    });
                    e.target.value = "";
                  }}
                />
              </label>
              <Button onClick={() => setOpen(true)}>
                <Plus className="h-4 w-4" aria-hidden="true" />
                {t("home.newProject")}
              </Button>
            </div>
          </details>
        }
      />
      <div className="mb-10 max-w-3xl">
        <p className="mb-4 text-lg text-muted">{t("journey.homeSubtitle")}</p>
        <form
          className="rounded-2xl border-2 border-line bg-card p-3 shadow-sm transition focus-within:border-brand"
          onSubmit={(e) => {
            e.preventDefault();
            start();
          }}
        >
          <textarea
            required
            rows={4}
            autoFocus
            ref={box}
            aria-label={t("journey.title")}
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                start();
              }
            }}
            placeholder={t("journey.homePlaceholder")}
            className="w-full resize-none bg-transparent px-2 py-1 text-lg text-ink outline-none placeholder:text-muted focus-visible:outline-none"
          />
          <div className="flex items-center justify-between gap-3 px-2">
            <span className="text-xs text-muted">{t("journey.enterHint")}</span>
            <Button type="submit" disabled={!problem.trim()} loading={guidedCreate.isPending}>
              {t("journey.begin")}
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </form>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-sm text-muted">{t("journey.tryOne")}</span>
          {(["sales", "cracks", "messages"] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => {
                setProblem(t(`journey.examples.${k}`));
                box.current?.focus();
              }}
              className="rounded-full border border-line px-3 py-1 text-sm hover:border-brand hover:text-brand"
            >
              {t(`journey.examples.${k}`)}
            </button>
          ))}
        </div>
        <ErrorNote error={guidedCreate.error} />
      </div>
      <details>
        <summary className="mb-3 cursor-pointer text-sm text-muted">
          {t("journey.advanced")}
        </summary>
        <div className="grid gap-4 md:grid-cols-2">
          <EngineStatus />
          <HardwareCard />
        </div>
      </details>
      <ErrorNote error={importer.error} />
      <div className="mb-3 mt-8 flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">{t("home.recent")}</h2>
        {archivedCount > 0 && (
          <label className="flex items-center gap-2 text-sm text-muted">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(e) => setShowArchived(e.target.checked)}
            />
            {t("home.showArchived", { count: archivedCount })}
          </label>
        )}
      </div>
      {isPending && <Spinner />}
      <ErrorNote error={error} />
      {!isPending && !error && projects.length === 0 && <EmptyState>{t("home.empty")}</EmptyState>}
      <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {projects.map((p) => (
          <li key={p.id}>
            <Link
              to="/projects/$projectId"
              params={{ projectId: p.id }}
              className="block rounded-pt border border-line bg-card p-4 hover:border-brand"
            >
              <p className="font-semibold">{p.name}</p>
              {p.goal && <p className="mt-1 line-clamp-2 text-sm text-muted">{p.goal}</p>}
              <div className="mt-3 flex flex-wrap gap-2">
                {p.status === "archived" && <Badge>{t("project.archived")}</Badge>}
                <Badge>{p.privacy_level}</Badge>
                {p.modalities?.map((m) => (
                  <Badge key={m}>{t(`modality.${m}`)}</Badge>
                ))}
                <span className="text-xs text-muted">
                  {formatDate(p.updated_at, i18n.language)}
                </span>
              </div>
            </Link>
          </li>
        ))}
      </ul>
      <NewProjectDialog open={open} onOpenChange={setOpen} />
    </>
  );
}
