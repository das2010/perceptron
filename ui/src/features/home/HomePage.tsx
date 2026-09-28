import { Link, useNavigate } from "@tanstack/react-router";
import { Cpu, Plus } from "lucide-react";
import { useState, type FormEvent } from "react";
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
import { EngineStatus } from "@/features/system/EngineStatus";
import { useCreateProject, useHardware, useProjects } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

const PRIVACY = ["L0", "L1", "L2", "L3"] as const;

function NewProjectDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const create = useCreateProject();
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [privacy, setPrivacy] = useState<(typeof PRIVACY)[number]>("L1");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    create.mutate(
      { name, goal, description: "", privacy_level: privacy },
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
        <Field label={t("project.name")}>
          <Input required value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label={t("project.goal")} hint={t("project.goalHint")}>
          <Textarea value={goal} onChange={(e) => setGoal(e.target.value)} />
        </Field>
        <Field label={t("project.privacy")} hint={t(`privacy.${privacy}`)}>
          <Select value={privacy} onChange={(e) => setPrivacy(e.target.value as (typeof PRIVACY)[number])}>
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
  const [open, setOpen] = useState(false);
  const { data, isPending, error } = useProjects();
  const projects = Array.isArray(data) ? data : [];

  return (
    <>
      <PageHeader
        title={t("app.tagline")}
        actions={
          <Button onClick={() => setOpen(true)}>
            <Plus className="h-4 w-4" aria-hidden="true" />
            {t("home.newProject")}
          </Button>
        }
      />
      <div className="grid gap-4 md:grid-cols-2">
        <EngineStatus />
        <HardwareCard />
      </div>
      <h2 className="mb-3 mt-8 text-lg font-semibold">{t("home.recent")}</h2>
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
                <Badge>{p.privacy_level}</Badge>
                {p.modalities?.map((m) => (
                  <Badge key={m}>{t(`modality.${m}`)}</Badge>
                ))}
                <span className="text-xs text-muted">{formatDate(p.updated_at, i18n.language)}</span>
              </div>
            </Link>
          </li>
        ))}
      </ul>
      <NewProjectDialog open={open} onOpenChange={setOpen} />
    </>
  );
}
