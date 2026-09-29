import { Link } from "@tanstack/react-router";
import { Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Select,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import {
  useDatasets,
  useIngest,
  useUpload,
  type DatasetVersion,
  type DataSource,
  type SourcePreview,
} from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

import { ProfileView } from "./ProfileView";
import { RemoteSources } from "./RemoteSources";
import { ServerSources } from "./ServerSources";
import { downloadFromEngine } from "@/lib/api/download";
import { RetentionCard } from "./Retention";

export function UploadPanel({ onIngested }: { onIngested: (dv: DatasetVersion) => void }) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const upload = useUpload(projectId);
  const ingest = useIngest(projectId);
  const fileRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<{ source: DataSource; preview: SourcePreview } | null>(null);
  const [target, setTarget] = useState("");

  const onFiles = (list: FileList | null) => {
    if (!list?.length) return;
    upload.mutate(Array.from(list), {
      onSuccess: (res) => {
        setState(res);
        const cols = res.preview.columns;
        setTarget(cols.includes("target") ? "target" : "");
      },
    });
  };

  return (
    <Card>
      <CardTitle>{t("data.addTitle")}</CardTitle>
      <p className="mb-4 text-sm text-muted">{t("data.addHint")}</p>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="secondary"
          onClick={() => fileRef.current?.click()}
          loading={upload.isPending}
        >
          <Upload className="h-4 w-4" aria-hidden="true" />
          {t("data.uploadFile")}
        </Button>
        <Button
          variant="secondary"
          onClick={() => folderRef.current?.click()}
          disabled={upload.isPending}
        >
          {t("data.uploadFolder")}
        </Button>
        <input
          ref={fileRef}
          type="file"
          className="hidden"
          data-testid="file-input"
          accept=".csv,.tsv,.xlsx,.parquet,.json,.jsonl,.zip,.txt"
          onChange={(e) => onFiles(e.target.files)}
        />
        <input
          ref={folderRef}
          type="file"
          className="hidden"
          multiple
          data-testid="folder-input"
          {...{ webkitdirectory: "", directory: "" }}
          onChange={(e) => onFiles(e.target.files)}
        />
      </div>
      <RemoteSources
        projectId={projectId}
        onReady={(res) => {
          setState(res);
          setTarget(res.preview.columns.includes("target") ? "target" : "");
        }}
      />
      <ServerSources
        projectId={projectId}
        onReady={(res) => {
          setState(res);
          setTarget(res.preview.columns.includes("target") ? "target" : "");
        }}
      />
      {upload.progress && (
        <div className="mt-3 space-y-1" role="status" aria-live="polite">
          <div
            className="h-2 rounded bg-canvas"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round((upload.progress.sent / (upload.progress.total || 1)) * 100)}
          >
            <div
              className="h-2 rounded bg-primary transition-all"
              style={{
                width: `${Math.round((upload.progress.sent / (upload.progress.total || 1)) * 100)}%`,
              }}
            />
          </div>
          <p className="text-xs text-muted">
            {upload.progress.phase === "upload"
              ? t("data.uploading", {
                  pct: Math.round((upload.progress.sent / (upload.progress.total || 1)) * 100),
                  sent: (upload.progress.sent / 2 ** 20).toFixed(0),
                  total: (upload.progress.total / 2 ** 20).toFixed(0),
                })
              : t("data.analyzing")}
          </p>
        </div>
      )}
      <ErrorNote error={upload.error ?? ingest.error} />

      {state && (
        <div className="mt-6 space-y-4">
          <p className="text-sm">
            {t("data.previewOf", { name: state.source.name })}{" "}
            <Badge>{t(`sourceKind.${state.preview.kind}`)}</Badge>
          </p>
          {state.preview.classes && (
            <div className="text-sm">
              <p>
                {t("data.folderSummary", {
                  total: state.preview.total ?? 0,
                  count: Object.keys(state.preview.classes).length,
                })}
              </p>
              <p className="mt-1 flex flex-wrap gap-1">
                {Object.entries(state.preview.classes).map(([name, n]) => (
                  <Badge key={name}>
                    {name}: {n}
                  </Badge>
                ))}
              </p>
            </div>
          )}
          {state.preview.rows.length > 0 && (
            <Table>
              <thead>
                <tr>
                  {state.preview.columns.map((c) => (
                    <Th key={c}>{c}</Th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {state.preview.rows.slice(0, 5).map((row, i) => (
                  <tr key={i}>
                    {state.preview.columns.map((c) => (
                      <Td key={c} className="max-w-48 truncate">
                        {String((row as Record<string, unknown>)[c] ?? "")}
                      </Td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
          {state.preview.kind === "table" && (
            <Field label={t("data.target")} hint={t("data.targetHint")}>
              <Select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                aria-label={t("data.target")}
              >
                <option value="">{t("data.targetAuto")}</option>
                {state.preview.columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Select>
            </Field>
          )}
          {ingest.isPending && (
            <p className="text-xs text-muted" role="status">
              {t("data.ingesting")}
            </p>
          )}
          <Button
            loading={ingest.isPending}
            onClick={() =>
              ingest.mutate(
                { sourceId: state.source.id, target: target || null },
                {
                  onSuccess: (dv) => {
                    setState(null);
                    onIngested(dv);
                  },
                },
              )
            }
          >
            {t("data.ingest")}
          </Button>
        </div>
      )}
    </Card>
  );
}

export function DataPage() {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const { data, isPending, error } = useDatasets(projectId);
  const datasets = data ?? [];
  const [selected, setSelected] = useState<string | undefined>();
  const current = selected ?? datasets[0]?.id; // la más nueva

  return (
    <div className="space-y-6">
      <UploadPanel onIngested={(dv) => setSelected(dv.id)} />
      <Card>
        <CardTitle>{t("data.versions")}</CardTitle>
        {isPending && <Spinner />}
        <ErrorNote error={error} />
        {!isPending && datasets.length === 0 && <EmptyState>{t("data.empty")}</EmptyState>}
        {datasets.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>{t("data.version")}</Th>
                <Th>{t("data.modality")}</Th>
                <Th>{t("data.samples")}</Th>
                <Th>{t("data.target")}</Th>
                <Th>{t("data.created")}</Th>
                <Th />
              </tr>
            </thead>
            <tbody>
              {datasets.map((dv) => (
                <tr key={dv.id} className={dv.id === current ? "bg-canvas" : undefined}>
                  <Td className="font-mono text-xs">{dv.content_hash.slice(0, 10)}</Td>
                  <Td>{dv.modality ? t(`modality.${dv.modality}`) : "—"}</Td>
                  <Td>{dv.num_samples.toLocaleString(i18n.language)}</Td>
                  <Td>{dv.target ?? "—"}</Td>
                  <Td>{formatDate(dv.created_at, i18n.language)}</Td>
                  <Td className="text-right">
                    <Button size="sm" variant="ghost" onClick={() => setSelected(dv.id)}>
                      {t("data.showProfile")}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      title={t("data.dvcHint")}
                      onClick={() =>
                        void downloadFromEngine(
                          `/api/v1/datasets/${dv.id}/dvc.zip`,
                          `${dv.content_hash.slice(0, 12)}-dvc.zip`,
                        )
                      }
                    >
                      DVC
                    </Button>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
      <RetentionCard projectId={projectId} versions={datasets.length} />
      {current && (
        <>
          <ProfileView datasetVersionId={current} />
          <Link
            to="/projects/$projectId/train"
            params={{ projectId }}
            search={{}}
            className="inline-flex rounded-pt bg-primary px-4 py-2 text-sm font-semibold text-on-primary"
          >
            {t("data.goTrain")}
          </Link>
        </>
      )}
    </div>
  );
}
