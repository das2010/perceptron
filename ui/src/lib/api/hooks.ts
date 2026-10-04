/**
 * Hooks de datos por recurso (TanStack Query) sobre el cliente tipado de OpenAPI.
 * Ninguna lógica de ML vive en la UI (CLAUDE.md): solo llamadas al Engine.
 */
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { getApiClient } from "./client";
import { postForm } from "./upload";
import type { components } from "./schema";
import { getPlatform } from "@/lib/platform/bridge";

export type Schemas = components["schemas"];
/** Las entidades siempre traen `id` en las respuestas (en el schema figura con default). */
type WithId<T> = T & { id: string };

export type Project = WithId<Schemas["Project"]>;
export type ProjectTemplate = Schemas["ProjectTemplate"];
export type DatasetVersion = WithId<Schemas["DatasetVersion"]>;
export type SemanticType = Schemas["SemanticType"];
export type SymbolicFit = WithId<Schemas["SymbolicFit"]>;
export type ProfileCard = Schemas["ProfileCard"];
export type Run = WithId<Schemas["Run"]>;
export type ModelVersion = WithId<Schemas["ModelVersion"]>;
export type EvaluationReport = Schemas["EvaluationReport"];
// Tipos de respuesta (el schema distingue entrada/salida de algunos modelos).
export type ArchProposals = NonNullable<ReturnType<typeof useProposeArchitecture>["data"]>;
export type HPOStrategy = NonNullable<ReturnType<typeof useHpoStrategy>["data"]>;
export type Budget = Schemas["Budget"];
export type Pipeline = Schemas["Pipeline"];
export type Job = Schemas["Job"];
export type DataSource = WithId<Schemas["DataSource"]>;
export type SourcePreview = Schemas["SourcePreview"];
export type ProviderView = Schemas["ProviderView"];
export type ProfilesView = Schemas["ProfilesView"];
export type LLMCall = WithId<Schemas["LLMCall"]>;
export type Diagnosis = Schemas["Diagnosis"];
export type Report = Schemas["Report"];
export type HardwareReport = Schemas["HardwareReport"];

/** Error del Engine con el `code` estable de `PerceptronError` (para i18n). */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
    readonly details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

interface ApiResult<T> {
  data?: T;
  error?: unknown;
  response: Response;
}

export function unwrap<T>({ data, error, response }: ApiResult<T>): T {
  if (error !== undefined || data === undefined) {
    const body = (error ?? {}) as {
      code?: string;
      message?: string;
      details?: Record<string, unknown>;
    };
    throw new ApiError(
      body.message ?? `HTTP ${response.status}`,
      body.code ?? "http_error",
      response.status,
      body.details ?? {},
    );
  }
  return data;
}

export const keys = {
  projects: ["projects"] as const,
  project: (id: string) => ["projects", id] as const,
  datasets: (pid: string) => ["projects", pid, "datasets"] as const,
  profile: (dv: string) => ["datasets", dv, "profile"] as const,
  runs: (pid: string) => ["projects", pid, "runs"] as const,
  run: (rid: string) => ["runs", rid] as const,
  evaluation: (rid: string) => ["runs", rid, "evaluation"] as const,
  models: (pid: string) => ["projects", pid, "models"] as const,
  hardware: ["system", "hardware"] as const,
  providers: ["llm", "providers"] as const,
  llmProfiles: ["llm", "profiles"] as const,
  audit: (pid: string) => ["llm", "audit", pid] as const,
  job: (id: string) => ["jobs", id] as const,
  symbolic: (pid: string) => ["projects", pid, "symbolic"] as const,
};

// ---------------------------------------------------------------- proyectos

export function useProjects() {
  return useQuery({
    queryKey: keys.projects,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/projects")) as Project[],
  });
}

export function useProject(projectId: string) {
  return useQuery({
    queryKey: keys.project(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}", {
          params: { path: { project_id: projectId } },
        }),
      ) as Project,
  });
}

export function useProjectTemplates() {
  return useQuery({
    queryKey: ["projects", "templates"] as const,
    staleTime: Infinity,
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/projects/templates")) as ProjectTemplate[],
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["ProjectCreate"]) =>
      unwrap(await (await getApiClient()).POST("/api/v1/projects", { body })) as Project,
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.projects }),
  });
}

/** Archivar/restaurar (y otros cambios) con bloqueo optimista por `version`. */
export function useUpdateProject(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["ProjectPatch"]) =>
      unwrap(
        await (
          await getApiClient()
        ).PATCH("/api/v1/projects/{project_id}", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ) as Project,
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.projects }),
  });
}

/** Importa un paquete .perceptron (RF-PRJ-03). */
export function useImportProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file, file.name);
      return unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/import", {
          body: form as unknown as Schemas["Body_importProject"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      ) as Project;
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.projects }),
  });
}

export function useDuplicateProject(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (name?: string) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/duplicate", {
          params: { path: { project_id: projectId } },
          body: { name: name || null },
        }),
      ) as Project,
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.projects }),
  });
}

export function useDeleteProject(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const res = await (
        await getApiClient()
      ).DELETE("/api/v1/projects/{project_id}", {
        params: { path: { project_id: projectId } },
      });
      if (res.error) unwrap(res);
    },
    onSuccess: () => {
      qc.removeQueries({ queryKey: keys.project(projectId) });
      void qc.invalidateQueries({ queryKey: keys.projects });
    },
  });
}

// ---------------------------------------------------------------- datos

export function useDatasets(projectId: string) {
  return useQuery({
    queryKey: keys.datasets(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/datasets", {
          params: { path: { project_id: projectId } },
        }),
      ) as DatasetVersion[],
  });
}

export function useProfile(datasetVersionId: string | undefined) {
  return useQuery({
    queryKey: keys.profile(datasetVersionId ?? ""),
    enabled: Boolean(datasetVersionId),
    queryFn: async () => {
      const api = await getApiClient();
      const path = { dataset_version_id: datasetVersionId ?? "" };
      const got = await api.GET("/api/v1/datasets/{dataset_version_id}/profile", {
        params: { path },
      });
      if (got.response.status === 404) {
        // Todavía no se perfiló: se calcula ahora (RF-PRF-07).
        return unwrap(
          await api.POST("/api/v1/datasets/{dataset_version_id}/profile", { params: { path } }),
        );
      }
      return unwrap(got);
    },
  });
}

/** Sube archivos o una carpeta (rutas relativas) y la previsualiza (RF-ING-01). */
/** Avance de una subida: bytes enviados y luego el análisis del contenido (vista previa). */
export interface UploadProgress {
  phase: "upload" | "preview";
  sent: number;
  total: number;
}

export function useUpload(projectId: string) {
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const mutation = useMutation({
    mutationFn: async (files: File[]) => {
      const form = new FormData();
      for (const f of files) form.append("files", f, f.webkitRelativePath || f.name);
      const total = files.reduce((n, f) => n + f.size, 0);
      setProgress({ phase: "upload", sent: 0, total });
      const source = await postForm<DataSource>(
        `/api/v1/projects/${projectId}/uploads`,
        form,
        (fraction) => setProgress({ phase: "upload", sent: Math.round(fraction * total), total }),
      );
      setProgress({ phase: "preview", sent: total, total });
      const api = await getApiClient();
      const preview = unwrap(
        await api.POST("/api/v1/sources/{source_id}/preview", {
          params: { path: { source_id: source.id } },
        }),
      );
      return { source, preview };
    },
    onSettled: () => setProgress(null),
  });
  return Object.assign(mutation, { progress });
}

export function useIngest(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({
      sourceId,
      target,
      overrides,
    }: {
      sourceId: string;
      target?: string | null;
      /** Corrección manual de tipos (RF-ING-06): crea otra versión del dataset. */
      overrides?: Record<string, SemanticType>;
    }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/sources/{source_id}/ingest", {
          params: { path: { source_id: sourceId } },
          body: { target: target || null, overrides: overrides ?? null },
        }),
      ) as DatasetVersion,
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets(projectId) }),
  });
}

// ---------------------------------------------------------------- entrenamiento

export function useProposePipeline(projectId: string) {
  return useMutation({
    mutationFn: async (datasetVersionId: string) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/pipelines/propose", {
          params: { path: { project_id: projectId } },
          body: { dataset_version_id: datasetVersionId },
        }),
      ),
  });
}

export function useProposeArchitecture(projectId: string) {
  return useMutation({
    mutationFn: async (body: Schemas["ProposeArchBody"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/arch/propose", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
  });
}

export function useHpoStrategy(projectId: string) {
  return useMutation({
    mutationFn: async (body: Schemas["StrategyBody"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/hpo/strategy", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
  });
}

export function useCreateStudy(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["StudyCreate"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/studies", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.runs(projectId) }),
  });
}

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: keys.job(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "succeeded" || status === "failed" || status === "cancelled" ? false : 2000;
    },
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/jobs/{job_id}", {
          params: { path: { job_id: jobId ?? "" } },
        }),
      ),
  });
}

export function useRuns(projectId: string, refetchMs?: number) {
  return useQuery({
    queryKey: keys.runs(projectId),
    refetchInterval: refetchMs ?? false,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/runs", {
          params: { path: { project_id: projectId } },
        }),
      ) as Run[],
  });
}

/** Enlace al run en la UI de MLflow (opcional, RF-TRK-02); el desktop la levanta. */
export function useOpenInMlflow(runId: string) {
  return useMutation({
    mutationFn: async () => {
      const link = unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/mlflow", { params: { path: { run_id: runId } } }),
      ) as { url: string };
      await getPlatform().openExternal(link.url);
      return link.url;
    },
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: keys.run(runId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/runs/{run_id}", {
          params: { path: { run_id: runId } },
        }),
      ) as Run,
  });
}

export function useEvaluation(runId: string) {
  return useQuery({
    queryKey: keys.evaluation(runId),
    retry: false,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/runs/{run_id}/evaluation", {
          params: { path: { run_id: runId } },
        }),
      ),
  });
}

export function useEvaluate(runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/evaluate", {
          params: { path: { run_id: runId } },
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.evaluation(runId) }),
  });
}

export function useRegister(runId: string, projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/register", {
          params: { path: { run_id: runId } },
        }),
      ) as ModelVersion,
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.models(projectId) }),
  });
}

export function useDiagnosis(runId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["runs", runId, "diagnosis"],
    enabled,
    retry: false,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/runs/{run_id}/diagnosis", {
          params: { path: { run_id: runId } },
        }),
      ),
  });
}

export function useReport(runId: string) {
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/report", {
          params: { path: { run_id: runId } },
          body: { mode: "auto", language: "español" },
        }),
      ),
  });
}

export function useModels(projectId: string) {
  return useQuery({
    queryKey: keys.models(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/models", {
          params: { path: { project_id: projectId } },
        }),
      ) as ModelVersion[],
  });
}

// ---------------------------------------------------------------- sistema y LLM

export function useHardware() {
  return useQuery({
    queryKey: keys.hardware,
    staleTime: 60_000,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/system/hardware")),
  });
}

export function useProviders() {
  return useQuery({
    queryKey: keys.providers,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/llm/providers")),
  });
}

export function useSaveProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ name, body }: { name: string; body: Schemas["ProviderUpdate"] }) =>
      unwrap(
        await (
          await getApiClient()
        ).PUT("/api/v1/llm/providers/{name}", {
          params: { path: { name } },
          body,
        }),
      ),
    onSuccess: (data) => qc.setQueryData(keys.providers, data),
  });
}

export function useLlmProfiles() {
  return useQuery({
    queryKey: keys.llmProfiles,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/llm/profiles")),
  });
}

export function useSetActiveProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (active: string) =>
      unwrap(
        await (
          await getApiClient()
        ).PUT("/api/v1/llm/profiles", { body: { active, profiles: {} } }),
      ),
    onSuccess: (data) => qc.setQueryData(keys.llmProfiles, data),
  });
}

export function useTestLlm() {
  return useMutation({
    mutationFn: async (projectId?: string) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/llm/test", {
          body: { project_id: projectId ?? null, purpose: "copilot" },
        }),
      ),
  });
}

export function useAudit(projectId: string) {
  return useQuery({
    queryKey: keys.audit(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/llm/audit", {
          params: { query: { project: projectId, limit: 200 } },
        }),
      ) as LLMCall[],
  });
}

// ---------------------------------------------------------------- wizard (RF-WIZ-04)

export type DraftView = Schemas["DraftView"];
export type DraftValues = Schemas["DraftValues"];
export type UseCaseBrief = Schemas["UseCaseBrief"];
export type WizardPlan = Schemas["WizardPlan"];
export type BriefPatch = Schemas["BriefPatch"];
export type IntakeTurn = Schemas["IntakeTurn"];
export type PlanDiff = Schemas["PlanDiff"];

/** Reconciliación de la ficha con el perfil de los datos (ADR-0040, fase 2). */
export function useDraftReconcile(projectId: string) {
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/draft/reconcile", {
          params: { path: { project_id: projectId } },
        }),
      ),
  });
}

/** Entrevista del wizard (ADR-0040): propone cambios a la ficha; no aplica nada. */
export function useDraftIntake(projectId: string) {
  return useMutation({
    mutationFn: async (body: { message: string; history: IntakeTurn[] }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/draft/intake", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
  });
}

export function useDraft(projectId: string | undefined) {
  return useQuery({
    queryKey: ["projects", projectId ?? "", "draft"],
    enabled: Boolean(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/draft", {
          params: { path: { project_id: projectId ?? "" } },
        }),
      ),
  });
}

/** Guarda cambios del borrador con bloqueo optimista (usa la versión en caché). */
export function useUpdateDraft(projectId: string) {
  const qc = useQueryClient();
  const key = ["projects", projectId, "draft"];
  return useMutation({
    // En serie: cada guardado lee la versión que dejó el anterior (si no, el blur de un campo
    // y el "Siguiente" salen con la misma versión y el segundo choca con 409).
    scope: { id: `draft:${projectId}` },
    mutationFn: async ({
      values,
      step,
      origin = "user",
    }: {
      values?: Partial<DraftValues>;
      step?: string;
      origin?: "user" | "copilot";
    }) => {
      const current = qc.getQueryData<DraftView>(key);
      return unwrap(
        await (
          await getApiClient()
        ).PATCH("/api/v1/projects/{project_id}/draft", {
          params: { path: { project_id: projectId } },
          body: {
            version: current?.draft.version ?? 1,
            values: (values ?? {}) as Record<string, unknown>,
            ...(step ? { step } : {}),
            origin,
          },
        }),
      );
    },
    onSuccess: (data) => {
      qc.setQueryData(key, data);
      void qc.invalidateQueries({ queryKey: keys.project(projectId) });
    },
    // Si otro cliente (o el copiloto) cambió el borrador, recargarlo para el próximo intento.
    onError: () => void qc.invalidateQueries({ queryKey: key }),
  });
}

// ---------------------------------------------------------------- editores (RF-ARC-05, RF-PIP-02)

export type ArchSpecRecord = WithId<Schemas["ArchSpecRecord"]>;
export type ValidationReport = Awaited<ReturnType<typeof validateArchSpec>>;
export type PipelineSpec = Schemas["PipelineSpec"];

export function useArchSpecs(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "archspecs"],
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/archspecs", {
          params: { path: { project_id: projectId } },
        }),
      ) as ArchSpecRecord[],
  });
}

export function useArchSpec(archspecId: string) {
  return useQuery({
    queryKey: ["archspecs", archspecId],
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/archspecs/{archspec_id}", {
          params: { path: { archspec_id: archspecId } },
        }),
      ) as ArchSpecRecord,
  });
}

export function useSaveArchSpec(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (spec: Record<string, unknown>) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/archspecs", {
          params: { path: { project_id: projectId } },
          body: { spec: spec as Schemas["ArchSpec"] },
        }),
      ) as ArchSpecRecord,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["projects", projectId, "archspecs"] }),
  });
}

export function useCatalogBlocks(modality: string | undefined) {
  return useQuery({
    queryKey: ["catalog", "blocks", modality],
    enabled: Boolean(modality),
    staleTime: Infinity,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/catalog/blocks", {
          params: { query: { modality: modality as Schemas["Modality"] } },
        }),
      ),
  });
}

export async function validateArchSpec(spec: Record<string, unknown>) {
  return unwrap(await (await getApiClient()).POST("/api/v1/arch/validate", { body: spec }));
}

export function useArchCode(spec: Record<string, unknown> | null, enabled: boolean) {
  return useQuery({
    queryKey: ["arch", "code", spec],
    enabled: enabled && spec !== null,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/arch/to-code", { body: spec as Schemas["ArchSpec"] }),
      ).code,
  });
}

export function usePipelines(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "pipelines"],
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/pipelines", {
          params: { path: { project_id: projectId } },
        }),
      ),
  });
}

export function usePipeline(pipelineId: string) {
  return useQuery({
    queryKey: ["pipelines", pipelineId],
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/pipelines/{pipeline_id}", {
          params: { path: { pipeline_id: pipelineId } },
        }),
      ),
  });
}

/** Guarda el grafo del pipeline con bloqueo optimista (versión de la caché). */
export function useUpdatePipeline(pipelineId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (graph: PipelineSpec) => {
      const current = qc.getQueryData<Pipeline>(["pipelines", pipelineId]);
      return unwrap(
        await (
          await getApiClient()
        ).PUT("/api/v1/pipelines/{pipeline_id}", {
          params: { path: { pipeline_id: pipelineId } },
          body: { graph, version: current?.version ?? 1 },
        }),
      );
    },
    onSuccess: (data) => {
      qc.setQueryData(["pipelines", pipelineId], data);
      void qc.invalidateQueries({ queryKey: ["projects", data.project_id, "pipelines"] });
    },
  });
}

/** Salida intermedia de un grafo (guardado o no) tras un paso, sobre train (RF-PIP-02). */
export function usePreviewSteps(projectId: string) {
  return useMutation({
    mutationFn: async (body: {
      dataset_version_id: string;
      graph: PipelineSpec;
      upto_step: string | null;
    }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/pipelines/preview-steps", {
          params: { path: { project_id: projectId } },
          body: { ...body, rows: 10 },
        }),
      ),
  });
}

// ---------------------------------------------------------------- sub-wizard de definición (§7.6)

export type DefinePlan = Schemas["DefinePlan"];

export function useDefinitionPlan(
  projectId: string,
  datasetVersionId: string,
  pipelineId: string,
  choices: Record<string, string>,
) {
  return useQuery({
    queryKey: ["projects", projectId, "arch-define", datasetVersionId, pipelineId, choices],
    placeholderData: (prev) => prev,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/arch/define", {
          params: { path: { project_id: projectId } },
          body: { dataset_version_id: datasetVersionId, pipeline_id: pipelineId, choices },
        }),
      ),
  });
}

export function useBuildDefinition(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      dataset_version_id: string;
      pipeline_id: string;
      choices: Record<string, string>;
    }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/arch/define/build", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ) as ArchSpecRecord,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["projects", projectId, "archspecs"] }),
  });
}

// ---------------------------------------------------------------- historia y comparación de runs

async function fetchRunHistory(runId: string) {
  return unwrap(
    await (
      await getApiClient()
    ).GET("/api/v1/runs/{run_id}/history", { params: { path: { run_id: runId } } }),
  );
}

export function useRunHistory(runId: string) {
  return useQuery({ queryKey: ["runs", runId, "history"], queryFn: () => fetchRunHistory(runId) });
}

export type ConfigDiff = Schemas["ConfigDiff"];
export type StudyAnalysis = Schemas["StudyAnalysis"];

/** Visualizaciones de un estudio de HPO (RF-HPO-06). */
export function useStudyAnalysis(studyId: string | null) {
  return useQuery({
    queryKey: ["studies", studyId, "analysis"] as const,
    enabled: Boolean(studyId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/studies/{study_id}/analysis", {
          params: { path: { study_id: studyId ?? "" } },
        }),
      ) as StudyAnalysis,
  });
}

/** Qué cambia entre runs en ArchSpec y pipeline (RF-TRK-04). */
export function useCompareConfigs(runIds: string[]) {
  return useQuery({
    queryKey: ["runs", "compare", "config", ...runIds] as const,
    enabled: runIds.length > 1,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/compare/config", { body: { run_ids: runIds } }),
      ) as ConfigDiff,
  });
}

/** Historias por época de varios runs (comparación, SPEC §11.2). */
export function useRunHistories(runIds: string[]) {
  return useQueries({
    queries: runIds.map((id) => ({
      queryKey: ["runs", id, "history"],
      queryFn: () => fetchRunHistory(id),
    })),
  });
}

// ---------------------------------------------------------------- modo experto (RF-ARC-06)

export function useArchCodeStarter(archspecId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["archspecs", archspecId, "code", "starter"],
    enabled,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/archspecs/{archspec_id}/code/starter", {
          params: { path: { archspec_id: archspecId } },
        }),
      ).code,
  });
}

export function useArchSource(archspecId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["archspecs", archspecId, "code"],
    enabled,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/archspecs/{archspec_id}/code", {
          params: { path: { archspec_id: archspecId } },
        }),
      ).code,
  });
}

export async function lintArchCode(source: string) {
  return unwrap(await (await getApiClient()).POST("/api/v1/arch/code/lint", { body: { source } }));
}

export function useCreateCodeArchSpec(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      base_archspec_id: string;
      source: string;
      name: string | null;
      acknowledge_risk: boolean;
    }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/archspecs/code", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["projects", projectId, "archspecs"] }),
  });
}

// ---------------------------------------------------------------- export y playground (Capa 4a)

export type ExportReport = Schemas["ExportReport"];
export type PlaygroundResult = Schemas["PlaygroundResult"];

export function useExportReport(runId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["runs", runId, "export"],
    enabled,
    retry: false,
    queryFn: async () => {
      const res = await (
        await getApiClient()
      ).GET("/api/v1/runs/{run_id}/export", { params: { path: { run_id: runId } } });
      if (res.response.status === 404) return null;
      return unwrap(res);
    },
  });
}

export function useExportRun(runId: string) {
  return useMutation({
    mutationFn: async (body: Schemas["ExportRequest"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/export", { params: { path: { run_id: runId } }, body }),
      ),
  });
}

export function usePredictRows(runId: string) {
  return useMutation({
    mutationFn: async (rows: Record<string, unknown>[]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/predict", {
          params: { path: { run_id: runId } },
          body: { rows },
        }),
      ),
  });
}

export function usePredictTexts(runId: string) {
  return useMutation({
    mutationFn: async (texts: string[]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/predict/text", {
          params: { path: { run_id: runId } },
          body: { texts },
        }),
      ),
  });
}

export function usePredictFile(runId: string) {
  return useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file, file.name);
      return unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/predict/file", {
          params: { path: { run_id: runId } },
          body: form as unknown as Schemas["Body_predictFile"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      );
    },
  });
}

export function useDatasetSample(datasetVersionId: string | undefined) {
  return useQuery({
    queryKey: ["datasets", datasetVersionId, "sample"],
    enabled: Boolean(datasetVersionId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/datasets/{dataset_version_id}/samples", {
          params: { path: { dataset_version_id: datasetVersionId ?? "" }, query: { limit: 1 } },
        }),
      ),
  });
}

// ---------------------------------------------------------------- evaluación avanzada (Capa 4b)

export type ErrorAnalysis = Schemas["ErrorAnalysis"];
export type FairnessReport = Schemas["FairnessReport"];
export type GlobalExplanation = Schemas["GlobalExplanation"];
export type LocalExplanation = Schemas["LocalExplanation"];
export type RobustnessReport = Schemas["RobustnessReport"];

export function useErrorAnalysis(runId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["runs", runId, "errors"],
    enabled,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/runs/{run_id}/errors", { params: { path: { run_id: runId } } }),
      ),
  });
}

export function useExplanation(runId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["runs", runId, "explain"],
    enabled,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/runs/{run_id}/explain", { params: { path: { run_id: runId } } }),
      ),
  });
}

export function useRobustness(runId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["runs", runId, "robustness"],
    enabled,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/runs/{run_id}/robustness", { params: { path: { run_id: runId } } }),
      ),
  });
}

export function useFairness(runId: string) {
  return useMutation({
    mutationFn: async (body: { attributes: string[]; positive_class: string | null }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/fairness", {
          params: { path: { run_id: runId } },
          body: { ...body, threshold: 0.1 },
        }),
      ),
  });
}

export function useExplainRow(runId: string) {
  return useMutation({
    mutationFn: async (row: Record<string, unknown>) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/explain/row", {
          params: { path: { run_id: runId } },
          body: { row },
        }),
      ),
  });
}

export function useExplainImage(runId: string) {
  return useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file, file.name);
      return unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/explain/image", {
          params: { path: { run_id: runId } },
          body: form as unknown as Schemas["Body_explainImage"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      );
    },
  });
}

export function useExplainText(runId: string) {
  return useMutation({
    mutationFn: async (text: string) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/explain/text", {
          params: { path: { run_id: runId } },
          body: { text },
        }),
      ),
  });
}

export function useExplainAudio(runId: string) {
  return useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file, file.name);
      return unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/runs/{run_id}/explain/audio", {
          params: { path: { run_id: runId } },
          body: form as unknown as Schemas["Body_explainAudio"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      );
    },
  });
}

// ---------------------------------------------------------------- etiquetado (Capa 4c)

export type LabelSet = WithId<Schemas["LabelSet"]> & { classes: string[] };
export type LabelingSummary = Schemas["LabelingSummary"];
export type LabelSample = Schemas["Sample"];
export type LabelUpdate = Schemas["LabelUpdate"];
export type Box = Schemas["Box"];
export type Polygon = Schemas["Polygon"];
export type Segment = Schemas["Segment"];

const lsKey = (id: string) => ["labelsets", id] as const;

export function useLabelSets(datasetVersionId: string | undefined) {
  return useQuery({
    queryKey: ["datasets", datasetVersionId, "labelsets"],
    enabled: Boolean(datasetVersionId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/datasets/{dataset_version_id}/labelsets", {
          params: { path: { dataset_version_id: datasetVersionId ?? "" } },
        }),
      ) as LabelSet[],
  });
}

export function useCreateLabelSet(datasetVersionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["LabelSetCreate"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/datasets/{dataset_version_id}/labelsets", {
          params: { path: { dataset_version_id: datasetVersionId } },
          body,
        }),
      ) as LabelSet,
    onSuccess: () =>
      void qc.invalidateQueries({ queryKey: ["datasets", datasetVersionId, "labelsets"] }),
  });
}

export function useLabelSummary(labelsetId: string | undefined) {
  return useQuery({
    queryKey: [...lsKey(labelsetId ?? ""), "summary"],
    enabled: Boolean(labelsetId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/labelsets/{labelset_id}", {
          params: { path: { labelset_id: labelsetId ?? "" } },
        }),
      ),
  });
}

export function useLabelQueue(
  labelsetId: string | undefined,
  strategy: "uncertainty" | "diversity" | "random",
) {
  return useQuery({
    queryKey: [...lsKey(labelsetId ?? ""), "queue", strategy],
    enabled: Boolean(labelsetId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/labelsets/{labelset_id}/queue", {
          params: { path: { labelset_id: labelsetId ?? "" }, query: { strategy, limit: 50 } },
        }),
      ),
  });
}

function useLabelMutation<T>(labelsetId: string, fn: (id: string, arg: T) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (arg: T) => fn(labelsetId, arg),
    onSuccess: () => void qc.invalidateQueries({ queryKey: lsKey(labelsetId) }),
  });
}

export function useSetLabels(labelsetId: string) {
  return useLabelMutation(labelsetId, async (id, updates: LabelUpdate[]) =>
    unwrap(
      await (
        await getApiClient()
      ).PUT("/api/v1/labelsets/{labelset_id}/labels", {
        params: { path: { labelset_id: id } },
        body: { updates },
      }),
    ),
  );
}

export function useAcceptSuggestions(labelsetId: string) {
  return useLabelMutation(labelsetId, async (id, min_confidence: number) =>
    unwrap(
      await (
        await getApiClient()
      ).POST("/api/v1/labelsets/{labelset_id}/accept", {
        params: { path: { labelset_id: id } },
        body: { min_confidence },
      }),
    ),
  );
}

export function usePrelabelWithModel(labelsetId: string) {
  return useLabelMutation(labelsetId, async (id, run_id: string) =>
    unwrap(
      await (
        await getApiClient()
      ).POST("/api/v1/labelsets/{labelset_id}/prelabel", {
        params: { path: { labelset_id: id } },
        body: { method: "model", run_id, limit: 2000 },
      }),
    ),
  );
}

/** Zero-shot local con los nombres de clase (RF-LBL-02): no necesita un modelo entrenado. */
export function usePrelabelZeroShot(labelsetId: string) {
  return useLabelMutation(labelsetId, async (id, limit: number) =>
    unwrap(
      await (
        await getApiClient()
      ).POST("/api/v1/labelsets/{labelset_id}/prelabel", {
        params: { path: { labelset_id: id } },
        body: { method: "zero_shot", limit },
      }),
    ),
  );
}

export function useApplyLabels(labelsetId: string, projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/labelsets/{labelset_id}/apply", {
          params: { path: { labelset_id: labelsetId } },
        }),
      ) as DatasetVersion,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.datasets(projectId) });
      void qc.invalidateQueries({ queryKey: lsKey(labelsetId) });
    },
  });
}

// ---------------------------------------------------------------- fuentes remotas (Capa 4c)

async function previewOf(source: DataSource) {
  const api = await getApiClient();
  const preview = unwrap(
    await api.POST("/api/v1/sources/{source_id}/preview", {
      params: { path: { source_id: source.id } },
    }),
  );
  return { source, preview };
}

export function useCreateDbSource(projectId: string) {
  return useMutation({
    mutationFn: async (body: Schemas["DbSourceCreate"]) =>
      previewOf(
        unwrap(
          await (
            await getApiClient()
          ).POST("/api/v1/projects/{project_id}/sources/db", {
            params: { path: { project_id: projectId } },
            body,
          }),
        ) as DataSource,
      ),
  });
}

export function useCreateHubSource(projectId: string) {
  return useMutation({
    mutationFn: async (body: Schemas["HubSourceCreate"]) =>
      previewOf(
        unwrap(
          await (
            await getApiClient()
          ).POST("/api/v1/projects/{project_id}/sources/hub", {
            params: { path: { project_id: projectId } },
            body,
          }),
        ) as DataSource,
      ),
  });
}

// ---------------------------------------------------------------- fuentes del servidor (Capa 5a)

export type ServerSourceRoot = Schemas["ServerSourceRoot"];
export type ServerListing = Schemas["ServerListing"];

/** Carpetas montadas en el Team Server que el Admin habilitó (RF-SRV-05). */
export function useServerSources(enabled: boolean) {
  return useQuery({
    queryKey: ["server", "sources"],
    enabled,
    retry: false,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/server/sources")),
  });
}

export function useBrowseServerSource(index: number | null, path: string) {
  return useQuery({
    queryKey: ["server", "sources", index, path],
    enabled: index !== null,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/server/sources/{index}/browse", {
          params: { path: { index: index ?? 0 }, query: { path } },
        }),
      ),
  });
}

export function useCreatePathSource(projectId: string) {
  return useMutation({
    mutationFn: async ({ path, kind }: { path: string; kind: "dir" | "file" }) =>
      previewOf(
        unwrap(
          await (
            await getApiClient()
          ).POST("/api/v1/projects/{project_id}/sources", {
            params: { path: { project_id: projectId } },
            body: { path, type: kind === "dir" ? "folder" : "file" },
          }),
        ) as DataSource,
      ),
  });
}

// ---------------------------------------------------------------- cola del Team Server (Capa 5b)

export type QueueView = Schemas["QueueView"];

/** Estudios en cola/en curso y workers con su hardware (RF-SRV-04). */
export function useQueue(enabled: boolean) {
  return useQuery({
    queryKey: ["server", "queue"],
    enabled,
    refetchInterval: 5000,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/server/queue")),
  });
}

// ---------------------------------------------------------------- servidores de equipo (Capa 5c)

export type RemoteServer = Schemas["RemoteServer"];

/** Team Servers conectados desde este desktop (RF-SRV-03). */
export function useRemoteServers(enabled = true) {
  return useQuery({
    queryKey: ["remote", "servers"],
    enabled,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/remote/servers")),
  });
}

export function useConnectRemote() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["ServerConnect"]) =>
      unwrap(await (await getApiClient()).POST("/api/v1/remote/servers", { body })),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["remote", "servers"] }),
  });
}

export function useRemoveRemote() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (name: string) => {
      const res = await (
        await getApiClient()
      ).DELETE("/api/v1/remote/servers/{name}", { params: { path: { name } } });
      if (!res.response.ok) unwrap(res);
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["remote", "servers"] }),
  });
}

/** Entrena en un worker del servidor con el progreso en vivo en el desktop. */
export function useCreateRemoteStudy(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["RemoteStudyCreate"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/remote/studies", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.runs(projectId) }),
  });
}

// ---------------------------------------------------------------- fórmula sugerida (ADR-0039)

export function useSymbolicFits(projectId: string) {
  return useQuery({
    queryKey: keys.symbolic(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/symbolic", {
          params: { path: { project_id: projectId } },
        }),
      ) as SymbolicFit[],
  });
}

export function useStartSymbolic(projectId: string) {
  return useMutation({
    mutationFn: async (body: Schemas["SymbolicRequest"]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/symbolic", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
  });
}

export function useSymbolicPredict(fitId: string) {
  return useMutation({
    mutationFn: async (rows: Record<string, number | null>[]) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/symbolic/{symbolic_fit_id}/predict", {
          params: { path: { symbolic_fit_id: fitId } },
          body: { rows },
        }),
      ).predictions,
  });
}

// ---------------------------------------------------------------- control de estudios

export type StudyView = Schemas["StudyView"];

export function useStudies(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "studies"],
    // Con estudios activos se refresca solo (pasan a terminados o interrumpidos).
    refetchInterval: (q) =>
      (q.state.data ?? []).some((v) => v.status === "running" || v.status === "queued")
        ? 4000
        : false,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/studies", {
          params: { path: { project_id: projectId } },
        }),
      ) as StudyView[],
  });
}

export function useStudyAction(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ studyId, action }: { studyId: string; action: "pause" | "resume" }) => {
      const api = await getApiClient();
      const params = { params: { path: { study_id: studyId } } };
      return action === "pause"
        ? unwrap(await api.POST("/api/v1/studies/{study_id}/pause", params))
        : unwrap(await api.POST("/api/v1/studies/{study_id}/resume", params));
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["projects", projectId, "studies"] });
      void qc.invalidateQueries({ queryKey: keys.runs(projectId) });
    },
  });
}
