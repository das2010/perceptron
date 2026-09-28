/**
 * Hooks de datos por recurso (TanStack Query) sobre el cliente tipado de OpenAPI.
 * Ninguna lógica de ML vive en la UI (CLAUDE.md): solo llamadas al Engine.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getApiClient } from "./client";
import type { components } from "./schema";

export type Schemas = components["schemas"];
/** Las entidades siempre traen `id` en las respuestas (en el schema figura con default). */
type WithId<T> = T & { id: string };

export type Project = WithId<Schemas["Project"]>;
export type DatasetVersion = WithId<Schemas["DatasetVersion"]>;
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
    const body = (error ?? {}) as { code?: string; message?: string; details?: Record<string, unknown> };
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
};

// ---------------------------------------------------------------- proyectos

export function useProjects() {
  return useQuery({
    queryKey: keys.projects,
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/projects")) as Project[],
  });
}

export function useProject(projectId: string) {
  return useQuery({
    queryKey: keys.project(projectId),
    queryFn: async () =>
      unwrap(
        await (await getApiClient()).GET("/api/v1/projects/{project_id}", {
          params: { path: { project_id: projectId } },
        }),
      ) as Project,
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

// ---------------------------------------------------------------- datos

export function useDatasets(projectId: string) {
  return useQuery({
    queryKey: keys.datasets(projectId),
    queryFn: async () =>
      unwrap(
        await (await getApiClient()).GET("/api/v1/projects/{project_id}/datasets", {
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
export function useUpload(projectId: string) {
  return useMutation({
    mutationFn: async (files: File[]) => {
      const api = await getApiClient();
      const form = new FormData();
      for (const f of files) form.append("files", f, f.webkitRelativePath || f.name);
      const source = unwrap(
        await api.POST("/api/v1/projects/{project_id}/uploads", {
          params: { path: { project_id: projectId } },
          // openapi-fetch serializa JSON por defecto: el FormData va tal cual.
          body: form as unknown as Schemas["Body_uploadSource"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      ) as DataSource;
      const preview = unwrap(
        await api.POST("/api/v1/sources/{source_id}/preview", {
          params: { path: { source_id: source.id } },
        }),
      );
      return { source, preview };
    },
  });
}

export function useIngest(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ sourceId, target }: { sourceId: string; target?: string | null }) =>
      unwrap(
        await (await getApiClient()).POST("/api/v1/sources/{source_id}/ingest", {
          params: { path: { source_id: sourceId } },
          body: { target: target || null },
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
        await (await getApiClient()).POST("/api/v1/projects/{project_id}/pipelines/propose", {
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
        await (await getApiClient()).POST("/api/v1/projects/{project_id}/arch/propose", {
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
        await (await getApiClient()).POST("/api/v1/projects/{project_id}/hpo/strategy", {
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
        await (await getApiClient()).POST("/api/v1/projects/{project_id}/studies", {
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
        await (await getApiClient()).GET("/api/v1/jobs/{job_id}", {
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
        await (await getApiClient()).GET("/api/v1/projects/{project_id}/runs", {
          params: { path: { project_id: projectId } },
        }),
      ) as Run[],
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: keys.run(runId),
    queryFn: async () =>
      unwrap(
        await (await getApiClient()).GET("/api/v1/runs/{run_id}", {
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
        await (await getApiClient()).GET("/api/v1/runs/{run_id}/evaluation", {
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
        await (await getApiClient()).POST("/api/v1/runs/{run_id}/evaluate", {
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
        await (await getApiClient()).POST("/api/v1/runs/{run_id}/register", {
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
        await (await getApiClient()).GET("/api/v1/runs/{run_id}/diagnosis", {
          params: { path: { run_id: runId } },
        }),
      ),
  });
}

export function useReport(runId: string) {
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (await getApiClient()).POST("/api/v1/runs/{run_id}/report", {
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
        await (await getApiClient()).GET("/api/v1/projects/{project_id}/models", {
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
        await (await getApiClient()).PUT("/api/v1/llm/providers/{name}", {
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
        await (await getApiClient()).PUT("/api/v1/llm/profiles", { body: { active, profiles: {} } }),
      ),
    onSuccess: (data) => qc.setQueryData(keys.llmProfiles, data),
  });
}

export function useTestLlm() {
  return useMutation({
    mutationFn: async (projectId?: string) =>
      unwrap(
        await (await getApiClient()).POST("/api/v1/llm/test", {
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
        await (await getApiClient()).GET("/api/v1/llm/audit", {
          params: { query: { project: projectId, limit: 200 } },
        }),
      ) as LLMCall[],
  });
}
