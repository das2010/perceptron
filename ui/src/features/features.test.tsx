import { QueryClient } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App router={createTestRouter(path)} />
    </Providers>,
  );
}

const dataset = {
  id: "dsv_1",
  project_id: "prj_1",
  content_hash: "abcdef0123456789",
  num_samples: 400,
  size_bytes: 1000,
  modality: "tabular",
  target: "churn",
  version: 1,
  created_at: "2026-09-27T10:00:00Z",
};

const card = {
  card_version: "1",
  dataset_version_id: "dsv_1",
  content_hash: "abcdef",
  modality: "tabular",
  num_samples: 400,
  profiled_samples: 400,
  split_counts: { train: 280, val: 60, test: 60 },
  num_features: 2,
  target: {
    name: "churn",
    semantic: "categorical",
    task_hint: "classification",
    classes: [
      { value: "no", count: 300 },
      { value: "si", count: 100 },
    ],
    imbalance_ratio: 0.33,
  },
  columns: [
    {
      name: "edad",
      dtype: "Int64",
      semantic: "numeric",
      null_fraction: 0.05,
      n_unique: 50,
      numeric: { mean: 41.2, std: 9.1, quantiles: {}, histogram: [], outlier_fraction: 0 },
    },
  ],
  alerts: [{ code: "class_imbalance", severity: "warning", message: "Clases desbalanceadas" }],
};

const base = {
  "GET /api/v1/projects": () => [project()],
  "GET /api/v1/projects/prj_1": () => project(),
  "GET /api/v1/projects/prj_1/datasets": () => [dataset],
};

describe("pantallas", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("Datos: muestra el perfil con alertas y columnas", async () => {
    mockEngine({ ...base, "GET /api/v1/datasets/dsv_1/profile": () => card });
    renderAt("/projects/prj_1/data");
    expect(await screen.findByText("Perfil del dataset")).toBeInTheDocument();
    expect(screen.getByText("Clases desbalanceadas")).toBeInTheDocument();
    expect(screen.getByText("Atención")).toBeInTheDocument();
    expect(screen.getByText("edad")).toBeInTheDocument();
    expect(screen.getByText("5 %")).toBeInTheDocument();
  });

  it("Entrenar: las propuestas del LLM se marcan como IA y se aceptan", async () => {
    const proposal = (id: string, title: string) => ({
      archspec: {
        id,
        project_id: "prj_1",
        name: title,
        origin: "llm",
        content_hash: id,
        version: 1,
      },
      title,
      rationale: `Porque ${title}`,
      validation: { valid: true, issues: [] },
      estimates: { num_params: 12000, memory_mb: 3, epoch_time_s: 0.4 },
    });
    mockEngine({
      ...base,
      "POST /api/v1/projects/prj_1/pipelines/propose": () => ({
        id: "pip_1",
        project_id: "prj_1",
        name: "p",
        graph: { rationale: ["Se estandarizan las numéricas."] },
        version: 1,
      }),
      "POST /api/v1/projects/prj_1/arch/propose": () => ({
        origin: "llm",
        llm_call_id: "llc_1",
        proposals: [proposal("arc_a", "MLP chico"), proposal("arc_b", "MLP ancho")],
      }),
    });
    renderAt("/projects/prj_1/train");
    await userEvent.click(await screen.findByRole("button", { name: "Proponer preparación" }));
    expect(await screen.findByText("Se estandarizan las numéricas.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Proponer arquitecturas" }));
    const card = (await screen.findByText("MLP ancho")).closest("div.rounded-pt") as HTMLElement;
    expect(within(card).getByText("Sugerido por IA")).toBeInTheDocument();
    expect(screen.queryByText("Presupuesto y búsqueda de hiperparámetros")).not.toBeInTheDocument();
    await userEvent.click(within(card).getByRole("button", { name: "Aceptar" }));
    expect(
      await screen.findByText("Presupuesto y búsqueda de hiperparámetros"),
    ).toBeInTheDocument();
  });

  it("Configuración: la clave se envía una vez y no se muestra", async () => {
    let sent: Record<string, unknown> = {};
    const providers = [
      {
        name: "openai",
        kind: "openai",
        base_url: null,
        api_key_ref: "OPENAI_API_KEY",
        local: false,
        has_key: false,
        models: { "gpt-5": {} },
      },
    ];
    mockEngine({
      "GET /api/v1/projects": () => [],
      "GET /api/v1/llm/providers": () => providers,
      "GET /api/v1/llm/profiles": () => ({
        active: "openai",
        profiles: { openai: {}, ollama: {} },
      }),
      "PUT /api/v1/llm/providers/openai": async (req) => {
        sent = (await req.json()) as Record<string, unknown>;
        return [{ ...providers[0], has_key: true }];
      },
    });
    renderAt("/settings");
    const input = await screen.findByLabelText("Clave de openai");
    await userEvent.type(input, "sk-secreto");
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    expect(await screen.findByText("Clave guardada")).toBeInTheDocument();
    expect(sent).toMatchObject({ kind: "openai", api_key: "sk-secreto" });
    expect((input as HTMLInputElement).value).toBe("");
    expect(screen.queryByText("sk-secreto")).not.toBeInTheDocument();
  });
});
