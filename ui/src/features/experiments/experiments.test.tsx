import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

import { bestRunIds } from "./best";

const run = (n: number, lr: number) => ({
  id: `run-t00${n}`,
  project_id: "prj_1",
  dataset_version_id: "dsv_1",
  pipeline_id: "pip_1",
  archspec_id: "arc_1",
  device: "cpu",
  seed: 1,
  status: "succeeded",
  metrics: { val_loss: 0.5 - n / 10, val_accuracy: 0.8 + n / 100 },
  hyperparams: { lr, dropout: 0.1 },
  version: 1,
});

describe("comparación de runs", () => {
  beforeEach(() => resetApiClient());

  it("marca dos runs y muestra métricas e hiperparámetros lado a lado", async () => {
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/runs": () => [run(1, 0.001), run(2, 0.01)],
      "GET /api/v1/runs/run-t001/history": () => [{ epoch: 0, val_loss: 0.7 }],
      "GET /api/v1/runs/run-t002/history": () => [{ epoch: 0, val_loss: 0.6 }],
      "POST /api/v1/runs/compare/config": () => ({
        run_ids: ["run-t001", "run-t002"],
        same_archspec: false,
        same_pipeline: true,
        archspec: [{ path: "nodes[enc].params.hidden", values: [64, 128] }],
        pipeline: [],
      }),
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter("/projects/prj_1/experiments")} />
      </Providers>,
    );
    await userEvent.click(await screen.findByLabelText("Comparar t001"));
    expect(screen.getByText("Marcá dos o más runs para compararlos.")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Comparar t002"));
    const card = (await screen.findByText("Comparación de 2 runs")).closest("div")?.parentElement;
    if (!card) throw new Error("sin tarjeta");
    const lrRow = within(card).getByText("lr").closest("tr");
    expect(lrRow).toHaveClass("bg-canvas"); // difiere entre runs
    expect(within(card).getByText("dropout").closest("tr")).not.toHaveClass("bg-canvas");
    // RF-TRK-04: diff de la arquitectura (el pipeline es igual y no se muestra).
    expect(await within(card).findByText("Diferencias de arquitectura")).toBeInTheDocument();
    const hidden = within(card).getByText("nodes[enc].params.hidden").closest("tr");
    expect(hidden).toHaveTextContent("64");
    expect(hidden).toHaveTextContent("128");
    expect(within(card).queryByText("Diferencias de pipeline")).not.toBeInTheDocument();
    await userEvent.click(within(card).getByRole("button", { name: "Quitar selección" }));
    expect(screen.queryByText("Comparación de 2 runs")).not.toBeInTheDocument();
  });
});

describe("análisis del estudio (RF-HPO-06)", () => {
  beforeEach(() => resetApiClient());

  it("muestra historia, importancia, coordenadas y Pareto del estudio", async () => {
    const inStudy = (n: number, lr: number) => ({ ...run(n, lr), study_id: "stu_1" });
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/runs": () => [inStudy(1, 0.001), inStudy(2, 0.01)],
      "GET /api/v1/studies/stu_1/analysis": () => ({
        objectives: [
          { metric: "val_loss", direction: "minimize" },
          { metric: "val_accuracy", direction: "maximize" },
        ],
        params: ["dropout", "lr"],
        importance: { lr: 0.8, dropout: 0.2 },
        trials: [
          {
            number: 0,
            run_id: "run-t001",
            status: "succeeded",
            params: { lr: 0.001, dropout: 0.1 },
            values: [0.4, 0.81],
            best_so_far: 0.4,
            pareto: true,
          },
          {
            number: 1,
            run_id: "run-t002",
            status: "succeeded",
            params: { lr: 0.01, dropout: 0.1 },
            values: [0.3, 0.82],
            best_so_far: 0.3,
            pareto: true,
          },
        ],
      }),
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter("/projects/prj_1/experiments")} />
      </Providers>,
    );
    const card = (await screen.findByText("Análisis del estudio")).closest("div")?.parentElement;
    if (!card) throw new Error("sin tarjeta");
    for (const title of [
      "Historia de la optimización",
      "Importancia de hiperparámetros",
      "Coordenadas paralelas",
      "Frente de Pareto",
    ]) {
      expect(await within(card).findByRole("heading", { name: title })).toBeInTheDocument();
    }
  });
});

describe("mejor run por estudio", () => {
  it("marca el de menor val_loss de cada estudio, solo entre los terminados", () => {
    const r = (id: string, study: string, loss: number | null, status = "succeeded") =>
      ({ ...run(1, 0.001), id, study_id: study, status, metrics: { val_loss: loss } }) as never;
    const best = bestRunIds([
      r("a-t000", "a", 0.2),
      r("a-t001", "a", 0.05), // el mejor de «a»
      r("a-t002", "a", 0.01, "running"), // sin terminar: no cuenta
      r("b-t000", "b", 0.9),
      r("b-t001", "b", null),
    ]);
    expect([...best].sort()).toEqual(["a-t001", "b-t000"]);
  });
});

describe("fórmula sugerida (ADR-0039)", () => {
  beforeEach(() => resetApiClient());

  it("muestra la fórmula, la compara con la mejor red, la calcula y busca otra", async () => {
    const dataset = {
      id: "dsv_1",
      project_id: "prj_1",
      content_hash: "abcdef0123456789",
      num_samples: 397,
      size_bytes: 1,
      modality: "tabular",
      target: "multiplo",
      version: 1,
      created_at: "2026-10-04T10:00:00Z",
    };
    const fit = {
      id: "sym_1",
      project_id: "prj_1",
      dataset_version_id: "dsv_1",
      target: "multiplo",
      features: ["numero"],
      expression: "3*x1",
      formula: "3·numero",
      python: "import math\n\n\ndef formula(numero):\n    return 3*numero\n",
      excel_es: "=(3*A2)",
      excel_en: "=(3*A2)",
      candidates: [{ length: 3, val_rmse: 0, formula: "3·numero" }],
      metrics: { val: { mae: 0, r2: 1 }, test: { mae: 0, r2: 1 } },
      parity: [
        [3, 3],
        [600, 600],
      ],
      warnings: [],
      n_train: 277,
      duration_s: 9,
      time_limit_s: 60,
      version: 1,
    };
    const best = {
      ...run(1, 0.001),
      dataset_version_id: "dsv_1",
      metrics: { val_loss: 0.01, val_mae: 3.2, val_r2: 0.9998 },
    };
    const { calls } = mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/runs": () => [best],
      "GET /api/v1/projects/prj_1/datasets": () => [dataset],
      "GET /api/v1/projects/prj_1/symbolic": () => [fit],
      "POST /api/v1/symbolic/sym_1/predict": () => ({ predictions: [1500] }),
      "POST /api/v1/projects/prj_1/symbolic": () => ({
        job: { id: "job_s", kind: "symbolic", status: "queued" },
      }),
      "GET /api/v1/jobs/job_s": () => ({ id: "job_s", kind: "symbolic", status: "running" }),
    });
    const user = userEvent.setup();
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter("/projects/prj_1/experiments")} />
      </Providers>,
    );
    expect(await screen.findByLabelText("Fórmula")).toHaveTextContent("multiplo = 3·numero");
    expect(screen.getByText("Mejor red (t001)")).toBeInTheDocument();
    await user.type(screen.getByLabelText("numero"), "500");
    await user.click(screen.getByRole("button", { name: "Calcular" }));
    expect(await screen.findByRole("status")).toHaveTextContent("multiplo = 1500");
    const predict = calls.find((r) => r.url.endsWith("/symbolic/sym_1/predict"));
    expect(await predict?.clone().json()).toEqual({ rows: [{ numero: 500 }] });
    await user.click(screen.getByRole("button", { name: "Buscar fórmula" }));
    const started = calls.find(
      (r) => r.method === "POST" && r.url.endsWith("/projects/prj_1/symbolic"),
    );
    expect(await started?.clone().json()).toEqual({
      dataset_version_id: "dsv_1",
      time_limit_s: 60,
    });
  });
});

describe("control de estudios", () => {
  beforeEach(() => resetApiClient());

  it("muestra el estado, detiene el que entrena y reanuda el interrumpido", async () => {
    const view = (id: string, status: string, extra: Record<string, unknown> = {}) => ({
      study: { id, project_id: "prj_1", name: `hpo-${id}`, version: 1 },
      status,
      job_id: status === "running" ? "job_1" : null,
      trials_done: 5,
      trials_total: 20,
      best_run_id: `${id}-t003`,
      reason: null,
      resumable: false,
      ...extra,
    });
    const { calls } = mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/runs": () => [],
      "GET /api/v1/projects/prj_1/datasets": () => [],
      "GET /api/v1/projects/prj_1/symbolic": () => [],
      "GET /api/v1/projects/prj_1/studies": () => [
        view("std_a", "running"),
        view("std_b", "interrupted", {
          resumable: true,
          reason: "El worker se reinició durante el entrenamiento.",
        }),
      ],
      "POST /api/v1/studies/std_a/pause": () => ({
        id: "job_1",
        kind: "study",
        status: "cancelled",
      }),
      "POST /api/v1/studies/std_b/resume": () => ({
        study: { id: "std_b" },
        job: { id: "job_2", kind: "study", status: "queued" },
      }),
    });
    const user = userEvent.setup();
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter("/projects/prj_1/experiments")} />
      </Providers>,
    );
    expect(await screen.findByText("Interrumpido")).toBeInTheDocument();
    expect(screen.getByText("El worker se reinició durante el entrenamiento.")).toBeInTheDocument();
    expect(screen.getAllByText("5 de 20")).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "Detener" }));
    await user.click(screen.getByRole("button", { name: "Reanudar" }));
    await waitFor(() =>
      expect(calls.map((r) => `${r.method} ${new URL(r.url).pathname}`)).toEqual(
        expect.arrayContaining([
          "POST /api/v1/studies/std_a/pause",
          "POST /api/v1/studies/std_b/resume",
        ]),
      ),
    );
  });
});
