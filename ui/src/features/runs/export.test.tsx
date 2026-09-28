import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

const RUN = "std_1-t000";
const run = {
  id: RUN,
  project_id: "prj_1",
  study_id: "std_1",
  archspec_id: "arc_1",
  pipeline_id: "pip_1",
  dataset_version_id: "dsv_1",
  status: "succeeded",
  device: "cpu",
  hyperparams: {},
  metrics: { val_loss: 0.4 },
  seed: 42,
  version: 2,
};
const artifact = (format: string, extra: Record<string, unknown> = {}) => ({
  format,
  file: `model.${format}`,
  size_bytes: 2048,
  sha256: "abc",
  legacy: false,
  verification: { samples: 16, max_abs_diff: 3e-6, tolerance: 1e-4, passed: true },
  error: null,
  ...extra,
});
const report = {
  run_id: RUN,
  created_at: "2026-09-28T00:00:00Z",
  signature: {
    inputs: { kind: "tabular", columns: ["edad", "plan"] },
    outputs: { task: "classification" },
  },
  inputs: [],
  outputs: [],
  artifacts: [
    artifact("onnx"),
    artifact("torchscript", { legacy: true }),
    artifact("onnx_int8", {
      verification: { samples: 16, max_abs_diff: 0.02, tolerance: null, passed: true },
    }),
  ],
};

describe("export y playground (Capa 4a)", () => {
  beforeEach(() => resetApiClient());

  it("exporta, muestra la verificación y predice en el playground", async () => {
    let exported = false;
    const predicted: unknown[] = [];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      [`GET /api/v1/runs/${RUN}`]: () => run,
      [`GET /api/v1/runs/${RUN}/history`]: () => [],
      [`GET /api/v1/runs/${RUN}/export`]: () =>
        exported
          ? report
          : new Response(JSON.stringify({ code: "not_found", message: "no" }), { status: 404 }),
      [`POST /api/v1/runs/${RUN}/export`]: () => {
        exported = true;
        return new Response(
          JSON.stringify({ job: { id: "job_1", kind: "export", status: "queued" } }),
          {
            status: 202,
            headers: { "Content-Type": "application/json" },
          },
        );
      },
      "GET /api/v1/jobs/job_1": () => ({ id: "job_1", kind: "export", status: "succeeded" }),
      "GET /api/v1/datasets/dsv_1/samples": () => [{ edad: 41, plan: "pro", churn: "1" }],
      [`POST /api/v1/runs/${RUN}/predict`]: async (req) => {
        predicted.push(await req.json());
        return {
          task: "classification",
          input_kind: "tabular",
          predictions: [
            { prediction: "1", confidence: 0.83, probabilities: { "0": 0.17, "1": 0.83 } },
          ],
        };
      },
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter(`/projects/prj_1/runs/${RUN}`)} />
      </Providers>,
    );
    await userEvent.click(await screen.findByLabelText("ONNX INT8 (CPU)"));
    await userEvent.click(screen.getByRole("button", { name: /^Exportar$/ }));
    const table = await screen.findByRole("table", {}, { timeout: 5000 });
    expect(within(table).getByText("legacy")).toBeInTheDocument();
    expect(within(table).getAllByText(/dif\. 3\.0e-6 \(tol\. 1e-4\)/)).toHaveLength(2);
    expect(within(table).getByText(/tol\. —/)).toBeInTheDocument(); // INT8 solo se informa

    // Playground: completar con un ejemplo y predecir.
    await userEvent.click(await screen.findByRole("button", { name: "Completar con un ejemplo" }));
    expect(screen.getByLabelText("edad")).toHaveValue("41");
    await userEvent.click(screen.getByRole("button", { name: "Predecir" }));
    await waitFor(() => expect(predicted).toHaveLength(1));
    expect(predicted[0]).toEqual({ rows: [{ edad: "41", plan: "pro" }] });
    expect(await screen.findByText(/confianza 83/)).toBeInTheDocument();
  }, 20_000);
});
