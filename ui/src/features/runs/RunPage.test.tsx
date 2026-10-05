import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
  hyperparams: { lr: 0.001 },
  metrics: { val_loss: 0.4, val_accuracy: 0.8 },
  seed: 42,
  version: 2,
};

describe("RunPage", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("un run terminado ofrece evaluar en test", async () => {
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      [`GET /api/v1/runs/${RUN}`]: () => ({
        id: RUN,
        project_id: "prj_1",
        study_id: "std_1",
        archspec_id: "arc_1",
        pipeline_id: "pip_1",
        dataset_version_id: "dsv_1",
        status: "succeeded",
        device: "cpu",
        hyperparams: { lr: 0.001 },
        metrics: { val_loss: 0.4, val_accuracy: 0.8 },
        seed: 42,
        version: 2,
      }),
      [`GET /api/v1/runs/${RUN}/history`]: () => [
        {
          epoch: 0,
          train_loss: 0.7,
          val_loss: 0.6,
          lr: 0.001,
          epoch_time_s: 0.1,
          samples_per_s: 100,
        },
        {
          epoch: 1,
          train_loss: 0.5,
          val_loss: 0.4,
          lr: 0.001,
          epoch_time_s: 0.1,
          samples_per_s: 100,
        },
      ],
      [`GET /api/v1/runs/${RUN}/diagnosis`]: () => ({
        summary: "No se detectaron problemas.",
        problems: [],
        actions: [],
        origin: "rules",
      }),
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <Providers client={client}>
        <App router={createTestRouter(`/projects/prj_1/runs/${RUN}`)} />
      </Providers>,
    );
    expect(await screen.findByRole("button", { name: "Evaluar en test" })).toBeInTheDocument();
    expect(await screen.findByText("No se detectaron problemas.")).toBeInTheDocument();
  });

  it("el diagnóstico ofrece mejoras concretas: aplicar y entrenar (ADR-0041)", async () => {
    const bodies: unknown[] = [];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      [`GET /api/v1/runs/${RUN}`]: () => run,
      [`GET /api/v1/runs/${RUN}/history`]: () => [],
      [`GET /api/v1/runs/${RUN}/diagnosis`]: () => ({
        summary: "Sobreajusta.",
        problems: [],
        actions: [
          { kind: "add_augmentation", rationale: "Más variedad en train." },
          { kind: "add_regularization", rationale: "Más regularización." },
        ],
        origin: "llm",
      }),
      [`GET /api/v1/runs/${RUN}/improvements`]: () => [
        {
          index: 0,
          kind: "add_augmentation",
          rationale: "Más variedad en train.",
          applicable: false,
          hint: "pipeline",
        },
        {
          index: 1,
          kind: "add_regularization",
          rationale: "Más regularización.",
          applicable: true,
          change: "dropout: 0.2 → 0.3",
        },
      ],
      [`POST /api/v1/runs/${RUN}/improvements/1`]: () => ({
        archspec: {
          id: "arc_2",
          project_id: "prj_1",
          name: "mlp-mejora",
          origin: "llm",
          content_hash: "h",
          version: 1,
        },
        change: "dropout: 0.2 → 0.3",
        budget: { max_trials: 4, max_epochs_per_trial: 12 },
      }),
      "POST /api/v1/projects/prj_1/studies": async (req) => {
        bodies.push(await req.json());
        return { study: { id: "std_2", project_id: "prj_1", name: "s" }, job: { id: "job_9" } };
      },
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <Providers client={client}>
        <App router={createTestRouter(`/projects/prj_1/runs/${RUN}`)} />
      </Providers>,
    );
    expect(await screen.findByText("(dropout: 0.2 → 0.3)")).toBeInTheDocument();
    expect(screen.getByText(/Se hace en la preparación/)).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Aplicar" }));
    await user.click(await screen.findByRole("button", { name: /Entrenar con esta mejora/ }));
    await vi.waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toMatchObject({
      archspec_id: "arc_2",
      pipeline_id: "pip_1",
      dataset_version_id: "dsv_1",
      budget: { max_trials: 4, max_epochs_per_trial: 12 },
    });
  });
});
