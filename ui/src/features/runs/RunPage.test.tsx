import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

const RUN = "std_1-t000";

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
});
