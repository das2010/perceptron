import { QueryClient } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

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
    await userEvent.click(within(card).getByRole("button", { name: "Quitar selección" }));
    expect(screen.queryByText("Comparación de 2 runs")).not.toBeInTheDocument();
  });
});
