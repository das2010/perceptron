import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { PipelineSuggestions } from "./PipelineSuggestions";

const step = (params: Record<string, unknown>) => ({
  id: "impute",
  kind: "impute_numeric",
  columns: ["edad"],
  params,
});

describe("sugerencias de pipeline (RF-PIP-05)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("muestra el diff, y aplica solo las aceptadas", async () => {
    let applied: { version: number; changes: { op: string }[] } | null = null;
    const change = (op: string, rationale: string) => ({
      op,
      step_id: "impute",
      step: op === "remove" ? null : step({ strategy: "mean" }),
      after: null,
      rationale,
    });
    mockEngine({
      "GET /api/v1/projects/prj_1/datasets": () => [
        { id: "dsv_1", content_hash: "a".repeat(64), num_samples: 400, project_id: "prj_1" },
      ],
      "POST /api/v1/pipelines/pip_1/suggestions": () => ({
        pipeline_id: "pip_1",
        pipeline_version: 3,
        origin: "llm",
        llm_call_id: "llm_1",
        items: [
          {
            index: 0,
            change: change("update", "La edad tiene outliers: la media es mejor aquí."),
            before: step({ strategy: "median" }),
            after: step({ strategy: "mean" }),
          },
          { index: 1, change: change("remove", "Sin faltantes."), before: step({}), after: null },
        ],
      }),
      "POST /api/v1/pipelines/pip_1/suggestions/apply": async (req) => {
        applied = (await req.json()) as typeof applied;
        return { id: "pip_1", version: 4, project_id: "prj_1", name: "p", graph: {} };
      },
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <PipelineSuggestions projectId="prj_1" pipelineId="pip_1" />
      </Providers>,
    );
    await userEvent.click(await screen.findByRole("button", { name: "Pedir sugerencias" }));
    expect(await screen.findByText("Cambiar el paso impute")).toBeInTheDocument();
    expect(screen.getByText(/"strategy": "median"/)).toBeInTheDocument();
    expect(screen.getAllByText(/"strategy": "mean"/).length).toBeGreaterThan(0);
    const [acceptUpdate] = screen.getAllByRole("button", { name: "Aceptar" });
    await userEvent.click(acceptUpdate!);
    await userEvent.click(screen.getAllByRole("button", { name: "Descartar" })[1]!);
    await userEvent.click(screen.getByRole("button", { name: "Aplicar 1 sugerencia aceptada" }));
    await waitFor(() => expect(applied).not.toBeNull());
    expect(applied!.version).toBe(3);
    expect(applied!.changes.map((c) => c.op)).toEqual(["update"]);
  });
});
