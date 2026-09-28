import { QueryClient } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

const report = {
  id: "drf_1",
  deployment_id: "dep_1",
  window_start: "2026-09-28T10:00:00Z",
  window_end: "2026-09-28T11:00:00Z",
  severity: "high",
  metrics: {
    data: {
      n_current: 100,
      features: [
        {
          feature: "plan",
          kind: "categorical",
          severity: "none",
          psi: 0.01,
          js: 0.02,
          chi2_pvalue: 0.8,
        },
        {
          feature: "antiguedad_meses",
          kind: "numeric",
          severity: "high",
          psi: 1.9,
          js: 0.6,
          ks_pvalue: 0,
          reference_mean: 36.2,
          current_mean: 3.4,
        },
      ],
    },
    embedding: {
      severity: "medium",
      domain_auc: 0.74,
      mmd_pvalue: 0.02,
      centroid_distance: 1.3,
      n_current: 100,
    },
    performance: {
      metric: "roc_auc",
      current: 0.71,
      baseline: 0.83,
      n_labeled: 90,
      severity: "high",
    },
  },
  version: 1,
};

describe("monitoreo (Capa 6a)", () => {
  beforeEach(() => resetApiClient());

  it("muestra el drift por feature, la performance y resuelve alertas", async () => {
    const actions: string[] = [];
    let status = "open";
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/deployments": () => [
        {
          id: "dep_1",
          project_id: "prj_1",
          name: "churn-api",
          model_version_id: "mdl_1",
          endpoint: "/api/v1/deployments/dep_1/predict",
          status: "active",
          monitoring: { email: [] },
          version: 1,
        },
      ],
      "GET /api/v1/deployments/dep_1/drift": () => [report],
      "GET /api/v1/projects/prj_1/alerts": () => [
        {
          id: "alr_1",
          project_id: "prj_1",
          kind: "data_drift",
          severity: "high",
          title: "Drift de datos en churn-api",
          message: "Features con drift: antiguedad_meses",
          status,
          created_at: "2026-09-28T11:00:00Z",
          version: 1,
        },
      ],
      "POST /api/v1/alerts/alr_1/resolve": () => {
        actions.push("resolve");
        status = "resolved";
        return { id: "alr_1", status };
      },
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <Providers client={client}>
        <App router={createTestRouter("/projects/prj_1/monitoring")} />
      </Providers>,
    );
    const row = (await screen.findByText("antiguedad_meses", { selector: "td" })).closest("tr");
    expect(row && within(row).getByText("Alto")).toBeInTheDocument();
    expect(row && within(row).getByText(/36,2\d* → 3,4/)).toBeInTheDocument();
    expect(screen.getByText(/roc_auc con feedback: 0,71/)).toBeInTheDocument();
    expect(screen.getByTestId("embedding-drift")).toHaveTextContent(/AUC de dominio 0,74/);
    expect(screen.getByText("Drift de datos en churn-api")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Resolver" }));
    expect(actions).toEqual(["resolve"]);
    expect(await screen.findByText("No hay alertas.")).toBeInTheDocument();
  });
});
