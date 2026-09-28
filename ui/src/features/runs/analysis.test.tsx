import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { AnalysisPanel } from "./AnalysisPanel";

const RUN = "std_1-t000";

describe("evaluación avanzada (Capa 4b)", () => {
  beforeEach(() => resetApiClient());

  it("muestra errores y compara la equidad entre grupos", async () => {
    const fairnessBodies: unknown[] = [];
    mockEngine({
      [`GET /api/v1/runs/${RUN}/errors`]: () => ({
        task: "classification",
        metric: "accuracy",
        overall: 0.8,
        num_samples: 100,
        num_errors: 20,
        label_issues: 3,
        slices: [{ column: "region", value: "Sur", support: 20, metric: 0.55, gap: 0.25 }],
        confusions: [{ actual: "1", predicted: "0", count: 12 }],
        samples: [
          {
            row: 4,
            actual: "1",
            predicted: "0",
            confidence: 0.91,
            label_issue: true,
            features: { edad: 41, region: "Sur" },
          },
        ],
      }),
      "GET /api/v1/datasets/dsv_1/profile": () => ({
        num_samples: 100,
        target: { name: "churn" },
        columns: [{ name: "edad" }, { name: "region" }, { name: "churn" }],
        alerts: [],
      }),
      [`POST /api/v1/runs/${RUN}/fairness`]: async (req) => {
        fairnessBodies.push(await req.json());
        return [
          {
            attribute: "region",
            task: "classification",
            positive_class: "1",
            threshold: 0.1,
            demographic_parity_difference: 0.3,
            equalized_odds_difference: 0.2,
            alerts: ["Paridad demográfica: la tasa de «1» difiere 0.30 entre grupos de region"],
            groups: [
              {
                group: "Norte",
                support: 40,
                selection_rate: 0.1,
                accuracy: 0.9,
                tpr: 0.8,
                fpr: 0.05,
              },
              { group: "Sur", support: 30, selection_rate: 0.4, accuracy: 0.7, tpr: 0.6, fpr: 0.2 },
            ],
          },
        ];
      },
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <AnalysisPanel runId={RUN} datasetVersionId="dsv_1" classes={["0", "1"]} />
      </Providers>,
    );
    expect(await screen.findByText(/20 errores sobre 100 casos/)).toBeInTheDocument();
    expect(screen.getByText("3 posibles errores de etiqueta")).toBeInTheDocument();
    expect(screen.getByText("«1» predicho como «0»: 12")).toBeInTheDocument();
    expect(screen.getByTitle("La etiqueta podría estar mal")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Equidad" }));
    await userEvent.click(await screen.findByLabelText("region"));
    expect(screen.queryByLabelText("churn")).not.toBeInTheDocument(); // el target no es atributo
    await userEvent.click(screen.getByRole("button", { name: "Comparar grupos" }));
    await waitFor(() => expect(fairnessBodies).toHaveLength(1));
    expect(fairnessBodies[0]).toMatchObject({ attributes: ["region"], positive_class: null });
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Paridad demográfica");
    const table = screen.getAllByRole("table").at(-1);
    if (!table) throw new Error("sin tabla");
    expect(within(table).getByText("Sur")).toBeInTheDocument();
  });
});
