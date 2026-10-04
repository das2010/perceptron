import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Providers } from "@/app/providers";
import type { EvaluationReport } from "@/lib/api/hooks";

import { CostThreshold } from "./CostThreshold";

const summary = (fn: number, fp: number, recall: number, cost: number) => ({
  tp: 50 - fn,
  fp,
  fn,
  tn: 100 - fp,
  precision: 0.7,
  recall,
  cost,
});

describe("umbral por costo (ADR-0040)", () => {
  it("compara el umbral elegido en validación con el del 50 % en test", () => {
    const report = {
      details: {
        cost_threshold: {
          positive_class: "falla",
          worse: "false_negative",
          ratio: 10,
          threshold: 0.31,
          test: summary(3, 30, 0.94, 60),
          test_at_50: summary(15, 8, 0.7, 158),
        },
      },
    } as unknown as EvaluationReport;
    render(
      <Providers client={new QueryClient()}>
        <CostThreshold report={report} />
      </Providers>,
    );
    expect(screen.getByText("Umbral por costo de los errores")).toBeInTheDocument();
    expect(screen.getByText(/No detectar «falla» se consideró 10 veces peor/)).toBeInTheDocument();
    expect(screen.getByText("Umbral por costo (0,31)")).toBeInTheDocument();
    expect(screen.getByText("158")).toBeInTheDocument();
  });

  it("no muestra nada sin costos declarados", () => {
    const { container } = render(
      <Providers client={new QueryClient()}>
        <CostThreshold report={{ details: {} } as unknown as EvaluationReport} />
      </Providers>,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
