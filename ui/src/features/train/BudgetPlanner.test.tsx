import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { BudgetPlanner } from "./BudgetPlanner";

const plan = {
  max_trials: 40,
  max_epochs_per_trial: 25,
  tuned_params: ["lr"],
  epoch_time_s: 0.5,
  time_budget_s: 1200,
  estimated_s: 500,
  reasons: [],
};

beforeEach(() => resetApiClient());
afterEach(() => vi.unstubAllGlobals());

function mount(onPlan: () => void, release: Promise<void>) {
  mockEngine({
    "POST /api/v1/projects/prj_1/hpo/plan": async () => {
      await release;
      return plan;
    },
  });
  const ui = (trials: number) => (
    <Providers client={new QueryClient()}>
      <BudgetPlanner
        projectId="prj_1"
        archspecId="arc_1"
        datasetVersionId="dsv_1"
        trials={trials}
        epochs={10}
        onPlan={onPlan}
      />
    </Providers>
  );
  const view = render(ui(10));
  return { rerender: (trials: number) => view.rerender(ui(trials)) };
}

it("el plan automático no pisa lo que el usuario editó mientras se calculaba", async () => {
  const onPlan = vi.fn();
  let done = () => {};
  const { rerender } = mount(onPlan, new Promise<void>((r) => (done = r)));
  rerender(3); // el usuario escribe 3 intentos antes de que llegue el plan
  done();
  expect(await screen.findByText("Propuesta: 40 intentos de hasta 25 épocas.")).toBeInTheDocument();
  expect(onPlan).not.toHaveBeenCalled();
});

it("sin ediciones, el plan automático se aplica", async () => {
  const onPlan = vi.fn();
  mount(onPlan, Promise.resolve());
  expect(await screen.findByText("Propuesta: 40 intentos de hasta 25 épocas.")).toBeInTheDocument();
  expect(onPlan).toHaveBeenCalledWith(plan);
});
