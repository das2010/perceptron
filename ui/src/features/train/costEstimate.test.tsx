import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { CostEstimate } from "./CostEstimate";

describe("costo por dispositivo (RF-PRF-08)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("muestra tamaño efectivo, memoria y tiempo por época en cada dispositivo", async () => {
    mockEngine({
      "POST /api/v1/projects/prj_1/arch/estimate": () => ({
        n_train: 280,
        num_samples: 400,
        size_bytes: 20 * 2 ** 20,
        input_shape: null,
        sample_mb: null,
        train_tensor_mb: null,
        batch_size: 32,
        num_params: 12345,
        memory_mb: 3.5,
        devices: [
          { device: "cpu", name: "Ryzen", memory_gb: 16, fits: true, epoch_time_s: 1.2 },
          { device: "cuda", name: "RTX 4060", memory_gb: 0.001, fits: false, epoch_time_s: 0.1 },
        ],
      }),
    });
    render(
      <Providers client={new QueryClient()}>
        <CostEstimate projectId="prj_1" archspecId="arc_1" datasetVersionId="dsv_1" />
      </Providers>,
    );
    await userEvent.click(screen.getByRole("button", { name: /Estimar costo por dispositivo/ }));
    expect(await screen.findByText(/280 ejemplos de train de 400/)).toBeInTheDocument();
    expect(screen.getByText("RTX 4060")).toBeInTheDocument();
    expect(screen.getByText("no entra")).toBeInTheDocument();
    expect(screen.getByText("entra")).toBeInTheDocument();
  });
});
