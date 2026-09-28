import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import type { ExportReport } from "@/lib/api/hooks";
import { mockEngine } from "@/test/engine";

import { PlaygroundPanel } from "./PlaygroundPanel";

const prediction = {
  predictions: [
    { prediction: "acceso", confidence: 0.9, probabilities: { acceso: 0.9, facturacion: 0.1 } },
  ],
  task: "classification",
};

function report(kind: string) {
  return { signature: { inputs: { kind } }, artifacts: [] } as unknown as ExportReport;
}

function wrap(kind: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <PlaygroundPanel runId="run_1" datasetVersionId="dsv_1" report={report(kind)} />
    </Providers>,
  );
}

describe("playground de texto y audio (RF-EXP-02)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("clasifica un texto escrito", async () => {
    let body: unknown = null;
    mockEngine({
      "POST /api/v1/runs/run_1/predict/text": async (req) => {
        body = await req.json();
        return { ...prediction, input_kind: "tokens" };
      },
    });
    wrap("tokens");
    await userEvent.type(screen.getByLabelText("Texto a clasificar"), "No puedo entrar");
    await userEvent.click(screen.getByRole("button", { name: "Predecir" }));
    expect((await screen.findAllByText("acceso")).length).toBeGreaterThan(0);
    expect(body).toEqual({ texts: ["No puedo entrar"] });
    // Sin explicación local para texto (por ahora solo tabular e imagen).
    expect(screen.queryByRole("button", { name: /Explicar/ })).not.toBeInTheDocument();
  });

  it("acepta los formatos de audio del entrenamiento", () => {
    mockEngine({});
    wrap("spectrogram");
    const input = screen.getByLabelText("Audio a clasificar");
    expect(input).toHaveAttribute("type", "file");
    for (const ext of [".wav", ".flac", ".ogg", ".mp3"]) {
      expect(input.getAttribute("accept")).toContain(ext);
    }
    expect(screen.queryByLabelText("Imagen a clasificar")).not.toBeInTheDocument();
  });
});
