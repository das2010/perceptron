import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { ModelsCacheCard } from "./ModelsCache";

const report = (models: { id: string; size_bytes: number }[]) => ({
  path: "C:/Users/ana/Perceptron/models_cache",
  offline: false,
  total_bytes: models.reduce((a, m) => a + m.size_bytes, 0),
  models: models.map((m) => ({ ...m, source: "huggingface", files: 3, last_used: null })),
  available: ["resnet18"],
});

function wrap() {
  return render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <ModelsCacheCard />
    </Providers>,
  );
}

describe("caché de modelos preentrenados (RF-TRN-11)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("lista, verifica, borra y predescarga", async () => {
    let prefetched: unknown = null;
    mockEngine({
      "GET /api/v1/system/models-cache": () =>
        report([{ id: "org/encoder", size_bytes: 300 * 1024 ** 2 }]),
      "POST /api/v1/system/models-cache/verify": () => ({ checked: 3, corrupt: [] }),
      // openapi-fetch codifica la "/" del id; el servidor la decodifica ({model_id:path}).
      "DELETE /api/v1/system/models-cache/org%2Fencoder": () => report([]),
      "POST /api/v1/system/models-cache/prefetch": async (req) => {
        prefetched = await req.json();
        return { id: "job_1", kind: "prefetch", status: "queued", refs: {} };
      },
    });
    wrap();
    const row = (await screen.findByText("org/encoder")).closest("tr");
    if (!row) throw new Error("sin fila");
    expect(within(row).getByText("300.0 MB")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Verificar integridad/ }));
    expect(await screen.findByText("3 archivos verificados: todo en orden.")).toBeInTheDocument();
    await userEvent.selectOptions(
      screen.getByLabelText("Descargar para usar sin conexión"),
      "resnet18",
    );
    await userEvent.click(screen.getByRole("button", { name: /^Descargar$/ }));
    await waitFor(() => expect(prefetched).toEqual({ model: "resnet18" }));
    await userEvent.click(screen.getByRole("button", { name: "Borrar org/encoder de la caché" }));
    expect(
      await screen.findByText("Todavía no se descargó ningún modelo preentrenado."),
    ).toBeInTheDocument();
  });
});
