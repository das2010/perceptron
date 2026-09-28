import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

// Monaco no corre en jsdom: un textarea con la misma interfaz.
vi.mock("@/components/CodeView", () => ({
  default: ({
    code,
    onChange,
    label,
  }: {
    code: string;
    onChange?: (v: string) => void;
    label?: string;
  }) => (
    <textarea
      aria-label={label}
      value={code}
      readOnly={!onChange}
      onChange={(e) => onChange?.(e.target.value)}
    />
  ),
}));

const STARTER = "def build_model(config):\n    return None\n";
const record = (extra: Record<string, unknown> = {}) => ({
  id: "arc_1",
  project_id: "prj_1",
  name: "tabular-mlp",
  origin: "rules",
  content_hash: "h",
  spec: { name: "tabular-mlp", modality: "tabular", nodes: [], edges: [] },
  version: 1,
  ...extra,
});

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App router={createTestRouter(path)} />
    </Providers>,
  );
}

describe("modo experto (RF-ARC-06)", () => {
  beforeEach(() => resetApiClient());

  it("lint en vivo, confirmación obligatoria y prueba en el sandbox al guardar", async () => {
    const created: Record<string, unknown>[] = [];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/archspecs/arc_1": () => record(),
      "GET /api/v1/archspecs/arc_1/code/starter": () => ({ code: STARTER }),
      "POST /api/v1/arch/code/lint": async (req) => {
        const { source } = (await req.json()) as { source: string };
        return source.includes("import os")
          ? { valid: false, issues: [{ line: 1, col: 0, message: "import no permitido: os" }] }
          : { valid: true, issues: [] };
      },
      "POST /api/v1/projects/prj_1/archspecs/code": async (req) => {
        created.push((await req.json()) as Record<string, unknown>);
        return new Response(
          JSON.stringify({
            record: record({ id: "arc_2", origin: "manual", code_path: "archspecs/arc_2.py" }),
            check: { ok: true, num_params: 1234, output_shape: [2], violations: [] },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        );
      },
      "GET /api/v1/archspecs/arc_2": () =>
        record({ id: "arc_2", origin: "manual", code_path: "archspecs/arc_2.py" }),
      "GET /api/v1/archspecs/arc_2/code": () => ({ code: STARTER }),
    });
    renderAt("/projects/prj_1/archspecs/arc_1/code");
    const editor = await screen.findByLabelText("Código del modelo");
    expect(await screen.findByText("Sin problemas", {}, { timeout: 3000 })).toBeInTheDocument();
    const save = screen.getByRole("button", { name: "Probar y guardar" });
    expect(save).toBeDisabled(); // falta confirmar

    fireEvent.change(editor, { target: { value: `import os\n${STARTER}` } });
    expect(
      await screen.findByText("import no permitido: os", {}, { timeout: 3000 }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText(/Entiendo que el código no es declarativo/));
    expect(save).toBeDisabled(); // hay problemas

    fireEvent.change(editor, { target: { value: STARTER } });
    await waitFor(() => expect(save).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(save);
    await waitFor(() => expect(created).toHaveLength(1));
    expect(created[0]).toMatchObject({
      base_archspec_id: "arc_1",
      acknowledge_risk: true,
      source: STARTER,
    });
    // La ArchSpec de código se muestra como fuente, marcada como no declarativa.
    expect(await screen.findByText("Código experto (no declarativo)")).toBeInTheDocument();
  }, 20_000);
});
