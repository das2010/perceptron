import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

import { addBlock, hpName, move, removeBlock, type Spec } from "./spec";

// React Flow mide el contenedor: jsdom no trae ResizeObserver ni DOMMatrix.
beforeAll(() => {
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
  (globalThis as { DOMMatrixReadOnly?: unknown }).DOMMatrixReadOnly ??= class {
    m22 = 1;
  };
});

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App router={createTestRouter(path)} />
    </Providers>,
  );
}

const spec = (): Spec => ({
  name: "mlp-tabular",
  modality: "tabular",
  nodes: [
    { id: "embed", block: "tabular.embed", params: {} },
    { id: "mlp", block: "tabular.mlp", params: { hidden: [128, 64], dropout: { hp: "dropout", default: 0.1 } } },
    { id: "head", block: "head.classification", params: {} },
  ],
  edges: [
    ["input", "embed"],
    ["embed", "mlp"],
    ["mlp", "head"],
  ],
});

const blocks = [
  {
    key: "tabular.mlp",
    description: "MLP",
    params: {
      hidden: { type: "int_list", default: [128] },
      dropout: { type: "float", default: 0.1, low: 0, high: 0.6, tunable: true },
    },
  },
  { key: "tabular.embed", description: "Embeddings", params: {} },
  { key: "head.classification", description: "Cabeza", params: {} },
  { key: "tabular.resnet", description: "ResNet tabular", params: {} },
];

describe("operaciones sobre specs", () => {
  it("agregar y quitar bloques mantiene la cadena conectada", () => {
    const { spec: added, id } = addBlock(spec(), "tabular.mlp");
    expect(id).toBe("mlp_2");
    expect(added.edges.at(-1)).toEqual(["head", "mlp_2"]);
    const removed = removeBlock(spec(), "mlp");
    expect(removed.nodes.map((n) => n.id)).toEqual(["embed", "head"]);
    expect(removed.edges).toEqual([
      ["input", "embed"],
      ["embed", "head"],
    ]);
  });

  it("nombres de hiperparámetro válidos y reordenamiento", () => {
    expect(hpName("MLP-2", "dropout")).toBe("mlp_2_dropout");
    expect(hpName("2x", "lr")).toBe("x_lr");
    expect(move(["a", "b", "c"], 2, -1)).toEqual(["a", "c", "b"]);
    expect(move(["a", "b"], 0, -1)).toEqual(["a", "b"]);
  });
});

describe("editor visual de ArchSpec (RF-ARC-05)", () => {
  beforeEach(() => resetApiClient());

  it("valida en vivo, edita parámetros y guarda como nueva", async () => {
    const validated: Spec[] = [];
    let saved: { spec: Spec } | null = null;
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/archspecs/arc_1": () => ({
        id: "arc_1",
        project_id: "prj_1",
        name: "mlp-tabular",
        origin: "rules",
        content_hash: "abc",
        spec: spec(),
        version: 1,
      }),
      "GET /api/v1/catalog/blocks": () => blocks,
      "POST /api/v1/arch/validate": async (req) => {
        const body = (await req.json()) as Spec;
        validated.push(body);
        const ok = body.nodes.at(-1)?.block === "head.classification";
        return {
          valid: ok,
          issues: ok
            ? []
            : [{ stage: "graph", path: "nodes", message: "falta la cabeza", severity: "error" }],
          num_params: 12345,
          estimated_memory_mb: 3,
        };
      },
      "POST /api/v1/projects/prj_1/archspecs": async (req) => {
        saved = (await req.json()) as { spec: Spec };
        return new Response(
          JSON.stringify({ id: "arc_2", project_id: "prj_1", name: saved.spec.name, origin: "manual", content_hash: "x", spec: saved.spec, version: 1 }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        );
      },
      "GET /api/v1/archspecs/arc_2": () => ({
        id: "arc_2",
        project_id: "prj_1",
        name: "mlp-editada",
        origin: "manual",
        content_hash: "x",
        spec: { ...spec(), name: "mlp-editada" },
        version: 1,
      }),
    });
    renderAt("/projects/prj_1/archspecs/arc_1");
    expect(await screen.findByText("Válida", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText(/12\.345 parámetros/)).toBeInTheDocument();

    // Agregar un bloque al final deja la cabeza en el medio: el validador lo marca.
    await userEvent.selectOptions(screen.getByLabelText("Agregar bloque"), "tabular.resnet");
    await userEvent.click(screen.getByRole("button", { name: "Agregar" }));
    expect(await screen.findByText(/falta la cabeza/, {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Guardar como nueva/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Quitar" }));
    expect(await screen.findByText("Válida", {}, { timeout: 3000 })).toBeInTheDocument();

    const name = screen.getByLabelText("Nombre");
    await userEvent.clear(name);
    await userEvent.type(name, "mlp-editada");
    await waitFor(() => expect(validated.at(-1)?.name).toBe("mlp-editada"), { timeout: 3000 });
    await waitFor(() => expect(screen.getByRole("button", { name: /Guardar como nueva/ })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: /Guardar como nueva/ }));
    await waitFor(() => expect(saved?.spec.name).toBe("mlp-editada"));
    expect(saved!.spec.nodes.map((n) => n.id)).toEqual(["embed", "mlp", "head"]);
  });
});

describe("editor visual de pipeline (RF-PIP-02)", () => {
  beforeEach(() => resetApiClient());

  it("reordena pasos y guarda con la versión", async () => {
    let put: { graph: { steps: { id: string }[] }; version: number } | null = null;
    const graph = {
      modality: "tabular",
      pipeline_version: "1.0",
      target: null,
      rationale: ["Imputar antes de escalar"],
      steps: [
        { id: "imp", kind: "impute_numeric", columns: ["edad"], params: { strategy: "median" } },
        { id: "esc", kind: "scale", columns: ["edad"], params: {} },
      ],
    };
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/datasets": () => [],
      "GET /api/v1/pipelines/pip_1": () => ({
        id: "pip_1",
        project_id: "prj_1",
        name: "tabular",
        origin: "rules",
        graph,
        version: put ? 4 : 3,
      }),
      "PUT /api/v1/pipelines/pip_1": async (req) => {
        put = (await req.json()) as typeof put;
        return { id: "pip_1", project_id: "prj_1", name: "tabular", origin: "rules", graph: put!.graph, version: 4 };
      },
    });
    renderAt("/projects/prj_1/pipelines/pip_1");
    expect(await screen.findByText("Imputar antes de escalar")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /^2\. scale/ }));
    await userEvent.click(screen.getByRole("button", { name: "Subir" }));
    expect(screen.getByText("Sin guardar")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /^Guardar$/ }));
    await waitFor(() => expect(put?.version).toBe(3));
    expect(put!.graph.steps.map((s) => s.id)).toEqual(["esc", "imp"]);
  });
});
