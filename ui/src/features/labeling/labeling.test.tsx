import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

const labelset = {
  id: "lbl_1",
  dataset_version_id: "dsv_1",
  kind: "class",
  classes: ["bueno", "malo"],
  name: "reclamos",
  target: "label",
  origin: "human",
  version: 1,
};
const sample = (i: number, label: string, confidence: number) => ({
  sample_id: `row:${i}`,
  split: "train",
  path: null,
  text: `texto ${i}`,
  fields: {},
  item: {
    sample_id: `row:${i}`,
    label,
    confidence,
    status: "suggested",
    origin: "model",
    boxes: [],
  },
});

describe("etiquetado asistido (Capa 4c)", () => {
  beforeEach(() => resetApiClient());

  it("etiqueta con atajos, acepta la sugerencia con Enter y acepta en lote", async () => {
    const puts: unknown[] = [];
    const accepts: unknown[] = [];
    const done = new Set<string>();
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/datasets": () => [
        {
          id: "dsv_1",
          project_id: "prj_1",
          content_hash: "abcdef1234",
          num_samples: 10,
          size_bytes: 1,
          target: "label",
        },
      ],
      "GET /api/v1/projects/prj_1/runs": () => [],
      "GET /api/v1/datasets/dsv_1/labelsets": () => [labelset],
      "GET /api/v1/labelsets/lbl_1": () => ({
        labelset,
        total: 10,
        accepted: 4,
        suggested: 2,
        unlabeled: 6,
        by_class: { bueno: 3, malo: 1 },
      }),
      "GET /api/v1/labelsets/lbl_1/queue": () =>
        [sample(5, "malo", 0.55), sample(6, "bueno", 0.62)].filter((x) => !done.has(x.sample_id)),
      "PUT /api/v1/labelsets/lbl_1/labels": async (req) => {
        const body = (await req.json()) as { updates: { sample_id: string }[] };
        puts.push(body);
        body.updates.forEach((u) => done.add(u.sample_id));
        return { count: 1 };
      },
      "POST /api/v1/labelsets/lbl_1/accept": async (req) => {
        accepts.push(await req.json());
        return { count: 2 };
      },
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter("/projects/prj_1/labeling")} />
      </Providers>,
    );
    expect(await screen.findByText("texto 5")).toBeInTheDocument();
    expect(screen.getByText(/4 de 10 etiquetadas/)).toBeInTheDocument();
    expect(screen.getByText(/Sugerencia: «malo» \(55 %\)/)).toBeInTheDocument();

    await userEvent.keyboard("1"); // primera clase: «bueno» (corrige la sugerencia)
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ updates: [{ sample_id: "row:5", label: "bueno", boxes: [] }] });
    expect(await screen.findByText("texto 6")).toBeInTheDocument();

    await userEvent.keyboard("{Enter}"); // acepta la sugerencia «bueno»
    await waitFor(() => expect(puts).toHaveLength(2));
    expect(puts[1]).toEqual({ updates: [{ sample_id: "row:6", label: "bueno", boxes: [] }] });

    await userEvent.click(screen.getByRole("button", { name: /Aceptar sugerencias confiables/ }));
    await waitFor(() => expect(accepts).toEqual([{ min_confidence: 0.9 }]));
  });
});
