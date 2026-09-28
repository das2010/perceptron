import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

type Put = { updates: Record<string, unknown>[] };

function setup(kind: "mask" | "temporal_event", path: string) {
  const labelset = {
    id: "lbl_2",
    dataset_version_id: "dsv_1",
    kind,
    classes: ["rayón", "mancha"],
    name: "formas",
    target: "label",
    origin: "human",
    version: 1,
  };
  const puts: Put[] = [];
  mockEngine({
    "GET /api/v1/projects": () => [project()],
    "GET /api/v1/projects/prj_1": () => project(),
    "GET /api/v1/projects/prj_1/datasets": () => [
      { id: "dsv_1", project_id: "prj_1", content_hash: "abcdef1234", num_samples: 3 },
    ],
    "GET /api/v1/projects/prj_1/runs": () => [],
    "GET /api/v1/datasets/dsv_1/labelsets": () => [labelset],
    "GET /api/v1/labelsets/lbl_2": () => ({
      labelset,
      total: 3,
      accepted: 0,
      suggested: 0,
      unlabeled: 3,
      by_class: {},
    }),
    "GET /api/v1/labelsets/lbl_2/queue": () => [
      { sample_id: "row:0", split: "train", path, text: null, fields: {}, item: null },
    ],
    "GET /api/v1/labelsets/lbl_2/samples/row:0/file": () => new Response("x"),
    "PUT /api/v1/labelsets/lbl_2/labels": async (req) => {
      puts.push((await req.json()) as Put);
      return { count: 1 };
    },
  });
  vi.stubGlobal(
    "URL",
    Object.assign(URL, { createObjectURL: vi.fn(() => "blob:muestra"), revokeObjectURL: vi.fn() }),
  );
  const view = render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <App router={createTestRouter("/projects/prj_1/labeling")} />
    </Providers>,
  );
  return { puts, view };
}

describe("formas de etiquetado (RF-LBL-01)", () => {
  beforeEach(() => resetApiClient());

  it("dibuja un polígono con clics y lo guarda normalizado", async () => {
    const { puts } = setup("mask", "img/pieza.png");
    const canvas = await screen.findByTestId("polygon-canvas", {}, { timeout: 5000 });
    canvas.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 200, height: 100, right: 200, bottom: 100 }) as DOMRect;
    fireEvent.click(canvas, { clientX: 20, clientY: 10 });
    fireEvent.click(canvas, { clientX: 180, clientY: 10 });
    fireEvent.click(canvas, { clientX: 100, clientY: 90 });
    await userEvent.click(screen.getByRole("button", { name: "Cerrar polígono" }));
    expect(screen.getByText("1 polígono")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    await waitFor(() => expect(puts).toHaveLength(1));
    const u = puts[0]?.updates[0];
    expect(u?.polygons).toEqual([
      {
        points: [
          [0.1, 0.1],
          [0.9, 0.1],
          [0.5, 0.9],
        ],
        label: "rayón",
      },
    ]);
  });

  it("marca inicio y fin de un evento sobre el audio", async () => {
    const { puts, view } = setup("temporal_event", "normal/clip_001.wav");
    await screen.findByTestId("segment-bar", {}, { timeout: 5000 });
    const audio = view.container.querySelector("audio");
    if (!audio) throw new Error("sin reproductor");
    Object.defineProperty(audio, "currentTime", { value: 0.25, writable: true });
    await userEvent.click(screen.getByRole("button", { name: "Marcar inicio" }));
    audio.currentTime = 0.75;
    await userEvent.click(screen.getByRole("button", { name: "Marcar fin" }));
    expect(screen.getByText("1 segmento")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Clase de la forma"), "mancha");
    audio.currentTime = 0.8;
    await userEvent.click(screen.getByRole("button", { name: "Marcar inicio" }));
    audio.currentTime = 0.9;
    await userEvent.click(screen.getByRole("button", { name: "Marcar fin" }));
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]?.updates[0]?.segments).toEqual([
      { start_s: 0.25, end_s: 0.75, label: "rayón" },
      { start_s: 0.8, end_s: 0.9, label: "mancha" },
    ]);
  });
});
