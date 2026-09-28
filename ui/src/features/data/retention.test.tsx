import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { RetentionCard } from "./Retention";

describe("retención de versiones (RF-MON-07)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("muestra la vista previa y aplica solo tras confirmar", async () => {
    const calls: { dry_run: boolean; keep_last: number }[] = [];
    mockEngine({
      "POST /api/v1/projects/prj_1/datasets/retention": async (req) => {
        const body = (await req.json()) as { dry_run: boolean; keep_last: number };
        calls.push(body);
        return {
          dry_run: body.dry_run,
          keep_last: body.keep_last,
          kept: ["dsv_3"],
          in_use: ["dsv_1"],
          deleted: ["dsv_2"],
          freed_bytes: 5 * 2 ** 20,
        };
      },
    });
    render(
      <Providers client={new QueryClient()}>
        <RetentionCard projectId="prj_1" versions={3} />
      </Providers>,
    );
    fireEvent.change(screen.getByLabelText("Conservar las últimas"), { target: { value: "1" } });
    await userEvent.click(screen.getByRole("button", { name: "Ver qué se borraría" }));
    expect(await screen.findByText(/Se borraría 1 versión \(5\.0 MB\)/)).toBeInTheDocument();
    expect(calls).toEqual([{ keep_last: 1, dry_run: true }]);
    await userEvent.click(screen.getByRole("button", { name: "Borrar 1 versión" }));
    await waitFor(() => expect(calls.at(-1)).toEqual({ keep_last: 1, dry_run: false }));
  });
});
