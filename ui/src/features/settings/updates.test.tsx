import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import {
  setPlatform,
  WebPlatformBridge,
  type DesktopUpdater,
  type UpdateInfo,
  type UpdateProgress,
} from "@/lib/platform/bridge";
import { hardware, health, mockEngine } from "@/test/engine";

import { UpdatesCard } from "./Updates";

const next: UpdateInfo = {
  version: "0.2.0",
  current_version: "0.1.0",
  notes: "Monitoreo en producción",
  date: "2026-10-01T12:00:00Z",
};

/** Solo el updater: sin runtime, para no pasar por la pantalla de preparación. */
function withUpdater(check: () => Promise<UpdateInfo | null>, install = async () => {}) {
  let emit: (p: UpdateProgress) => void = () => {};
  const updater: DesktopUpdater = {
    check: vi.fn(check),
    onProgress: vi.fn(async (cb: (p: UpdateProgress) => void) => {
      emit = cb;
      return () => {};
    }),
    install: vi.fn(install),
  };
  class Bridge extends WebPlatformBridge {
    readonly updater = updater;
  }
  setPlatform(new Bridge());
  return { updater, emit: (p: UpdateProgress) => emit(p) };
}

const client = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

describe("actualizaciones del desktop (ADR-0037)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => {
    setPlatform(new WebPlatformBridge());
    vi.unstubAllGlobals();
  });

  it("en la web no muestra la tarjeta", () => {
    render(
      <Providers client={client()}>
        <UpdatesCard />
      </Providers>,
    );
    expect(screen.queryByText("Actualizaciones")).not.toBeInTheDocument();
  });

  it("informa que está al día y permite volver a buscar", async () => {
    const fake = withUpdater(async () => null);
    render(
      <Providers client={client()}>
        <UpdatesCard />
      </Providers>,
    );
    expect(await screen.findByText("Tenés la última versión.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Buscar actualizaciones" }));
    await waitFor(() => expect(fake.updater.check).toHaveBeenCalledTimes(2));
  });

  it("muestra la versión nueva e instala con progreso", async () => {
    let finish: () => void = () => {};
    const fake = withUpdater(
      async () => next,
      () => new Promise<void>((r) => (finish = r)),
    );
    render(
      <Providers client={client()}>
        <UpdatesCard />
      </Providers>,
    );
    expect(await screen.findByText("Hay una versión nueva: 0.2.0")).toBeInTheDocument();
    expect(screen.getByText("Monitoreo en producción")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Instalar y reiniciar/ }));
    await waitFor(() => expect(fake.updater.install).toHaveBeenCalled());
    fake.emit({ downloaded: 5 * 1_048_576, total: 10 * 1_048_576 });
    expect(await screen.findByText(/5\.0 de 10\.0 MB/)).toBeInTheDocument();
    finish();
  });

  it("si la instalación falla, muestra el error y vuelve a consultar", async () => {
    const fake = withUpdater(
      async () => next,
      async () => {
        throw new Error("firma inválida");
      },
    );
    render(
      <Providers client={client()}>
        <UpdatesCard />
      </Providers>,
    );
    await userEvent.click(await screen.findByRole("button", { name: /Instalar y reiniciar/ }));
    expect(await screen.findByText(/firma inválida/)).toBeInTheDocument();
    await waitFor(() => expect(fake.updater.check).toHaveBeenCalledTimes(2));
  });

  it("avisa en el encabezado al abrir la app y se puede descartar", async () => {
    withUpdater(async () => next);
    mockEngine({
      "GET /api/v1/system/health": () => health,
      "GET /api/v1/system/hardware": () => hardware,
      "GET /api/v1/projects": () => [],
    });
    render(
      <Providers client={client()}>
        <App router={createTestRouter("/")} />
      </Providers>,
    );
    expect(await screen.findByText("Perceptron 0.2.0 está disponible.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Ver actualización" })).toHaveAttribute(
      "href",
      "/settings",
    );
    await userEvent.click(screen.getByRole("button", { name: "Descartar aviso" }));
    expect(screen.queryByText("Perceptron 0.2.0 está disponible.")).not.toBeInTheDocument();
  });
});
