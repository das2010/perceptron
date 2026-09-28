import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { DesktopSettings } from "@/features/settings/DesktopSettings";
import {
  setPlatform,
  WebPlatformBridge,
  type PlatformBridge,
  type RuntimeProgress,
} from "@/lib/platform/bridge";

import { RuntimeGate } from "./RuntimeGate";

function fakeDesktop(engine: () => Promise<{ baseUrl: string; token: string }>) {
  let emit: (p: RuntimeProgress) => void = () => {};
  const runtime = {
    state: vi.fn(async () => ({
      app_version: "0.1.0",
      torch_variant: "cpu",
      torch_index: "https://download.pytorch.org/whl/cpu",
      nvidia_driver: null,
    })),
    onProgress: vi.fn(async (cb: (p: RuntimeProgress) => void) => {
      emit = cb;
      return () => {};
    }),
    setTorchVariant: vi.fn(async () => {}),
    restart: vi.fn(async () => {}),
  };
  const bridge: PlatformBridge = {
    kind: "desktop",
    runtime,
    engine: vi.fn(engine),
    pickDirectory: async () => null,
    getSecret: async () => null,
    setSecret: async () => {},
    openExternal: async () => {},
  };
  return { bridge, runtime, emit: (p: RuntimeProgress) => emit(p) };
}

const wrap = (ui: React.ReactNode) =>
  render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {ui}
    </Providers>,
  );

describe("desktop (ADR-0026)", () => {
  afterEach(() => setPlatform(new WebPlatformBridge()));

  it("muestra el progreso del aprovisionamiento y entra cuando el Engine está listo", async () => {
    let resolve: (v: { baseUrl: string; token: string }) => void = () => {};
    const fake = fakeDesktop(() => new Promise((r) => (resolve = r)));
    setPlatform(fake.bridge);
    wrap(
      <RuntimeGate>
        <p>app</p>
      </RuntimeGate>,
    );
    expect(screen.getByText("Preparando Perceptron")).toBeInTheDocument();
    await waitFor(() => expect(fake.runtime.onProgress).toHaveBeenCalled());
    fake.emit({ step: "python", message: "Instalando Python" });
    fake.emit({ step: "torch", message: "Instalando PyTorch (cpu)" });
    expect(await screen.findByText("Instalando PyTorch (cpu)")).toBeInTheDocument();
    resolve({ baseUrl: "http://127.0.0.1:5000", token: "t" });
    expect(await screen.findByText("app")).toBeInTheDocument();
  });

  it("si falla, muestra el error y permite reintentar", async () => {
    let calls = 0;
    const fake = fakeDesktop(() =>
      ++calls === 1
        ? Promise.reject(new Error("sin conexión"))
        : Promise.resolve({ baseUrl: "http://127.0.0.1:5001", token: "t" }),
    );
    setPlatform(fake.bridge);
    wrap(
      <RuntimeGate>
        <p>app</p>
      </RuntimeGate>,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("sin conexión");
    await userEvent.click(screen.getByRole("button", { name: "Reintentar" }));
    expect(await screen.findByText("app")).toBeInTheDocument();
    expect(fake.runtime.restart).toHaveBeenCalledTimes(1);
  });

  it("en la web no hay pantalla de preparación ni configuración del motor local", () => {
    wrap(
      <>
        <RuntimeGate>
          <p>app</p>
        </RuntimeGate>
        <DesktopSettings />
      </>,
    );
    expect(screen.getByText("app")).toBeInTheDocument();
    expect(screen.queryByText("Motor local (PyTorch)")).not.toBeInTheDocument();
  });

  it("cambia la variante de PyTorch sin reinstalar la app", async () => {
    const fake = fakeDesktop(async () => ({ baseUrl: "http://127.0.0.1:5000", token: "t" }));
    setPlatform(fake.bridge);
    wrap(<DesktopSettings />);
    expect(await screen.findByText("https://download.pytorch.org/whl/cpu")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Cambiar variante"), "cuda");
    await userEvent.click(screen.getByRole("button", { name: "Aplicar" }));
    await waitFor(() => expect(fake.runtime.setTorchVariant).toHaveBeenCalledWith("cuda"));
  });
});
