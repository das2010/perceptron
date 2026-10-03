import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { reconnectDelay, useEngineSocket } from "./ws";

class FakeSocket {
  static all: FakeSocket[] = [];
  onopen: (() => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  closedByClient = false;
  constructor(public url: string) {
    FakeSocket.all.push(this);
  }
  close() {
    this.closedByClient = true;
  }
}

const flush = () => act(async () => await Promise.resolve());

describe("useEngineSocket (reconexión)", () => {
  beforeEach(() => {
    FakeSocket.all = [];
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeSocket);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("se reconecta si la conexión se corta y entrega los mensajes", async () => {
    const seen: unknown[] = [];
    const { result, unmount } = renderHook(() =>
      useEngineSocket<{ n: number }>("/api/v1/ws/jobs/job_1", (m) => seen.push(m)),
    );
    await flush();
    expect(FakeSocket.all).toHaveLength(1);
    act(() => FakeSocket.all[0]?.onopen?.());
    expect(result.current).toBe(true);
    act(() => FakeSocket.all[0]?.onclose?.({ code: 1006 })); // corte de red
    expect(result.current).toBe(false);
    await act(async () => {
      vi.advanceTimersByTime(reconnectDelay(0));
      await Promise.resolve();
    });
    expect(FakeSocket.all).toHaveLength(2);
    act(() => FakeSocket.all[1]?.onmessage?.({ data: JSON.stringify({ n: 7 }) }));
    expect(seen).toEqual([{ n: 7 }]);
    unmount();
    expect(FakeSocket.all[1]?.closedByClient).toBe(true);
  });

  it("no se reconecta con un cierre normal ni después de desmontar", async () => {
    const { unmount } = renderHook(() => useEngineSocket("/api/v1/ws/jobs/job_1", () => {}));
    await flush();
    act(() => FakeSocket.all[0]?.onclose?.({ code: 1000 }));
    await act(async () => {
      vi.advanceTimersByTime(60_000);
      await Promise.resolve();
    });
    expect(FakeSocket.all).toHaveLength(1);
    unmount();
  });

  it("la espera crece hasta 15 s", () => {
    expect([0, 1, 2, 3, 4, 10].map(reconnectDelay)).toEqual([1000, 2000, 4000, 8000, 15000, 15000]);
  });
});
