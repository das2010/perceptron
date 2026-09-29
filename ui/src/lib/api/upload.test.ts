import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "./hooks";
import { postForm } from "./upload";

class FakeXhr {
  static last: FakeXhr | null = null;
  status = 0;
  responseText = "";
  withCredentials = false;
  headers: Record<string, string> = {};
  url = "";
  upload: {
    onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null;
  } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor() {
    FakeXhr.last = this;
  }
  open(_method: string, url: string) {
    this.url = url;
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }
  send() {
    this.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 200 });
    this.upload.onprogress?.({ lengthComputable: true, loaded: 200, total: 200 });
  }
  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.onload?.();
  }
}

describe("postForm (subida con progreso)", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("informa el avance y devuelve el JSON", async () => {
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    const seen: number[] = [];
    const pending = postForm<{ id: string }>("/api/v1/projects/p/uploads", new FormData(), (f) =>
      seen.push(f),
    );
    await vi.waitFor(() => expect(FakeXhr.last).not.toBeNull());
    FakeXhr.last?.respond(201, { id: "src_1" });
    await expect(pending).resolves.toEqual({ id: "src_1" });
    expect(seen).toEqual([0.25, 1]);
    expect(FakeXhr.last?.url).toContain("/api/v1/projects/p/uploads");
    expect(FakeXhr.last?.withCredentials).toBe(true);
  });

  it("convierte los errores del Engine en ApiError", async () => {
    FakeXhr.last = null;
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    const pending = postForm("/api/v1/projects/p/uploads", new FormData());
    await vi.waitFor(() => expect(FakeXhr.last).not.toBeNull());
    FakeXhr.last?.respond(413, { code: "too_large", message: "archivo demasiado grande" });
    const err = await pending.catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).message).toBe("archivo demasiado grande");
  });
});
