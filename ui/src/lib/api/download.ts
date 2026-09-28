import { TOKEN_HEADER } from "@/lib/api/client";
import { getPlatform } from "@/lib/platform/bridge";

/** Descarga un archivo del Engine (con el token del sidecar) y lo guarda con `filename`. */
export async function downloadFromEngine(path: string, filename: string): Promise<void> {
  const { baseUrl, token } = await getPlatform().engine();
  const res = await fetch(new URL(path, baseUrl), {
    headers: token ? { [TOKEN_HEADER]: token } : {},
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { message?: string };
    throw new Error(body.message ?? `HTTP ${res.status}`);
  }
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
