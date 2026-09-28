/**
 * PlatformBridge (SPEC §4.2): aísla las diferencias entre el desktop (Tauri) y la
 * UI web del Team Server. La implementación de Tauri está en `./tauri.ts` (ADR-0026).
 */
export interface EngineConnection {
  /** URL base del Engine, sin el prefijo /api/v1. */
  baseUrl: string;
  /** Token efímero del sidecar (solo desktop). */
  token?: string;
}

/** Paso del aprovisionamiento del runtime embebido (evento `runtime://progress`). */
export interface RuntimeProgress {
  step: string;
  message: string;
}

/** Estado del runtime Python embebido del desktop (RF-TRN-02). */
export interface RuntimeState {
  app_version: string;
  torch_variant: string;
  torch_index: string;
  nvidia_driver: string | null;
}

/** Operaciones que solo existen en el desktop. */
export interface DesktopRuntime {
  state(): Promise<RuntimeState | null>;
  onProgress(cb: (p: RuntimeProgress) => void): Promise<() => void>;
  /** Reinstala la variante de PyTorch (sin reinstalar la app) y reinicia el Engine. */
  setTorchVariant(variant: string): Promise<void>;
  restart(): Promise<void>;
}

/** Versión nueva publicada (`latest.json`, firmada; ADR-0037). */
export interface UpdateInfo {
  version: string;
  current_version: string;
  notes: string | null;
  date: string | null;
}

/** Progreso de la descarga (evento `updater://progress`). */
export interface UpdateProgress {
  downloaded: number;
  total: number | null;
}

/** Actualización automática del desktop. */
export interface DesktopUpdater {
  /** Consulta el endpoint de releases; null si ya está en la última versión. */
  check(): Promise<UpdateInfo | null>;
  onProgress(cb: (p: UpdateProgress) => void): Promise<() => void>;
  /** Descarga, verifica la firma, instala y reinicia la app (no vuelve si sale bien). */
  install(): Promise<void>;
}

export interface PlatformBridge {
  readonly kind: "desktop" | "web";
  readonly runtime?: DesktopRuntime;
  readonly updater?: DesktopUpdater;
  engine(): Promise<EngineConnection>;
  /** Selector nativo de carpeta; en web devuelve null (se usa subida de archivos). */
  pickDirectory(): Promise<string | null>;
  getSecret(key: string): Promise<string | null>;
  setSecret(key: string, value: string): Promise<void>;
}

export class WebPlatformBridge implements PlatformBridge {
  readonly kind = "web" as const;

  engine(): Promise<EngineConnection> {
    // En web el Engine se sirve en el mismo origen (o vía el proxy de Vite en desarrollo).
    return Promise.resolve({ baseUrl: window.location.origin });
  }

  pickDirectory(): Promise<string | null> {
    return Promise.resolve(null);
  }

  getSecret(): Promise<string | null> {
    // En web los secretos viven en el servidor (cifrados), nunca en el navegador.
    return Promise.resolve(null);
  }

  setSecret(): Promise<void> {
    return Promise.reject(new Error("Secrets are managed by the Team Server in web mode"));
  }
}

let current: PlatformBridge = new WebPlatformBridge();

export function getPlatform(): PlatformBridge {
  return current;
}

export function setPlatform(bridge: PlatformBridge): void {
  current = bridge;
}

/** La UI corre dentro de Tauri (webview del desktop). */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
