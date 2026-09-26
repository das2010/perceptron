/**
 * PlatformBridge (SPEC §4.2): aísla las diferencias entre el desktop (Tauri) y la
 * UI web del Team Server. La implementación Tauri se agrega en Capa 3.
 */
export interface EngineConnection {
  /** URL base del Engine, sin el prefijo /api/v1. */
  baseUrl: string;
  /** Token efímero del sidecar (solo desktop). */
  token?: string;
}

export interface PlatformBridge {
  readonly kind: "desktop" | "web";
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
