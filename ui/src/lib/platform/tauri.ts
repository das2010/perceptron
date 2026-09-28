/** PlatformBridge del desktop (Tauri 2, ADR-0026): sidecar, keychain y diálogos nativos. */
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

import type {
  DesktopRuntime,
  EngineConnection,
  PlatformBridge,
  RuntimeProgress,
  RuntimeState,
} from "./bridge";

const runtime: DesktopRuntime = {
  state: () => invoke<RuntimeState | null>("runtime_state"),
  onProgress: (cb) => listen<RuntimeProgress>("runtime://progress", (e) => cb(e.payload)),
  setTorchVariant: async (variant) => {
    await invoke("set_torch_variant", { variant });
  },
  restart: async () => {
    await invoke("restart_engine");
  },
};

export class TauriPlatformBridge implements PlatformBridge {
  readonly kind = "desktop" as const;
  readonly runtime = runtime;

  /** Espera a que el runtime esté listo y el Engine haya emitido su línea `ready`. */
  engine(): Promise<EngineConnection> {
    return invoke<EngineConnection>("engine_connection");
  }

  async pickDirectory(): Promise<string | null> {
    const picked = await open({ directory: true, multiple: false });
    return typeof picked === "string" ? picked : null;
  }

  getSecret(key: string): Promise<string | null> {
    return invoke<string | null>("get_secret", { key });
  }

  setSecret(key: string, value: string): Promise<void> {
    return invoke("set_secret", { key, value });
  }
}
