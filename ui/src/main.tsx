import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@/styles/global.css";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { AuthGate } from "@/features/auth/AuthGate";
import { RuntimeGate } from "@/features/desktop/RuntimeGate";
import { isTauri, setPlatform } from "@/lib/platform/bridge";

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");

async function boot(el: HTMLElement) {
  // En el desktop, el Engine es un sidecar local con token efímero (ADR-0026).
  if (isTauri()) {
    const { TauriPlatformBridge } = await import("@/lib/platform/tauri");
    setPlatform(new TauriPlatformBridge());
  }
  createRoot(el).render(
    <StrictMode>
      <Providers>
        <RuntimeGate>
          <AuthGate>
            <App />
          </AuthGate>
        </RuntimeGate>
      </Providers>
    </StrictMode>,
  );
}

void boot(root);
