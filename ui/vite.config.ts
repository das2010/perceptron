import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { configDefaults } from "vitest/config";

// El mismo build sirve para Tauri (desktop) y para el Team Server (web), SPEC §4.2.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    // En desarrollo web, /api se redirige a un Engine local (`perceptron serve --port 8765`).
    proxy: { "/api": { target: "http://127.0.0.1:8765", ws: true } },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: true,
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
