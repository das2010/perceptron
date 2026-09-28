# Perceptron Desktop (Tauri 2)

Shell de escritorio con el Engine Python como sidecar (SPEC §4.2, ADR-0026).

- `src-tauri/`: app Rust.
  - `runtime.rs`: aprovisionamiento del runtime Python con uv y la variante de PyTorch por hardware.
  - `engine.rs`: sidecar `perceptron serve --new-token` y handshake `ready`.
  - `secrets.rs`: keychain del SO.
  - `lib.rs`: comandos para la UI y modo `--provision-only`.
- `runtime/`: `prepare.mjs` arma los recursos del instalador en `src-tauri/resources/`: uv, la wheel del Engine, las dependencias fijadas, las versiones de torch y `torch-indexes.json`.

## Desarrollo

```bash
pnpm -C desktop tauri dev
```

En debug, la app usa el Engine del repo (`uv run perceptron serve --new-token`) en lugar del runtime embebido. Para probar el runtime embebido en debug, definí `PERCEPTRON_EMBEDDED_RUNTIME=1` y corré antes `pnpm -C desktop prepare:runtime`.

## Instaladores

```bash
pnpm -C desktop prepare:runtime
pnpm -C desktop tauri build
```

Genera NSIS/MSI (Windows) y .deb/AppImage (Linux). El workflow `desktop.yml` los construye y prueba desde cero: instalación silenciosa y luego `Perceptron --provision-only --report <archivo>`, que aprovisiona, levanta el Engine y verifica `/system/health`.

## Runtime

El primer arranque necesita conexión (PyPI y el índice de PyTorch) y tarda unos minutos. El runtime vive en el directorio de datos de la app (`com.preteco.perceptron/runtime`). Se puede reubicar con `PERCEPTRON_RUNTIME_DIR`, por ejemplo en equipos corporativos que bloquean ejecutables en `%LOCALAPPDATA%`. La variante de PyTorch se cambia desde Configuración → Motor local sin reinstalar la app.

El ícono de `assets/icon-source.png` es provisorio: se reemplaza por el logo oficial que provea Marketing (SPEC §12).
