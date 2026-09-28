# ADR-0026: App de escritorio y runtime Python embebido (Capa 3c)
- Estado: aceptado
- Fecha: 2026-09-28
- Contexto: SPEC §2 y §4 fijan Tauri 2 con el Engine Python como sidecar en `127.0.0.1`, un puerto aleatorio y un token efímero. Los secretos van al keychain del SO, las actualizaciones son firmadas y el usuario no necesita Python instalado (§13.5). RF-TRN-02 pide instalar en el primer arranque la variante de PyTorch que corresponde al hardware (CUDA, ROCm, XPU o CPU) con uv, y poder cambiarla sin reinstalar la app. Una wheel CUDA de PyTorch pesa más de 2 GB, así que no se pueden empaquetar todas las variantes en el instalador.
- Decisión:
  - **Instalador liviano y aprovisionamiento en el primer arranque.** El instalador (NSIS/MSI en Windows, AppImage/.deb en Linux) trae:
    - la app Tauri;
    - el binario de `uv` (recurso empaquetado en CI desde el release oficial, con checksum);
    - la wheel de `perceptron-engine`;
    - `requirements.lock.txt`, exportado de `uv.lock` sin torch/torchvision/torchaudio.

    En el primer arranque, la app, desde Rust:
    1. instala Python 3.12 con `uv python install` en su directorio de datos;
    2. crea un venv e instala las dependencias fijadas y la wheel;
    3. detecta el hardware con el propio Engine (`detect_hardware`, que no requiere torch);
    4. instala torch y torchvision de la variante recomendada desde el índice oficial de PyTorch;
    5. guarda `runtime/state.json`.

    Si cambia la versión de la app, se vuelve a aprovisionar. Cambiar de variante reinstala solo torch.
  - **Mapa de variante a índice:** `cpu` → `/whl/cpu`; `cuda` → `/whl/cu130` si el driver NVIDIA es ≥ 580 y `/whl/cu126` si es ≥ 560; `rocm` → `/whl/rocm7.2` (solo Linux); `xpu` → `/whl/xpu`. El mapa vive en `desktop/runtime/torch-indexes.json`, así que se actualiza sin tocar código. Cada versión de PyTorch publica un conjunto distinto de variantes (2.14 no tiene `cu128` ni `rocm6.4`): el CI de desktop verifica con `desktop/scripts/check-torch-indexes.mjs` que todas las del mapa tengan las wheels fijadas.
  - **Sidecar.** Rust lanza `python -m perceptron.cli.main serve --new-token` con el venv del runtime y lee la línea `ready` de stdout (host, puerto, token). La UI la obtiene con el comando `engine_connection`. Al cerrar la app se mata el proceso del Engine. El Engine acepta CORS de los orígenes de Tauri: `tauri://localhost` en Linux y `http://tauri.localhost` en Windows.
  - **Puente de plataforma.** La UI detecta Tauri (`__TAURI_INTERNALS__`) y usa `TauriPlatformBridge`, que implementa `engine()`, `pickDirectory()` (plugin de diálogos) y `get/setSecret()` (crate `keyring`: Credential Manager / Secret Service). Mientras se aprovisiona, la UI muestra una pantalla de preparación con el progreso, que recibe por eventos `runtime://progress`.
  - **Updater y firma: pendientes.** Faltan las claves de firma del updater (`tauri-plugin-updater`) y un certificado de firma de código (Authenticode) de la empresa. Se incorporan en el build de release, nunca en el repo. Mientras tanto, los instaladores salen sin firmar y Windows SmartScreen los advierte.
  - **CI** (`desktop.yml`): corre en PRs que tocan `desktop/**` y en tags. Compila la app en `windows-latest` y `ubuntu-24.04`. El smoke test arranca el binario en modo `--provision-only`: aprovisiona un runtime limpio, detecta hardware, instala la variante, levanta el Engine y verifica `/system/health`. En los runners de CI la variante esperada es `cpu`.
- Consecuencias:
  - El primer arranque necesita conexión (PyPI y el índice de PyTorch) y tarda varios minutos.
  - Se puede agregar un instalador *offline* por variante más adelante, con las wheels dentro.
  - En máquinas corporativas que bloquean ejecutables en `%LOCALAPPDATA%`, el runtime se puede reubicar con la variable `PERCEPTRON_RUNTIME_DIR`; se documenta en la guía de instalación.
- Alternativas consideradas:
  - Congelar el Engine con PyInstaller: el ejecutable resultante pesa varios GB, no permite cambiar la variante de torch y dispara antivirus.
  - `python-build-standalone` empaquetado junto al venv: es lo que `uv` ya descarga, pero duplicaría el peso para las cuatro variantes.
  - Conda: licencia y tamaño.
