# Plan — Capa 3 (UI y Desktop)

SPEC §14: app React con tema Preteco (claro/oscuro), i18n es/en, pantallas de §11.2, wizard
completo con copiloto, editor visual de pipeline y de ArchSpec con sub-wizard de definición,
editor de código (Monaco), dashboards de entrenamiento en vivo, comparación de runs, vista de
agente; Tauri 2 con sidecar, keychain y runtime Python embebido con selección de la variante
de PyTorch; instaladores Windows y Linux.

**Aceptación §14:** E2E Playwright de UC-01 y UC-04 guiado sin código en < 15 min (O1);
instalador limpio en Windows 11 y Ubuntu 24.04 detecta hardware e instala la variante de
PyTorch correcta.

Se divide en tres sub-capas con un PR cada una, como las Capas 1 y 2. El frontend se valida
también localmente (node/pnpm funcionan en la máquina de desarrollo); el Engine, en CI.

## 3a — Shell, sistema de diseño y recorrido mínimo

Objetivo: ver de punta a punta un proyecto desde la UI antes de los editores.

- **Stack de §5.2:** TanStack Router, Zustand, Tailwind + componentes estilo shadcn/ui
  (Radix) tematizados con `tokens.css`, Apache ECharts, React Hook Form + Zod. Licencias
  MIT/Apache-2.0 (ADR-0024).
- **Layout §11.2:** sidebar de proyectos, área principal, panel de copiloto plegable
  (visible, se activa en 3b). Tema claro/oscuro con contraste WCAG AA; todo lo que viene del
  LLM con acento violeta + ícono y botones aceptar/rechazar.
- **Pantallas:**
  - Inicio: proyectos recientes, hardware (`/system/hardware`), estado del Engine.
  - Proyecto → Resumen, Datos (subida + ingesta, ProfileCard con alertas), Entrenar (pipeline
    propuesto → propuestas de arquitectura (reglas/LLM) → estrategia de HPO → lanzar),
    Experimentos (runs, curvas en vivo por WebSocket, comparación), Evaluación (métricas,
    matriz de confusión, ROC/PR), Modelos (registrados).
  - Configuración global: LLM (proveedores, clave write-only, perfiles, prueba de conexión,
    auditoría), idioma, tema.
- **Engine:** `POST /projects/{id}/uploads` (multipart) para ingerir desde el navegador
  (en desktop además se puede elegir una ruta local); resto de endpoints ya existen.
- **E2E:** Playwright en CI con el Engine real (`perceptron serve`) + Vite; UC-01 guiado
  (crear proyecto → subir CSV → perfil → entrenar con reglas → ver evaluación).

## 3b — Wizard, copiloto, agente y editores

- `ProjectDraft` versionado (RF-WIZ-04) + `GET/PATCH /projects/{id}/draft`; wizard de 9 pasos
  (RF-WIZ-01..03) con "¿Por qué?" y fallback por reglas.
- Copiloto: `WS /projects/{id}/copilot` con `Gateway.stream_chat` (RF-LLM-05) y `DraftPatch`
  estructurado (aceptar/rechazar).
- Vista de agente: lanzar con límites y aprobaciones, bitácora en vivo, aprobar/rechazar.
- Editor visual de pipeline (React Flow, RF-PIP-02) con vista previa por nodo.
- Editor visual de ArchSpec (React Flow, RF-ARC-05) con validación en vivo y resumen de
  parámetros; sub-wizard de definición (familia → backbone → cabeza → regularización) con el
  rol "guía de definición"; "ver como código" (RF-ARC-07) en Monaco.
- E2E: UC-04 guiado por el wizard.

## 3b′ — Modo experto (RF-ARC-06)

Salió de la 3b en un PR propio por ser un cambio de seguridad (ADR-0025):
- código Python con la interfaz `build_model(config)`;
- validación estática (AST) y dinámica en el sandbox;
- entrenamiento y evaluación dentro del sandbox;
- editor Monaco con lint en vivo y confirmación explícita.

También incluye la comparación de runs.

## 3c — Desktop (Tauri 2)

Diseño en ADR-0026.
- `desktop/src-tauri`:
  - sidecar `serve --new-token` (puerto aleatorio, token efímero, handshake `ready`);
  - keychain con el crate `keyring`;
  - selector de carpetas (plugin de diálogos);
  - modo `--provision-only` para el smoke test.

  El updater firmado y la firma de código quedan pendientes de las claves y el certificado de la empresa.
- Runtime embebido: uv instala Python 3.12, las dependencias fijadas sin torch y la wheel del Engine. Después detecta el hardware con el Engine e instala la variante de PyTorch desde su índice (RF-TRN-02). La variante se cambia desde Configuración sin reinstalar la app.
- CI `desktop.yml`: instaladores NSIS/MSI y .deb/AppImage en Windows y Ubuntu 24.04, con instalación limpia y aprovisionamiento desde cero.

## Riesgos

- La política de seguridad de la máquina de desarrollo bloquea el Python del venv: el
  runtime embebido del desktop podría estar bloqueado también. Se valida en CI y se documenta
  qué habilitar (ruta del runtime) para usarlo en equipos corporativos.
- Minutos de Actions (repo privado): el E2E de UI corre en Linux; los instaladores, solo en
  PRs de 3c y en tags.
