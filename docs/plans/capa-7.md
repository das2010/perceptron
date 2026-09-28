# Plan — Capa 7 (Empaquetado comercial)

SPEC §14:
- hooks de licenciamiento (§7.17) y feature flags;
- auto-update firmado y firma de código de instaladores (Windows Authenticode);
- telemetría opcional opt-in;
- documentación de usuario es/en;
- auditoría de licencias de dependencias y de pesos preentrenados;
- hardening de seguridad.

**Aceptación:**
- informe de licencias sin dependencias incompatibles con distribución comercial;
- instaladores firmados;
- actualización de versión N a N+1 sin pérdida de proyectos.

## Lo que se puede hacer sin recursos externos
- **Auditoría de licencias** (CI):
  - inventario de dependencias Python (uv), JS (pnpm) y Rust (cargo) con su licencia, más una política de permitidas/revisar/prohibidas (GPL/AGPL/SSPL en el runtime);
  - informe HTML/Markdown como artefacto;
  - el job falla si aparece algo prohibido.
  - **Pesos preentrenados:** el catálogo ya declara `commercial_ok`; se suma al mismo informe.
- **Licenciamiento v1 sin enforcement** (RF-LIC-03):
  - `LicenseProvider` con archivo de licencia firmado **Ed25519** validable offline (formato, verificación y la pantalla «Licencia» en la UI);
  - el enforcement queda apagado por flag;
  - `features.yaml` es la única fuente de las claves.
- **Telemetría opt-in** (D7):
  - apagada por defecto, con consentimiento explícito en el primer arranque y en Configuración;
  - contenido mínimo, sin datos ni nombres: versión, SO, hardware agregado, uso de features por contador y errores sin trazas de datos;
  - endpoint configurable;
  - en CI, cero envíos si no hay consentimiento.
- **Upgrade N → N+1:**
  - migraciones versionadas también para el workspace del desktop (hoy `create_all`), en SQLite con Alembic o un `schema_version`;
  - test de CI: se crea un workspace con el último release, se abre con la versión nueva y se verifican proyectos, datos, runs y modelos.
- **Hardening** (hecho en la 7c, ADR-0038 y `docs/security/asvs-l2.md`):
  - revisión con checklist OWASP ASVS nivel 2 para el servidor;
  - `bandit`/`pip-audit`/`pnpm audit`/`cargo audit` en CI;
  - CSP de Tauri, límites de upload y cabeceras.
- **Documentación de usuario es/en:** guía por caso de uso (UC-01…UC-10), instalación desktop y servidor, administración y privacidad. Se publica como sitio estático (MkDocs Material, MIT).

## Lo que necesito de vos
1. ~~D1~~ **Decidido** (2026-09-28): archivo firmado Ed25519 offline con topes de usuarios, servidores y GPUs (ADR-0035). Falta que Preteco genere su par de claves.
2. **Firma de código de Windows** (postergada por decisión del usuario, 2026-09-28): un certificado Authenticode de Preteco (OV/EV, idealmente en un HSM/servicio de firma en la nube) y el mecanismo para usarlo desde CI (Azure Trusted Signing, SignPath, o un runner con el token).
3. ~~Auto-update~~ **Decidido** (2026-09-28): claves del updater generadas por Preteco y releases en GitHub Releases del repo público (ADR-0037). Faltan los secrets `TAURI_SIGNING_PRIVATE_KEY` y `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
4. **D7, telemetría:** ¿se quiere? ¿A qué endpoint? ¿Qué contenido aprueba Preteco?
