# ADR-0038: Hardening de seguridad (Capa 7, OWASP ASVS nivel 2)
- Estado: aceptado (2026-09-28)
- Contexto: la Capa 7 pide hardening de seguridad antes de distribuir. El Team Server es
  multiusuario y expuesto en la red de la empresa. El desktop corre un Engine local con token
  efímero. Un inventario encontró, entre otras cosas:
  - rutas de cliente con letra de unidad (`C:x`) que en Windows escapan de la carpeta del proyecto;
  - conexiones a URLs cargadas por usuarios sin control de destino (SSRF);
  - tracebacks en respuestas;
  - lecturas de uploads sin tope;
  - el token del sidecar escrito en un log.
- Decisión:
  - **Rutas:** `core.paths.safe_parts` es la única forma de convertir una ruta que manda un
    cliente en componentes. Rechaza absolutas, `..`, letras de unidad y flujos de NTFS (`:`), NUL
    y nombres reservados de Windows. `ensure_within` resuelve symlinks antes de escribir o leer.
    `WorkspacePaths.project` exige un id de un solo componente. En el servidor, sin
    `PERCEPTRON_SOURCE_ROOTS` no se lee ninguna ruta propia (solo subidas), y las fuentes se
    vuelven a verificar al previsualizar e ingerir.
  - **SSRF (`core.netguard`):**
    - Aplica a las fuentes REST/WebSocket, los webhooks y las bases de datos remotas. Resuelve
      el host y, en el Team Server, rechaza direcciones que no sean públicas (loopback, RFC 1918,
      CGNAT, link-local/metadata, ULA, IPv4 mapeada).
    - `PERCEPTRON_NETWORK__ALLOWED_HOSTS` habilita hosts internos puntuales. En el desktop lo
      interno se permite por defecto.
    - No se siguen redirects y la paginación `Link` no puede cambiar de host, así que las
      credenciales no viajan a otro sitio.
    - Conectarse a otro Team Server es solo cosa del desktop.
  - **Errores:**
    - Los jobs informan tipo y mensaje; el traceback queda en el log.
    - Los 422 no devuelven el valor recibido.
    - El copiloto responde con un mensaje genérico ante errores internos.
  - **Cabeceras:** Engine y servidor ponen `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`,
    COOP y `Permissions-Policy`, y `Cache-Control: no-store` en la API. El servidor agrega la CSP
    de la SPA y HSTS con `includeSubDomains`. Swagger y `/openapi.json` quedan apagados en el
    servidor salvo `PERCEPTRON_API__DOCS=true`; el contrato se sigue generando desde el código.
  - **Límites:** los chunks de sincronización se leen en streaming con tope. Playground,
    explicación de imágenes e importación de etiquetas leen con `read_limited`. Los ZIP se
    rechazan por cantidad de archivos, tamaño descomprimido o ratio de compresión (zip bomb). Las
    subidas fallidas no dejan archivos a medias.
  - **Sesiones del servidor:**
    - La vida absoluta es de 30 días (`session_max_age_s`), además de la inactividad de 7.
    - Los WebSockets con cookie exigen `Origin` del mismo sitio (CSWSH).
    - SSO: en Entra ID se usa el UPN y no el claim `email` (nOAuth); Google y OIDC genérico
      exigen `email_verified`.
  - **Desktop:**
    - El token del sidecar no se escribe en `engine.out.log`.
    - Los logs enmascaran `?token=`, `Bearer …` y secretos en extras anidados.
    - Los comandos de keychain de la UI solo aceptan claves `ui.*`.
    - La CSP de Tauri suma `object-src 'none'`, `base-uri`, `form-action` y `frame-ancestors`.
  - **CI:**
    - Token `contents: read` por defecto.
    - `cargo audit` se suma a pip-audit y pnpm audit.
    - CodeQL (`security-extended`) para Python, TypeScript y workflows.
    - Dependabot semanal y agrupado.
    - Ruff con las reglas S cumple el rol de Bandit.
  - `torch>=2.6`: `torch.load` usa `weights_only=True` por defecto, lo que también cubre los
    checkpoints de Lightning.
- Consecuencias:
  - La verificación control por control está en `docs/security/asvs-l2.md`.
  - Riesgos aceptados y pendientes, documentados ahí:
    - MFA delegado al IdP (sin MFA en cuentas locales);
    - rate limiting en memoria por proceso;
    - sandbox del código experto sin aislamiento del sistema operativo;
    - DNS rebinding acotado, no eliminado;
    - actions fijadas por tag y no por SHA (Dependabot las mantiene).
