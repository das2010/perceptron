# ADR-0035: Licencia en archivo firmado Ed25519, offline (D1, Capa 7)
- Estado: aceptado (decisión del usuario, 2026-09-28)
- Contexto: D1 (SPEC §17) quedaba abierto; RF-LIC-01..03 piden hooks de licenciamiento que
  permitan asiento, servidor, GPU, suscripción u archivo firmado, sin enforcement en v1. El
  producto se instala on-premise y en desktops que pueden estar sin internet.
- Decisión:
  - **Archivo `license.json` firmado con Ed25519** (cryptography, Apache-2.0/BSD) y validable
    offline: `license` (id, titular, edición, topes `seats`/`servers`/`gpus`, features o `*`,
    emisión, `not_before`, `expires_at` opcional), `key_id` y firma sobre el JSON canónico.
  - Claves públicas confiables empaquetadas en `perceptron/licensing/keys/<key_id>.pub` (más
    `PERCEPTRON_LICENSE__PUBLIC_KEYS`); la privada queda en el gestor de secretos de Preteco y
    se usa con `perceptron license issue`. Rotación: agregar un `.pub` nuevo.
  - `SignedLicenseProvider` reemplaza al de desarrollo al crear el contexto. **Sin enforcement
    en v1** (`PERCEPTRON_LICENSE__ENFORCE=false`): todo queda habilitado y el estado se informa
    (válida, sin licencia, vencida, inválida, no reconocida, todavía no vigente). Con
    enforcement, las features salen de la licencia.
  - Topes: el Team Server informa el uso real (usuarios activos, GPUs de los workers, una
    instalación) y marca los excedidos (`/admin/license`), sin bloquear en v1.
  - Instalación por la UI (Configuración → Licencia) o `PUT /system/license` (Admin del
    servidor en el Team Server); se rechaza si la firma o la vigencia no son válidas.
- Consecuencias: pendiente de Preteco generar su par de claves (`perceptron license keygen`) y
  agregar la pública al paquete antes del primer release comercial.
