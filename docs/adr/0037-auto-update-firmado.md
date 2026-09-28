# ADR-0037: Auto-update firmado del desktop con GitHub Releases (Capa 7)
- Estado: aceptado (2026-09-28; claves generadas por Preteco)
- Contexto: SPEC §3 fija el Tauri updater firmado y la aceptación de la Capa 7 pide pasar de la
  versión N a la N+1 sin perder proyectos. El repositorio es público, así que GitHub Releases
  puede servir los paquetes sin hosting propio. La firma Authenticode es otro tema: depende de
  un certificado de Preteco y queda para más adelante.
- Decisión:
  - **`tauri-plugin-updater` (MIT/Apache-2.0)** con firma minisign. La clave pública
    (`AB4C4AB9007E7CA5`) va en `tauri.conf.json`. La privada, cifrada con contraseña, vive en la
    bóveda de Preteco y como secrets del repo (`TAURI_SIGNING_PRIVATE_KEY`,
    `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`), porque es el CI el que firma cada paquete.
  - **Endpoint** `https://github.com/das2010/perceptron/releases/latest/download/latest.json`.
    Un tag `vX.Y.Z` (igual a la versión de `tauri.conf.json`) arma el release con los
    instaladores, sus `.sig` y `latest.json` (`desktop/scripts/latest-json.mjs`): Windows usa
    el NSIS y Linux el AppImage (el `.deb` se actualiza con el gestor de paquetes).
  - **Flujo en la app:** al abrir se consulta una vez y, si hay versión nueva, se muestra un aviso
    en el encabezado. En *Configuración → Actualizaciones* se ven las notas y se instala: descarga
    con progreso, verificación de la firma, Engine detenido, instalación en modo pasivo y
    reinicio. Nada se instala sin que el usuario lo pida.
  - **Datos a salvo:** los proyectos viven en el workspace del usuario y el runtime Python se
    reaprovisiona solo cuando cambia la versión de la app. El instalador en modo `/UPDATE` no
    borra datos.
  - Los PRs sin el secret generan instaladores sin paquetes de actualización (aviso). En un tag,
    falta de secret = error.
- Verificación: el job `update` de `desktop.yml` compila N y N+1 con una **clave efímera** y un
  endpoint local, instala N, crea un proyecto, actualiza con `--update-only` (el updater real:
  consulta, descarga, verifica la firma e instala) y comprueba la versión registrada, el runtime
  de N+1 y que el proyecto sigue. Complementa a `scripts/upgrade_check.py` (compatibilidad del
  workspace).
- Consecuencias: si se pierde la clave privada o su contraseña, las instalaciones existentes no
  pueden recibir más actualizaciones automáticas y habría que reinstalar a mano con una clave
  nueva. En Windows, `currentUser` instala en `%LOCALAPPDATA%\Perceptron`, que también es el
  workspace por defecto. La prueba N → N+1 verifica que la actualización no toca los proyectos,
  pero separar ambas carpetas queda como mejora antes del primer release público.
