# Publicar una versión del desktop (uso interno de Preteco)

Las actualizaciones automáticas están firmadas con la clave del updater de Tauri (ADR-0037). La app instalada consulta `latest.json` en el último release de GitHub, verifica la firma e instala.

## Una sola vez: los secrets

En GitHub, en *Settings → Secrets and variables → Actions*, se cargan:

| Secret | Contenido |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | El contenido completo de `perceptron-updater.key` (desde la bóveda de Preteco) |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | La contraseña de esa clave |

La clave pública ya está en `desktop/src-tauri/tauri.conf.json` (`plugins.updater.pubkey`). Si se pierde la clave privada o su contraseña, las instalaciones existentes dejan de recibir actualizaciones automáticas.

## Cada versión

1. Subir la versión en `desktop/src-tauri/tauri.conf.json` (y en `Cargo.toml` y `desktop/package.json` para mantenerlas iguales), por PR a `main`.
2. Opcional: escribir las notas en `docs/releases/vX.Y.Z.md`. Son las que la app muestra antes de instalar.
3. Crear y subir el tag:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
4. El workflow **Desktop** compila, firma, prueba la actualización N → N+1 y publica el release con los instaladores, los `.sig` y `latest.json`.

El workflow falla si el tag no coincide con la versión de `tauri.conf.json` o si faltan los secrets.

## Qué recibe el usuario
- Al abrir la app, un aviso en el encabezado si hay versión nueva.
- En *Configuración → Actualizaciones*, las notas y el botón **Instalar y reiniciar**. Sus proyectos y datos no se modifican.
- En Windows se actualiza el instalador NSIS y en Linux el AppImage. El `.deb` se actualiza instalando el paquete nuevo.

## Instaladores sin firma Authenticode
Mientras no haya certificado de Preteco, Windows muestra "editor desconocido" al instalar por primera vez. Las actualizaciones automáticas no dependen de Authenticode: las valida la firma del updater.
