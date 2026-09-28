# Plan — Capa 5 (Team Server)

SPEC §14:
- servidor FastAPI con PostgreSQL, S3, MLflow server y Redis + workers GPU/CPU;
- auth local + OIDC y RBAC;
- sync de proyectos de equipo;
- envío de runs desde el desktop;
- modo estación de trabajo (UI web);
- consola de administración;
- políticas de LLM y privacidad, cuotas y auditoría;
- Docker Compose y Helm.

**Aceptación §14:**
- dos usuarios con roles distintos colaboran en un proyecto;
- un desktop sin GPU lanza un run en el worker GPU del servidor y ve el progreso en vivo;
- un usuario completa UC-07 solo desde el navegador;
- login SSO con Entra ID de prueba.

Son tres sub-capas con un PR cada una. En CI se levanta el stack con Docker Compose (PostgreSQL, S3, Redis, MLflow).

## 5a — Núcleo del servidor, auth local y RBAC
- **Almacenamiento:** `perceptron_server` monta el mismo `create_app` del Engine sobre PostgreSQL 16 (SQLAlchemy + Alembic sobre el almacén documental actual). El workspace del servidor queda en un volumen; el object storage S3 pasa a la 5b, con los workers (ADR-0030, propuesta de D2).
- **Auth local** (RF-SRV-01): Argon2 (argon2-cffi, MIT), sesiones JWT cortas con refresh, CSRF para la UI web y rate limiting en `/auth`.
- **RBAC** (RF-SRV-02): Admin/Editor/Viewer por workspace y por proyecto (`Membership`), aplicado como dependencia de FastAPI en cada router.
- **Auditoría** (RF-SRV-07): login, acceso a datasets, exportaciones, llamadas LLM y cambios de permisos.
- **Modo estación de trabajo** (RF-SRV-05): el servidor sirve la SPA; `WebPlatformBridge` con login; "fuentes del servidor", es decir rutas montadas que habilita el Admin.
- **Aceptación parcial:** dos usuarios (Editor y Viewer) en el mismo proyecto, con permisos verificados por la API y E2E web; UC-07 desde el navegador.

## 5b — Cola de jobs, workers y MLflow server (ADR-0031)
- **Cola:** Celery (BSD) sobre Valkey (BSD; Redis ≥ 7.4 no es permisivo), con colas `gpu` y `cpu` y cuotas por usuario y workspace (RF-SRV-04). Los workers comparten PostgreSQL y el volumen del workspace.
- **Progreso:** relay pub/sub; los mismos WS del Engine.
- **MLflow server:** backend PostgreSQL y artefactos proxied en volumen. El S3 queda para la 5c.
- **Imágenes:** CPU y GPU (`TORCH_VARIANT`) para server y worker; Compose con perfil `gpu`.

## 5c — Sync, administración y SSO
- **Sync desktop ↔ servidor** (RF-SRV-03): el servidor es la fuente de verdad; bloqueo optimista por versión; proyectos de equipo.
- **Envío desde el desktop:** "entrenar en el servidor" sube la versión de datos (resumible por chunks), crea el estudio remoto en la cola de la 5b y sigue el progreso por el WS del servidor (aceptación: desktop sin GPU → worker GPU).
- **Object storage** (D2, ADR-0030): datasets y artefactos por hash en S3 estándar (SeaweedFS en Compose) para workers sin volumen compartido.
- **Consola de administración** (RF-SRV-06): usuarios, grupos, SSO, proveedores LLM, políticas de privacidad, cuotas, almacenamiento, workers y auditoría.
- **OIDC con Authlib** (Entra ID, Google Workspace, genérico), con mapeo de grupos a roles.
- **Helm chart v1** (servidor, workers CPU/GPU, Valkey, MLflow; PostgreSQL y S3 externos).
- **Backups** (RF-SRV-08): scripts y guía para PostgreSQL y el object storage.

## Lo que necesito del usuario
1. **Entra ID de prueba** para la aceptación del SSO: un tenant o una app registration de prueba (client id, tenant id y secret como secrets del repo) y un usuario de prueba.
2. **Worker GPU** para la aceptación "desktop sin GPU → worker GPU". Opciones:
   - un runner self-hosted con GPU;
   - un runner GPU de GitHub (pago);
   - aceptar la prueba en CPU y validar la GPU a mano en su infraestructura.
3. Dónde desplegar el Team Server de prueba, si hace falta más allá del Compose en CI.
