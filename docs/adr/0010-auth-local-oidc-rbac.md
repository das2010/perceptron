# ADR-0010: Autenticación local + OIDC y RBAC Admin/Editor/Viewer
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §2, §3.2, §7.16.
- Decisión: Usuarios locales (Argon2) y SSO OIDC (Entra ID, Google Workspace, genérico) con Authlib; roles Admin/Editor/Viewer por workspace y por proyecto (`Membership.project_id` opcional). En desktop standalone, token efímero de sidecar.
- Consecuencias: Implementación en Capa 5; el dominio ya modela `User`, `Workspace` y `Membership`.
- Alternativas consideradas: —
