# OWASP ASVS 4.0.3 nivel 2: verificación del Team Server y del Engine

Revisión de la Capa 7 (ADR-0038). En cada capítulo figuran los controles aplicables, dónde están implementados y qué test los verifica. Los N/A y los riesgos aceptados se justifican al final.

Estados:
- ✅ cumple;
- 🟡 cumple parcialmente o con riesgo aceptado;
- N/A no aplica.

## V1 Arquitectura
| Control | Estado | Implementación |
|---|---|---|
| 1.1 Modelo de amenazas y decisiones | ✅ | SPEC §13, ADR-0030 (auth), ADR-0038 (hardening) |
| 1.4 Control de acceso centralizado | ✅ | `ServerAccess.authorize` sobre cada operación del Engine (`policy.py`); tests `test_server_rbac.py` |
| 1.5 Validación en el servidor | ✅ | Modelos Pydantic con `extra="forbid"`; validación de dominio en servicios |
| 1.14 Separación de componentes | ✅ | Workers de entrenamiento como subprocesos; cola con Valkey; sandbox para el código experto |

## V2 Autenticación
| Control | Estado | Implementación |
|---|---|---|
| 2.1.1 Contraseñas de 12 caracteres o más | ✅ | `min_password_length=12` (`accounts.py`) |
| 2.1.2 Largo máximo 128 o más | ✅ | Tope de 256 |
| 2.1.7 Contraseñas filtradas o comunes | 🟡 | Sin chequeo contra listas de contraseñas filtradas (pendiente) |
| 2.2.1 Anti-automatización | ✅ | Rate limit por IP en login, token y refresh; bloqueo por cuenta (5 fallos / 5 min); `test_server_auth.py` |
| 2.2.3 MFA | 🟡 | Se delega al IdP (SSO Entra ID / Google); las cuentas locales no tienen MFA |
| 2.4.1 Hash resistente | ✅ | Argon2id (argon2-cffi) con rehash al iniciar sesión |
| 2.5 Recuperación | N/A | No hay recuperación por email: la contraseña la restablece un Admin |
| 2.10 Secretos de servicio | ✅ | `secret_key` de 32 caracteres o más; claves LLM cifradas (AES-GCM) o en el keychain |

## V3 Sesiones
| Control | Estado | Implementación |
|---|---|---|
| 3.2.1 Token nuevo al autenticar | ✅ | Familia de sesión nueva por login |
| 3.2.3 Almacenamiento seguro | ✅ | Cookies `HttpOnly; Secure; SameSite=Strict` |
| 3.3.1 Logout invalida la sesión | ✅ | `revoke_session`; `test_logout_and_password_change_close_sessions` |
| 3.3.2 Re-autenticación periódica | ✅ | Inactividad de 7 días y vida absoluta de 30 (`session_max_age_s`); `test_sessions_have_an_absolute_lifetime` |
| 3.3.3 Cerrar las otras sesiones al cambiar la contraseña | ✅ | Se revocan todas |
| 3.5.2 Tokens firmados con expiración | ✅ | JWT HS256 de 15 minutos con `iss`, `typ` y `sid`; refresh rotativo con detección de reuso |
| 3.5.3 Tokens con estado | ✅ | Cada request revalida la sesión en la base |

## V4 Control de acceso
| Control | Estado | Implementación |
|---|---|---|
| 4.1.1 Aplicado en el servidor | ✅ | Middleware de sesión + `ServerAccess` |
| 4.1.3 Mínimo privilegio | ✅ | Roles Viewer, Editor y Admin por workspace/proyecto; Admin del servidor para la configuración global |
| 4.2.1 IDOR | ✅ | El proyecto se deduce de cada `*_id`; en sync, un id ajeno no se puede pisar (`test_sync_rejects_foreign_ids_and_stale_versions`) |
| 4.2.2 CSRF | ✅ | Doble envío `X-CSRF-Token` en escrituras con cookie; WebSockets con `Origin` verificado (`test_websocket_from_another_site_cannot_use_the_session_cookie`) |
| 4.3.1 Interfaces administrativas | ✅ | `/admin/*` y las operaciones `SERVER_ADMIN_OPS` |

## V5 Validación, sanitización y codificación
| Control | Estado | Implementación |
|---|---|---|
| 5.1.3 Validación de entradas | ✅ | Pydantic en todos los cuerpos; ids con formato en sync |
| 5.2.6 SSRF | ✅ | `core/netguard.py`: resolución y bloqueo de direcciones internas en el servidor, sin redirects, paginación sin cambio de host; `test_netguard.py`, `test_security_api.py` |
| 5.3.3 Salida en HTML | ✅ | React escapa por defecto; sin `dangerouslySetInnerHTML` con datos del usuario |
| 5.3.4 SQL | ✅ | SQLAlchemy con parámetros; las consultas de fuentes DB corren en una transacción descartada |
| 5.3.9 Path traversal | ✅ | `safe_parts`/`ensure_within`, incluidos los casos de Windows (`C:x`, `con`, `:`); `test_safe_paths.py` |
| 5.5.1 Deserialización | ✅ | Sin pickle; `torch.load(weights_only=True)` y `torch>=2.6`; `yaml.safe_load` |
| 5.5.2 XXE | ✅ | Las anotaciones VOC se leen con `xml.etree.ElementTree` de la stdlib: no resuelve entidades externas ni DTD remotas, y el expat que trae Python 3.12 (2.4 o posterior) limita la expansión de entidades |

## V7 Errores y logs
| Control | Estado | Implementación |
|---|---|---|
| 7.1.1 Sin credenciales en logs | ✅ | `redact`/`scrub` (`?token=`, `Bearer`, extras anidados); el token del sidecar no va a `engine.out.log`; `test_core.py` |
| 7.2 Auditoría | ✅ | Logins, escrituras, denegaciones, lecturas de datos y descargas (`audit.py`) |
| 7.4.1 Mensajes genéricos | ✅ | Jobs sin traceback; 422 sin el valor recibido; copiloto con error genérico; `test_security_api.py` |

## V8 Protección de datos
| Control | Estado | Implementación |
|---|---|---|
| 8.1.1 Sin caché de datos sensibles | ✅ | `Cache-Control: no-store` en toda la API |
| 8.3.1 Datos sensibles fuera de la URL | 🟡 | El token del sidecar viaja en `?token=` de los WebSockets locales (no hay cabeceras en el navegador) y se enmascara en los logs |
| 8.3.4 Datos al LLM | ✅ | `PrivacyFilter` L0–L3 y auditoría de cada llamada (ADR-0007) |

## V9 Comunicaciones
| Control | Estado | Implementación |
|---|---|---|
| 9.1.1 TLS | ✅ | Detrás de un proxy con TLS; HSTS cuando `COOKIE_SECURE=true` |
| 9.2.1 TLS saliente | 🟡 | SQL Server usa `TrustServerCertificate=yes` por compatibilidad con instalaciones corporativas |

## V10 Código malicioso
| Control | Estado | Implementación |
|---|---|---|
| 10.3.1 Actualizaciones firmadas | ✅ | Updater de Tauri con minisign (ADR-0037) |
| 10.3.2 Integridad de dependencias | ✅ | `uv.lock` y `pnpm-lock.yaml` con hashes; Dependabot |

## V12 Archivos
| Control | Estado | Implementación |
|---|---|---|
| 12.1.1 Tamaño máximo | ✅ | 10 GB por subida; chunks de 16 MB; `read_limited` en playground, explicación y etiquetas |
| 12.1.2 Archivos comprimidos | ✅ | `check_zip_limits`: cantidad, tamaño total y ratio; zip slip |
| 12.3.1 Nombres de archivo del usuario | ✅ | `safe_parts` |
| 12.5.1 Servir solo lo permitido | ✅ | Descargas por lista de archivos del informe; sync con prefijos permitidos; sin symlinks |
| 12.6.1 SSRF | ✅ | Ver 5.2.6 |

## V13 API
| Control | Estado | Implementación |
|---|---|---|
| 13.1.3 Sin secretos en la URL | 🟡 | Ver 8.3.1 |
| 13.2.1 Métodos habilitados | ✅ | CORS con métodos explícitos; en el servidor, sin CORS por defecto |
| 13.2.3 CSRF en APIs con cookie | ✅ | Ver 4.2.2 |
| 13.5 WebSockets | ✅ | Autorización por operación y `Origin` verificado |

## V14 Configuración
| Control | Estado | Implementación |
|---|---|---|
| 14.1 Build y despliegue | ✅ | CI con pip-audit, pnpm audit, cargo audit, CodeQL, licencias; token del CI `contents: read` |
| 14.2.1 Componentes sin vulnerabilidades conocidas | ✅ | Audits en cada PR |
| 14.3.2 Sin modo debug en producción | ✅ | Swagger y `/openapi.json` apagados en el servidor |
| 14.4 Cabeceras HTTP | ✅ | CSP de la SPA, `nosniff`, `X-Frame-Options`, `Referrer-Policy`, COOP, `Permissions-Policy`, HSTS; `test_server_platform.py` |
| 14.5.3 CORS | ✅ | Lista explícita; vacía en el servidor |

## Riesgos aceptados y pendientes
- **MFA en cuentas locales:** se recomienda SSO (Entra ID o Google) con MFA en el IdP y `password_login=false`.
- **Contraseñas filtradas:** sin chequeo offline contra listas; pendiente.
- **Rate limiting:** en memoria y por proceso. Con varias réplicas conviene limitar también en el proxy.
- **Sandbox del código experto:** guardas estáticas y de runtime en un subproceso, sin aislamiento del sistema operativo. En el servidor, el modo experto conviene habilitarlo solo para usuarios de confianza.
- **DNS rebinding:** se verifica antes de cada conexión, lo que acota la ventana entre la resolución y la conexión sin eliminarla.
- **Actions fijadas por tag:** Dependabot las actualiza; fijarlas por SHA queda como mejora.
