# SSO del Team Server (RF-SRV-01)

Perceptron usa OIDC *authorization code* con PKCE. La MFA y las políticas de acceso condicional quedan del lado del IdP.

Los proveedores se configuran en `PERCEPTRON_SERVER__OIDC`, un JSON con un objeto por proveedor, y la URL pública va en `PERCEPTRON_SERVER__PUBLIC_URL`.

## Microsoft Entra ID

1. **Crear la app.** En *Entra ID → App registrations → New registration*:
   - *Supported account types*: solo este directorio.
   - *Redirect URI (Web)*: `https://<servidor>/api/v1/auth/oidc/entra/callback`.
2. **Crear el secreto.** En *Certificates & secrets → New client secret*, copiá el valor.
3. **Agregar los grupos al token** (opcional, para mapear roles). En *Token configuration → Add groups claim* elegí «Security groups», con el ID del grupo en el ID token.
4. **Configurar Perceptron:**

```json
{
  "entra": {
    "kind": "entra",
    "display_name": "Microsoft (Empresa)",
    "tenant_id": "<Directory (tenant) ID>",
    "client_id": "<Application (client) ID>",
    "client_secret": "<secreto>",
    "allowed_domains": ["empresa.com"],
    "default_role": "viewer",
    "role_mapping": [
      {"group": "<object id del grupo Data Science>", "workspace": "Ciencia de datos", "role": "editor"},
      {"group": "<object id del grupo IT>", "workspace": "Equipo", "role": "admin"}
    ]
  }
}
```

## Google Workspace

- **Credenciales:** *OAuth client ID* de tipo «Web application», con el mismo redirect URI, terminado en `/google/callback`.
- **Configuración:** `"kind": "google"` y `allowed_domains: ["empresa.com"]`. El dominio se envía como `hd`.
- **Roles:** el ID token de Google no trae grupos. El rol sale de `default_role`, o se asigna a mano en la consola.

## Proveedor genérico

`"kind": "generic"` con `"issuer": "https://idp.empresa.com/realms/x"`. Sirven Keycloak, Authentik, Okta y otros.

## Comportamiento

- **Vinculación:** el usuario se vincula por email verificado por el IdP; si no existe, se crea (`auto_create`).
- **Roles por grupo:** los grupos del token dan roles en cada login. Los roles que se asignan a mano no se quitan.
- **Validaciones:** se validan firma (JWKS del emisor), emisor, audiencia, vencimiento, `nonce` y `state`. Los intentos fallidos quedan en la auditoría (`auth.sso_failed`).
- **Login local:** con `PERCEPTRON_SERVER__PASSWORD_LOGIN=false` queda solo el SSO. Conviene dejar un Admin local de emergencia antes de apagarlo.

La aceptación de la Capa 5 («login SSO con Entra ID de prueba») necesita un tenant o una app registration de prueba. En CI el flujo completo se prueba contra un IdP simulado con claves RSA reales.
