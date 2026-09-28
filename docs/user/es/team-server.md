# Team Server

El Team Server es la versión de equipo de Perceptron: corre en un servidor de tu organización
(on-premise), se usa desde el navegador con la misma interfaz que el desktop y suma usuarios, roles,
proyectos compartidos y una cola de entrenamiento con workers CPU y GPU. También sirve como destino de
entrenamiento para los desktops.

Esta página tiene dos partes: [para usuarios](#para-usuarios) y [para administradores](#para-administradores).

## Para usuarios

### Ingresar

Abrí en el navegador la URL que te pasó el Admin (por ejemplo, `https://perceptron.empresa.com`).

- **Con SSO:** tocá **Ingresar con «proveedor»** (por ejemplo, Microsoft o Google). La verificación en
  dos pasos la maneja tu proveedor de identidad.
- **Con usuario del servidor:** completá **Email** y **Contraseña** y tocá **Ingresar**.

Después de varios intentos fallidos la cuenta queda bloqueada unos minutos. Para salir, **Cerrar
sesión** (arriba a la derecha).

### Roles

| Rol | Qué puede hacer |
|---|---|
| **Admin** | Todo lo del Editor, más usuarios, roles, políticas de LLM y privacidad, auditoría y licencia |
| **Editor** | Crear y editar proyectos, cargar datos, entrenar, evaluar, registrar, exportar y desplegar |
| **Viewer** | Ver proyectos, runs, informes y modelos; usar el playground. No entrena, no exporta, no modifica |

Los roles se asignan por **workspace** y por **proyecto**, y el rol del proyecto manda: podés ser Editor
en un proyecto y Viewer en el resto. Si solo tenés lectura, al lado del nombre del proyecto aparece
**Solo lectura**. Los permisos los hace cumplir el servidor, no solo la pantalla.

### Proyectos compartidos

Los proyectos del workspace son del equipo: todos los que tienen rol en él los ven en la barra lateral,
con sus datos, runs y modelos. Cada cambio queda en la auditoría del servidor.

### Datos en el servidor

En la UI web no hay acceso a las carpetas de tu equipo. Tenés dos opciones en **Datos**:

- **Subir archivo** / **Subir carpeta** desde el navegador.
- **Fuentes del servidor:** carpetas que el Admin monta en el servidor (de solo lectura). Navegá la
  **Ruta** y tocá **Usar**. Conviene para datasets grandes que ya están en la red.

Las bases de datos y los datasets públicos funcionan igual que en el desktop. Ver [Datos](datos.md).

### Cola de entrenamiento

En la UI web, cada entrenamiento va a la **cola** y lo toma un worker. La página **Cola** (barra lateral)
muestra:

- El modo y las **cuotas**: cuántos estudios en curso puede tener cada persona y cada workspace.
- **Workers:** nombre, **Colas** (`gpu` / `cpu`), **Hardware** (GPU o «Solo CPU») y **Estado**
  (**Ocupado** / **Libre**). Si no hay workers conectados, los estudios esperan.
- **Estudios en cola o en curso:** estudio, cola, desde cuándo y en qué worker corre.

El progreso en vivo se ve en **Experimentos**, igual que en local. Las prioridades entre usuarios llegan
**próximamente**.

### Conectar el desktop a un servidor

Así entrenás en la GPU del servidor sin salir de tu desktop:

1. En el desktop, andá a **Configuración → Servidores de equipo**.
2. Completá **Nombre** (cómo lo vas a ver), **URL**, **Email** y **Contraseña**, y tocá **Conectar**. La
   contraseña se usa solo para el login: se guarda un token en el llavero, no la contraseña.
3. En un proyecto, en **Entrenar → paso 4**, elegí en **Dónde entrenar**: **En este equipo** o **En
   «nombre del servidor»**. Tildá **Usar la GPU del servidor** para ir a un worker con GPU.
4. Tocá **Entrenar ahora**. Perceptron sube al servidor el proyecto, la versión de datos, el pipeline y
   la arquitectura («Subiendo al servidor…»; la subida es por partes y se reanuda si se corta) y trae de
   vuelta los runs.

Para desconectar, usá **Quitar** en la lista de servidores. Por ahora la conexión desde el desktop usa
un usuario del servidor con contraseña (no SSO), y la promoción de un proyecto local completo a proyecto
de equipo llega **próximamente**.

## Para administradores

La instalación, el SSO y los backups están en la documentación de operación:

- [Despliegue (Docker Compose y Helm)](https://github.com/das2010/perceptron/blob/main/server/deploy/README.md)
- [SSO con Entra ID, Google Workspace u otro proveedor OIDC](https://github.com/das2010/perceptron/blob/main/docs/ops/sso.md)
- [Backups y restauración](https://github.com/das2010/perceptron/blob/main/docs/ops/backups.md)

### Consola de administración

**Administración** (barra lateral) tiene cuatro pestañas. **Usuarios** y **Auditoría** son para el Admin
del servidor; **Roles** y **Políticas**, también para los Admin de un workspace.

#### Usuarios

- **Nuevo usuario:** **Email**, **Nombre**, **Contraseña inicial** (mínimo 12 caracteres), **Rol en el
  workspace** y, si corresponde, **Admin del servidor**. Tocá **Crear usuario**.
- En la lista ves **Estado** (Activo, Inactivo, Bloqueado) y **Último ingreso**, y podés **Activar** /
  **Desactivar**, **Hacer admin** / **Quitar admin** o asignar una **Nueva contraseña**.
- Con SSO, los usuarios se crean solos en el primer ingreso y sus roles salen de los grupos del
  proveedor de identidad.

#### Roles

Elegí **Workspace**, **Usuario**, **Alcance** (**Todo el workspace** o un proyecto) y **Rol**, y tocá
**Asignar**. **Quitar rol** lo revoca. El rol de un proyecto manda sobre el del workspace.

#### Políticas

- **Proveedores permitidos:** separados por coma (por ejemplo, `ollama, anthropic`). Vacío = todos.
- **Privacidad máxima frente al LLM:** ningún proyecto del workspace envía más que este nivel (L0 =
  nada).
- **Máximo con LLM local:** puede ser mayor cuando el modelo corre en tu propia red (por ejemplo, L3
  con Ollama aunque el general sea L1).
- **SSO:** lista los proveedores configurados. Se configuran en el servidor, no desde la consola.

#### Auditoría

Registro de solo agregado con ingresos, escrituras, accesos denegados, lecturas de datos, descargas y
cambios de roles. Filtrá por **Acción** (prefijo, por ejemplo `auth.` o `api.createStudy`) y por
**Usuario**; ves **Cuándo**, **Usuario**, **Acción**, **Recurso** y **Resultado**. Las llamadas al LLM
se auditan por proyecto, en **Auditoría LLM**.

La gestión de cuotas, almacenamiento y estado de workers desde la consola llega **próximamente**; hoy
se configuran en el despliegue.

### Licencia

En **Configuración → Licencia** ves el **Estado** (Válida, Sin licencia, Vencida, Inválida, No
reconocida, Todavía no vigente), el **Titular**, la **Edición**, los **Topes** (usuarios, servidores y
GPU) y el vencimiento (o **Perpetua**).

Para instalarla, pegá en **Instalar una licencia** el contenido del `license.json` que te envió Preteco y
tocá **Instalar**. La licencia es un archivo firmado que se valida sin conexión.

!!! note
    En esta versión la licencia se informa pero **no limita el uso**.

### Telemetría

En **Configuración → Telemetría de uso** decidís si Perceptron envía estadísticas de uso. Está
**desactivada** por defecto y es anónima: versión, sistema operativo, hardware agregado y cuántas veces
se usa cada función. **Nunca** tus datos, nombres de columnas, rutas ni métricas de modelos.

- **Ver exactamente qué se enviaría** muestra el contenido antes de activarla.
- **Activar telemetría** / **Desactivar telemetría**.
- Si no hay un destino configurado, no se envía nada aunque esté activada.
