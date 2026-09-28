# LLM y privacidad

Perceptron usa un LLM como copiloto, arquitecto, estratega de hiperparámetros, diagnosticador, agente,
redactor del informe y ayudante de etiquetado. **Es opcional:** sin LLM (o con privacidad L0) todo
funciona con recomendaciones por reglas.

Perceptron nunca llama al proveedor directamente desde cada función: todas las llamadas pasan por un
único punto (el *gateway*) que aplica tu nivel de privacidad, valida las respuestas, controla el costo y
deja registro en la auditoría.

## Proveedores

| Proveedor | Dónde corre | Clave |
|---|---|---|
| **Anthropic Claude** (perfil por defecto) | Nube | Sí |
| OpenAI | Nube | Sí |
| Google Gemini | Nube | Sí |
| Kimi (Moonshot) | Nube | Sí |
| **Ollama** | Local (`http://127.0.0.1:11434`) | No |
| **OpenAI-compatible** (LM Studio, vLLM, llama.cpp server) | Local (por defecto `http://127.0.0.1:1234/v1`) | No |

### Configurarlo

En **Configuración**:

1. En **Proveedores** ves cada proveedor, **Dónde corre** (Local / Nube), los **Modelos del catálogo** y
   el estado de la **Clave** (No necesita clave / Clave guardada / Sin clave).
2. Pegá la clave del proveedor («Pegá la clave (no se vuelve a mostrar)») y tocá **Guardar**.
3. En **Perfil LLM → Perfil activo** elegí qué perfil usar. Un perfil define qué modelo atiende cada
   propósito: Copiloto, Arquitecto, Estratega de HPO, Diagnóstico, Agente, Informe y Etiquetado.
4. Tocá **Probar conexión**. Vas a ver, por ejemplo, «Conexión correcta: proveedor/modelo en 2 s».

Los modelos y sus precios viven en un **catálogo de configuración** que se actualiza sin reinstalar
(el Admin puede reemplazarlo con `PERCEPTRON_LLM__CATALOG_FILE`). Cada modelo declara qué sabe hacer; si
un modelo no acepta imágenes, por ejemplo, Perceptron no le manda imágenes.

### LLM local (Ollama o LM Studio)

Con un modelo local los datos no salen de tu equipo o de tu red:

1. Instalá Ollama o LM Studio y descargá un modelo. Conviene uno que respete salidas JSON estructuradas.
2. Dejalo corriendo en la dirección de la tabla.
3. En Perceptron elegí el perfil del proveedor local y tocá **Probar conexión**.

Un modelo chico en CPU funciona, pero puede tardar minutos por llamada con prompts largos.

## Niveles de privacidad (L0–L3)

Cada proyecto tiene su nivel, que elegís al crearlo en **Privacidad frente al LLM**. El nivel se ve como
una etiqueta al lado del nombre del proyecto.

| Nivel | Qué ve el LLM | Ejemplo con la tabla de churn | Cuándo usarlo |
|---|---|---|---|
| **L0 — Sin LLM** | Nada. Se usan solo reglas | — | Datos muy sensibles y sin LLM local |
| **L1 — Metadatos** (por defecto) | Esquema, estadísticas agregadas, distribución de clases, tamaños, métricas y curvas. **Nunca valores individuales** | «columna `edad`: numérica, media 47, 0 % nulos» | La mayoría de los proyectos |
| **L2 — Muestras anonimizadas** | L1 + unas pocas filas (5 por defecto) con los datos personales enmascarados | L1 + filas como «edad 52, plan premium, email ‹EMAIL›» | Cuando el LLM necesita entender el contenido (categorías, ejemplos) |
| **L3 — Muestras crudas** | L2 sin anonimizar, incluidas imágenes o audio si el modelo es multimodal | Filas tal cual | Datos no sensibles o LLM local |

**Qué enmascara L2:** emails, URLs, direcciones IP, IBAN, tarjetas, teléfonos, DNI y CUIT/CUIL, más las
reglas propias que se configuren. Para no dejar pasar nombres de personas, **los campos de texto libre
no se envían en L2** salvo que el Admin configure un detector de nombres (NER) con licencia compatible;
queda registrado en la auditoría. Si necesitás que el LLM lea texto libre, usá L3 con un LLM local.

En todos los niveles, las clases con muy pocos casos se envían con un seudónimo (se traduce de vuelta
en tu equipo) y las rutas de archivos se reemplazan.

Algunas funciones dependen del nivel: por ejemplo, el pre-etiquetado de texto con el LLM necesita L2 o
L3. Además:

- El contenido de tus datos se envía al LLM siempre marcado como **datos**, nunca como instrucciones, y
  las respuestas solo actúan a través de esquemas validados.
- Toda propuesta del LLM que cambia algo (arquitectura, estrategia, cambios al borrador) se valida y
  aparece como **Sugerido por IA**, para que la aceptes o la descartes.
- En el Team Server el Admin fija un **nivel máximo** por workspace (y, opcionalmente, uno más alto para
  LLM locales) y los **proveedores permitidos**. Ningún proyecto supera ese nivel. Ver
  [Team Server](team-server.md#politicas).

## Auditoría LLM

La pestaña **Auditoría LLM** del proyecto muestra **todas** las llamadas al LLM, con un resumen («12
llamadas · costo total $0,18»):

| Columna | Qué muestra |
|---|---|
| Cuándo | Fecha y hora |
| Propósito | Copiloto, Arquitecto, Estratega de HPO, Diagnóstico, Agente, Informe o Etiquetado |
| Proveedor/modelo | A quién se envió |
| Nivel | Nivel de privacidad aplicado |
| Estado / Costo | Resultado y costo estimado |

**Ver lo enviado** muestra el contenido **exacto** que salió, después del filtro, y la sección **Filtrado
por privacidad** con lo que se enmascaró o quitó. Así podés comprobar, por ejemplo, que en L1 no salió
ningún valor individual. Desde la CLI: `perceptron llm audit`.

## Costos y presupuesto

- Cada llamada estima su costo con los precios del catálogo; la auditoría muestra el acumulado.
- **Presupuesto por proyecto:** USD 5 por defecto (configurable con `PERCEPTRON_LLM__PROJECT_BUDGET_USD`).
- **Presupuesto del agente:** **Costo máximo de LLM (USD)** al lanzarlo.
- Antes de cada llamada se verifica el presupuesto: si no alcanza, no se envía y Perceptron sigue **por
  reglas** (vas a ver el motivo en «Se usaron reglas: …»).
- Las respuestas se guardan en **caché**: la misma consulta, con el mismo modelo, no se vuelve a pagar y
  da el mismo resultado.

Las cuotas de LLM por usuario y por workspace en el Team Server llegan **próximamente**.

## Dónde se guardan las claves

- **Desktop:** en el llavero del sistema operativo (Credential Manager en Windows, Secret Service en
  Linux). No quedan en el proyecto, en los logs ni en los exports.
- **Team Server:** cifradas en el servidor con una clave maestra que define el Admin.
- Lo mismo vale para contraseñas de bases de datos, tokens de Hugging Face o Kaggle y webhooks de
  alertas.
