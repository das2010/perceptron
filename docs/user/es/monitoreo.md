# Monitoreo y reentrenamiento

Un modelo que funciona hoy puede dejar de funcionar cuando cambian los datos. Perceptron vigila los
modelos en uso, avisa cuando algo cambia y puede reentrenarlos solo, promoviendo el nuevo modelo **solo
si mejora**.

## Champion, challenger y etapas

En **Modelos** cada modelo registrado tiene una **Etapa**: **Candidato**, **Staging**, **Producción** o
**Archivado**. El modelo en **Producción** es el **champion** (ícono de corona): el que se usa.

| Acción | Dónde | Qué hace |
|---|---|---|
| **Promover** | En un modelo que no es champion | Lo convierte en champion directamente |
| **Retar al champion** | En un modelo que no es champion | Evalúa los dos sobre el mismo conjunto de datos que ninguno usó para entrenar; el retador pasa a champion **solo si mejora** la métrica principal |
| **Volver al champion anterior** | Arriba de la tabla | Rollback en un clic: el champion anterior vuelve a producción |
| **Desplegar** | En el champion | Crea un modelo en uso vigilado y te lleva a **Monitoreo** |

El resultado de un reto se informa, por ejemplo: «El challenger no mejoró (roc_auc: 0,88 contra 0,91 en
120 casos): sigue el champion».

## Modelos en uso (deployments)

**Desplegar** sirve el champion (su export ONNX) desde el Engine o el Team Server y lo sigue: si
promovés otro modelo, el deployment pasa a usar el nuevo champion.

Antes de desplegar:

1. El modelo tiene que ser champion: si el proyecto todavía no tiene uno, tocá **Promover** en el
   modelo registrado.
2. Tiene que tener un export **ONNX** verificado (página del run → **Exportar el modelo**).

Por ahora los deployments monitoreados son para modelos **tabulares**; para imagen, texto, series y
audio llegan **próximamente**.

- Las aplicaciones piden predicciones al deployment (`POST /api/v1/deployments/{id}/predict`).
- Cada predicción se **registra** por muestreo. Solo se guardan las variables que usa el modelo (y una
  clave de negocio si la configuraste), no el resto del request.
- Cuando conocés el resultado real (el cliente se fue o no), lo enviás como **feedback**
  (`POST /api/v1/deployments/{id}/feedback`), asociado por id de predicción o por la clave.

En el desktop el Engine solo escucha en tu equipo: para que otras aplicaciones usen el deployment,
hacelo en el Team Server. Si solo necesitás servir el modelo sin monitoreo, usá el
[servidor de inferencia exportado](evaluacion-y-uso.md#servidor-de-inferencia-docker).

En **Monitoreo** ves cada deployment con su estado (**Activo** / **Detenido**), **Detener** /
**Reanudar** y **Chequear ahora**.

## Drift

Los chequeos corren cada N predicciones, en segundo plano, o cuando tocás **Chequear ahora**. Cada uno
compara una ventana de predicciones recientes («Ventana desde → hasta (N predicciones)») con los datos
de entrenamiento del champion.

- **Drift de datos**, por variable: la tabla muestra **Variable**, **Severidad**, **p-valor** y
  **Cambio**; en categóricas, también el porcentaje de categorías nuevas. Se usan PSI, KS,
  Jensen-Shannon y χ².
- **Drift de la salida del modelo:** si cambió la distribución de sus predicciones.
- **Performance con feedback:** la métrica sobre los casos que ya tienen resultado real, comparada con
  la del champion en test.

| Severidad | Qué significa | Qué hacer |
|---|---|---|
| **Sin drift** | Los datos se parecen a los de entrenamiento | Nada |
| **Leve** | Cambio chico, habitual | Seguir mirando |
| **Medio** | Cambio notable en una o más variables | Revisar la causa; considerar reentrenar |
| **Alto** | Los datos cambiaron mucho; el modelo puede estar fallando | Reentrenar o investigar ya |

Una sola variable con drift alto, si es una minoría de las variables, cuenta como medio para el total.
El drift sobre representaciones internas para imagen, texto y audio llega **próximamente**.

## Alertas

Las **Alertas** aparecen en la app y, si los configurás, por:

- **Alertas por email:** destinatarios separados por coma (requiere un servidor SMTP configurado).
- **Webhook (Teams / Slack):** la URL es de solo escritura y se guarda en el llavero.

Tocá **Guardar canales**. Para no inundarte, cada tipo de alerta tiene un período de espera antes de
repetirse. Filtrá por **Abiertas**, **Reconocidas**, **Resueltas** o **Todas**, y usá **Reconocer** y
**Resolver** para gestionarlas.

## Reentrenamiento automático

En **Monitoreo → Reentrenamiento automático** definís una política:

1. **Modelo en uso a vigilar:** el deployment.
2. Disparadores (podés combinar varios):
    - **Con drift de datos:** a partir de severidad **Medio** o **Alto**.
    - **Con N filas nuevas:** cuando se acumulan N filas etiquetadas nuevas (de fuentes streaming o
      feedback). Vacío = no usar.
    - **Por calendario (cron, UTC):** por ejemplo `0 3 * * 1` (lunes a las 3:00).
    - **Degradación de la métrica** con feedback: disponible desde la API; en la pantalla llega
      **próximamente**.
3. **Pedir aprobación antes de promover** (opcional).
4. Marcá **Política activa** y tocá **Guardar política**. **Reentrenar ahora** lo dispara a mano.

Qué pasa en cada reentrenamiento:

1. Se crea una versión de datos nueva: la del champion más las filas nuevas etiquetadas. Una parte de
   las filas nuevas se reserva como test; el test del champion no se mueve.
2. Se entrena un **challenger** con la misma arquitectura y pipeline, con una búsqueda reducida. En el
   Team Server va a la cola de workers.
3. Challenger y champion se comparan sobre el test nuevo, que ninguno de los dos vio.
4. Si el challenger mejora, se promueve (o queda **Espera aprobación**, con **Aprobar** / **Rechazar**).
   Si no, **No mejoró** y sigue el champion.

La tabla de ejecuciones muestra **Cuándo**, **Disparador**, **Estado** (En curso, Espera aprobación,
Promovido, No mejoró, Rechazado, Sin datos nuevos, Falló) y **Resultado**. Si un challenger promovido
resulta peor en la práctica, usá **Volver al champion anterior** en **Modelos**.

## Versiones de datos: diff y linaje

Cada reentrenamiento, etiquetado o carga deja una versión de datos nueva con su **linaje** (de qué
versión viene y qué se le hizo, por ejemplo «se agregaron 500 filas»). Entre dos versiones se puede ver
un **diff**: filas agregadas y eliminadas, cambios de esquema, cambios de distribución por columna y
archivos distintos. Por ahora el diff y el linaje se consultan desde la API del Engine; la vista en la
UI llega **próximamente**.
