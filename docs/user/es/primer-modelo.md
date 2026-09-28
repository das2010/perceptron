# Tu primer modelo: predecir churn (UC-01)

En este recorrido entrenás un modelo que anticipa qué clientes se van a dar de baja, a partir de una
tabla de clientes. No hace falta escribir código ni tener un LLM configurado: sin LLM, cada paso usa
recomendaciones **por reglas**. Con un equipo común tarda unos minutos.

## Qué necesitás

- Perceptron instalado ([Instalación](instalacion.md)) o acceso a un Team Server.
- Una tabla con una fila por cliente y una columna que diga si se fue. Podés usar el ejemplo del
  repositorio, `fixtures/uc01_churn/churn.csv` (400 clientes), con estas columnas:

| Columna | Qué es |
|---|---|
| `customer_id` | Identificador del cliente |
| `edad`, `region`, `plan`, `antiguedad_meses`, `cargo_mensual`, `tickets_90d` | Datos del cliente |
| `churn` | 1 si se dio de baja, 0 si no (lo que querés predecir) |

También sirve un Excel (`.xlsx`) con la misma idea.

## 1. Crear el proyecto

1. En **Inicio**, tocá **Nuevo proyecto**.
2. Completá **Nombre** (por ejemplo, «Churn clientes») y **Objetivo**: contalo en palabras, por ejemplo
   «Anticipar qué clientes se van a dar de baja». La IA usa este texto para proponer.
3. En **Privacidad frente al LLM** dejá **L1** (solo metadatos agregados). Ver
   [LLM y privacidad](llm-y-privacidad.md).
4. Tocá **Crear**. Se abre el proyecto en la pestaña **Datos**.

## 2. Agregar los datos

1. En **Agregar datos**, tocá **Subir archivo** y elegí `churn.csv`.
2. Revisá la **Vista previa de churn.csv**: las primeras filas y el tipo que Perceptron infirió para
   cada columna (Numérica, Categórica, Identificador, Fecha, etc.).
3. En **Columna objetivo** elegí `churn`. Si la dejás en «Detectar automáticamente», Perceptron la
   infiere.
4. Tocá **Crear versión de datos**.

Se crea una **versión de datos** inmutable: si más adelante cambiás el archivo, vas a tener otra
versión y los modelos anteriores siguen apuntando a la suya. Ver [Datos](datos.md).

## 3. Revisar el perfil

Al terminar el análisis aparece el **Perfil del dataset**:

- **Particiones:** cuántas filas van a entrenamiento, validación y test. El **test queda sellado**: no se
  usa para elegir el modelo, solo para la evaluación final.
- **Alertas de calidad:** desbalance de clases, columnas constantes, posibles fugas (por ejemplo, un
  identificador o una columna casi igual al objetivo), pocos datos. Cada alerta tiene severidad
  *Info*, *Atención* o *Importante*.
- **Distribución de clases** y **Columnas** (nulos, valores distintos, resumen).

Con UC-01 vas a ver, por ejemplo, que `customer_id` es un identificador: la preparación lo descarta.

## 4. Preparación de los datos

Tocá **Seguir: entrenar**. Se abre la pestaña **Entrenar**, que tiene cuatro pasos.

**Paso 1 — Datos y preparación.** Con la versión de datos elegida, tocá **Proponer preparación**.
Perceptron arma un pipeline (imputación de nulos, codificación de categorías, escalado…) y lista *por
qué* eligió cada paso. El pipeline se ajusta solo con los datos de entrenamiento, para no filtrar
información del test. Si querés verlo o cambiarlo, está en **Diseño → Pipelines de datos**.

## 5. Propuestas de arquitectura

**Paso 2 — Arquitectura.** Tocá **Proponer arquitecturas**. Aparecen 2 a 4 propuestas de red
neuronal, cada una con su justificación y una estimación («parámetros · MB · segundos por época»).

- Sin LLM (o con privacidad L0) las propuestas llevan la marca **Por reglas**.
- Con LLM, llevan **Sugerido por IA** y una justificación en lenguaje natural.
- La primera queda elegida; tocá **Elegir** en otra si preferís. **Abrir en el editor visual** te
  muestra el grafo de bloques.

## 6. Estrategia y presupuesto

**Paso 3 — Presupuesto y búsqueda de hiperparámetros.**

1. En **Intentos (trials)** poné cuántas configuraciones probar (para una primera prueba, 3 a 10).
2. En **Épocas máximas por intento**, por ejemplo 5 a 20.
3. Tocá **Recomendar estrategia**. Perceptron (o el LLM) elige la **Estrategia de búsqueda** (por
   ejemplo, TPE con poda) y los **Hiperparámetros a ajustar**. Podés descartarla y pedir otra.

Ver [Entrenamiento](entrenamiento.md) para las estrategias disponibles.

## 7. Entrenar en vivo

**Paso 4 — Entrenar.** Si tu desktop está conectado a un Team Server, en **Dónde entrenar** podés
elegir **En este equipo** o **En «servidor»** (y **Usar la GPU del servidor**). Tocá **Entrenar ahora**.

Pasás a **Experimentos → Entrenamiento en vivo**: ves cada época, las **Curvas de validación por
intento** y el estado del trabajo. Cuando todos los intentos terminan, el estado dice **Terminado**.

## 8. Evaluar en el test sellado

1. En la lista de **Entrenamientos**, abrí el run (los intentos se llaman `t000`, `t001`, …).
2. Revisá las **Curvas de entrenamiento** y el **Diagnóstico del entrenamiento** (sobreajuste,
   subajuste, tasa de aprendizaje, etc.).
3. Tocá **Evaluar en test**. Vas a ver las métricas en el test sellado (para churn: `roc_auc`,
   `accuracy`, precisión, recall, F1…) y la **Matriz de confusión**.

Como el test sellado no participó en ninguna decisión, esta es la estimación honesta de cómo va a
rendir el modelo con clientes nuevos.

## 9. Registrar el modelo

Tocá **Registrar modelo**. El botón pasa a **Modelo registrado** y el modelo aparece en la pestaña
**Modelos**, con sus métricas en test. Desde ahí podés desplegarlo y monitorearlo
([Monitoreo](monitoreo.md)).

## 10. Probarlo y exportarlo

En la misma página del run:

- **Evaluación avanzada:** errores, explicación, equidad, robustez e informe.
- **Exportar el modelo:** dejá tildado **ONNX** y tocá **Exportar**. Perceptron verifica que el ONNX
  dé los mismos resultados que PyTorch. Después podés descargar el **Servidor de inferencia (Docker)** o
  el **Proyecto de código**.
- **Probar el modelo** (playground, aparece cuando hay un ONNX verificado): tocá **Completar con un
  ejemplo**, cambiá valores y tocá **Predecir**. Ves la predicción, la confianza y, con **Explicar esta
  predicción**, la contribución de cada variable.

Todo esto está en [Evaluación y uso](evaluacion-y-uso.md).

## ¿Y ahora?

- Probá el **Wizard** del proyecto: el mismo recorrido en 9 pasos, con el copiloto explicando cada uno.
- Lanzá el **Agente** para que itere solo dentro de un presupuesto.
- Compará runs en **Experimentos** marcando dos o más y tocando **Comparar**.
