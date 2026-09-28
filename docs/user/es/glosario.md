# Glosario

| Término | Qué es |
|---|---|
| **Agente autónomo** | Modo en el que el LLM actúa como ingeniero de ML: propone, entrena, diagnostica e itera dentro de límites que aplica el sistema. Ver [Entrenamiento](entrenamiento.md#agente-autonomo) |
| **Alerta** | Aviso del monitoreo (drift, degradación, fallas de reentrenamiento). Puede estar abierta, reconocida o resuelta |
| **ArchSpec** | Descripción declarativa (JSON) de una arquitectura de red como grafo de bloques del catálogo. Se valida antes de entrenar y se puede ver como código PyTorch |
| **Arquitectura** | La estructura de la red neuronal: qué bloques tiene y cómo se conectan |
| **Catálogo** | Conjunto de bloques y plantillas de arquitectura permitidos, con sus rangos de hiperparámetros y licencias de pesos |
| **Champion** | El modelo en producción (etapa **Producción**) de un proyecto |
| **Challenger** | Modelo candidato (por ejemplo, reentrenado) que tiene que superar al champion sobre los mismos datos para ser promovido |
| **Checkpoint** | Copia guardada de los pesos durante el entrenamiento (el mejor y el último) |
| **Copiloto** | Panel del LLM que conversa con vos y propone cambios al borrador del proyecto |
| **Dataset Profile Card** | Resumen estructurado del dataset (estadísticas agregadas, alertas). Es lo que ve el LLM en L1 |
| **Deployment (modelo en uso)** | El champion servido y vigilado: registra predicciones, recibe feedback y calcula drift |
| **Diagnóstico** | Análisis de las curvas de un run: sobreajuste, subajuste, divergencia, tasa de aprendizaje, etc. |
| **Drift** | Cambio en los datos respecto de los de entrenamiento (*data drift*) o en la relación entre datos y resultado (*concept drift*) |
| **Engine** | El motor Python que ejecuta datos y ML. Es el mismo en el desktop y en el Team Server |
| **Época** | Una pasada completa por los datos de entrenamiento |
| **Estudio (study)** | Conjunto de intentos de búsqueda de hiperparámetros bajo una estrategia y un presupuesto |
| **Export** | El modelo en un formato para usar fuera de Perceptron: ONNX, torch.export o TorchScript |
| **Feedback** | El resultado real de una predicción, enviado después para medir la performance en producción |
| **Firma del modelo** | Descripción de entradas y salidas, versión y hash, incluida en todos los exports |
| **HPO** | Búsqueda (optimización) de hiperparámetros |
| **Hiperparámetro** | Parámetro que se fija antes de entrenar (tasa de aprendizaje, tamaño de capa, dropout…) |
| **Leakage (fuga)** | Información del resultado que se cuela en las variables de entrada; hace que el modelo parezca mejor de lo que es |
| **Linaje** | De qué versión de datos viene otra y qué transformación se le aplicó |
| **Modo experto** | Definir el modelo como código PyTorch en lugar de ArchSpec; corre en un sandbox |
| **Nivel de privacidad (L0–L3)** | Qué información del dataset puede recibir el LLM en un proyecto. Ver [LLM y privacidad](llm-y-privacidad.md) |
| **Particiones (splits)** | División de los datos en entrenamiento, validación y test |
| **Perfil LLM** | Qué modelo de LLM atiende cada propósito (copiloto, arquitecto, agente…) |
| **Pipeline (preparación)** | Pasos de preparación de datos (imputación, codificación, escalado, etc.), ajustados solo con entrenamiento y empaquetados con el modelo |
| **Playground** | Pantalla para probar el modelo con un caso y ver la predicción y su explicación |
| **Poda (pruning)** | Cortar temprano los intentos que van mal durante una búsqueda (mediana, ASHA, Hyperband) |
| **Política de reentrenamiento** | Reglas que disparan un reentrenamiento (drift, cron, volumen, degradación) y cómo se promueve el resultado |
| **Registro de modelos** | La lista de **Modelos** de un proyecto, con su etapa (Candidato, Staging, Producción, Archivado) |
| **Rollback** | Volver al champion anterior |
| **Run** | Una ejecución de entrenamiento con una configuración concreta; en una búsqueda, cada intento es un run |
| **Sandbox** | Entorno aislado (sin red, sin procesos, sin acceso a archivos fuera del run) donde corre el código experto |
| **Team Server** | Servidor on-premise de equipo: usuarios, roles, cola de workers y UI web |
| **Test sellado** | Partición de test que no se usa para ninguna decisión; solo se abre para la evaluación final |
| **Trial (intento)** | Una configuración de hiperparámetros probada dentro de un estudio |
| **Versión de datos** | Instantánea inmutable de un dataset, identificada por el hash de su contenido |
| **Worker** | Proceso del Team Server que toma estudios de la cola y los entrena, en CPU o GPU |
| **Workspace** | Espacio de trabajo del Team Server que agrupa proyectos, miembros y políticas |
