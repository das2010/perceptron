# Perceptron — Guía de usuario

Perceptron es una aplicación de Preteco para **entrenar redes neuronales en tu propio hardware**, con un
LLM que te guía desde los datos crudos hasta un modelo evaluado, explicado, exportado y monitoreado.

- **Datos:** tablas (CSV, Excel, Parquet, JSON), imágenes, texto, series temporales y audio.
- **Tareas:** clasificación, regresión, pronóstico, detección de anomalías, detección de objetos,
  segmentación, OCR y reconocimiento de patrones de audio.
- **Sin escribir código** en el modo guiado; con control total (ArchSpec, código PyTorch, espacios de
  búsqueda) en el modo experto.
- **Tus datos no salen de tu red:** el entrenamiento es local y vos decidís qué puede ver el LLM
  (niveles de privacidad L0–L3).

## Desktop y Team Server

| | Desktop | Team Server |
|---|---|---|
| Qué es | App de escritorio para Windows 10/11 y Linux | Servidor on-premise para el equipo, con UI web |
| Dónde corre el entrenamiento | En tu equipo (CPU o GPU) | En los workers del servidor (CPU o GPU) |
| Usuarios | Vos | Varios, con login local o SSO y roles Admin / Editor / Viewer |
| Datos | Archivos y carpetas de tu equipo, bases de datos, datasets públicos | Archivos que subís y «fuentes del servidor» que habilita el Admin |
| Extras | — | Cola de entrenamiento, proyectos compartidos, consola de administración |

Los dos usan **la misma interfaz**. Además podés combinarlos: conectás tu desktop a un Team Server y
elegís, en cada entrenamiento, si corre en tu equipo o en la GPU del servidor. Ver
[Team Server](team-server.md).

## ¿Quién sos?

| Persona | Qué te conviene | Por dónde empezar |
|---|---|---|
| **Analista de negocio** — conocés el problema y los datos, no programás | Modo guiado: el wizard y el copiloto deciden casi todo y te lo explican | [Tu primer modelo](primer-modelo.md) |
| **Desarrollador** — programás, pero no sos experto en ML | Modo guiado, mirando la arquitectura y los parámetros; exportás el modelo a tu aplicación | [Tu primer modelo](primer-modelo.md) → [Evaluación y uso](evaluacion-y-uso.md) |
| **Data scientist / ML engineer** | Editor visual de ArchSpec, código experto, estrategias de HPO, comparación de runs, agente | [Entrenamiento](entrenamiento.md) |
| **Admin de plataforma** | Usuarios, roles, políticas de LLM y privacidad, licencia, cola y workers | [Team Server](team-server.md) y [LLM y privacidad](llm-y-privacidad.md) |

## Cómo está organizada la app

**Barra lateral:** Inicio · Proyectos · Cola · Administración · Configuración.
«Cola» y «Administración» solo tienen contenido en el Team Server.

- **Inicio:** estado del Engine («Engine operativo»), hardware detectado y proyectos recientes. Desde
  acá creás un proyecto con **Nuevo proyecto**.
- **Proyecto:** cada proyecto tiene estas pestañas:

| Pestaña | Para qué |
|---|---|
| Resumen | Objetivo, versiones de datos, entrenamientos y modelos; accesos a los próximos pasos |
| Wizard | Recorrido guiado de 9 pasos, del objetivo al lanzamiento |
| Datos | Cargar datos, crear versiones y ver el perfil del dataset |
| Etiquetado | Etiquetado asistido cuando faltan etiquetas |
| Diseño | Arquitecturas y pipelines de datos del proyecto; editores visuales y código experto |
| Entrenar | Camino corto: preparación → arquitectura → búsqueda → **Entrenar ahora** |
| Experimentos | Entrenamiento en vivo, lista de runs y comparación |
| Agente | Ciclo autónomo: el LLM propone, entrena, diagnostica e itera |
| Modelos | Modelos registrados, champion/challenger, despliegue y rollback |
| Monitoreo | Modelos en uso, drift, alertas y reentrenamiento automático |
| Auditoría LLM | Qué se le envió al LLM en cada llamada y cuánto costó |

- **Copiloto:** panel plegable a la derecha (botón «Mostrar u ocultar el copiloto»). Le preguntás en
  lenguaje natural y te propone cambios.
- **Idioma y tema:** selectores arriba a la derecha (ES/EN; Claro, Oscuro o Sistema).

!!! note "Lo que viene de la IA"
    Todo lo que propone el LLM aparece en violeta con la marca **Sugerido por IA** y botones
    **Aceptar** / **Descartar**. Nada se aplica sin que lo aceptes.

## Mapa de esta guía

1. [Instalación](instalacion.md) — desktop en Windows y Linux; puntero al despliegue del Team Server.
2. [Tu primer modelo](primer-modelo.md) — recorrido completo con el caso de churn (UC-01).
3. [Datos](datos.md) — fuentes, versiones, perfil y alertas, etiquetado asistido.
4. [Entrenamiento](entrenamiento.md) — wizard, copiloto, arquitecturas, HPO, experimentos y agente.
5. [Evaluación y uso](evaluacion-y-uso.md) — métricas, explicaciones, informe, export y playground.
6. [Monitoreo](monitoreo.md) — despliegues, drift, alertas, champion/challenger y reentrenamiento.
7. [LLM y privacidad](llm-y-privacidad.md) — proveedores, niveles L0–L3, auditoría y costos.
8. [Team Server](team-server.md) — login, roles, cola, conexión del desktop y administración.
9. [Glosario](glosario.md).
