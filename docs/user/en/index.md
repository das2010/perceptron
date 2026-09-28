# Perceptron — User guide

Perceptron is a Preteco application for **training neural networks on your own hardware**, with an LLM
that guides you from raw data to a model that is evaluated, explained, exported and monitored.

- **Data:** tables (CSV, Excel, Parquet, JSON), images, text, time series and audio.
- **Tasks:** classification, regression, forecasting, anomaly detection, object detection,
  segmentation, OCR and audio pattern recognition.
- **No code** in guided mode; full control (ArchSpec, PyTorch code, search spaces) in expert mode.
- **Your data stays on your network:** training is local and you decide what the LLM may see (privacy
  levels L0–L3).

## Desktop and Team Server

| | Desktop | Team Server |
|---|---|---|
| What it is | Desktop app for Windows 10/11 and Linux | On-premise server for the team, with a web UI |
| Where training runs | On your machine (CPU or GPU) | On the server workers (CPU or GPU) |
| Users | You | Many, with local or SSO sign-in and Admin / Editor / Viewer roles |
| Data | Files and folders on your machine, databases, public datasets | Files you upload and "server sources" enabled by the Admin |
| Extras | — | Training queue, shared projects, administration console |

Both use **the same interface**. You can also combine them: connect your desktop to a Team Server and
choose, for each training, whether it runs on your machine or on the server GPU. See
[Team Server](team-server.md).

## Who are you?

| Persona | What suits you | Where to start |
|---|---|---|
| **Business analyst** — you know the problem and the data, you don't code | Guided mode: the wizard and the copilot decide almost everything and explain it | [Your first model](primer-modelo.md) |
| **Developer** — you code, but you are not an ML expert | Guided mode, looking at the architecture and parameters; you export the model into your application | [Your first model](primer-modelo.md) → [Evaluation and use](evaluacion-y-uso.md) |
| **Data scientist / ML engineer** | Visual ArchSpec editor, expert code, HPO strategies, run comparison, agent | [Training](entrenamiento.md) |
| **Platform admin** | Users, roles, LLM and privacy policies, license, queue and workers | [Team Server](team-server.md) and [LLM and privacy](llm-y-privacidad.md) |

## How the app is organized

**Sidebar:** Home · Projects · Queue · Administration · Settings.
"Queue" and "Administration" only have content on the Team Server.

- **Home:** Engine status ("Engine is running"), detected hardware and recent projects. You create a
  project here with **New project**.
- **Project:** each project has these tabs:

| Tab | What for |
|---|---|
| Overview | Goal, data versions, training runs and models; shortcuts to the next steps |
| Wizard | 9-step guided path, from the goal to the launch |
| Data | Load data, create versions and see the dataset profile |
| Labeling | Assisted labeling when labels are missing |
| Design | The project's architectures and data pipelines; visual editors and expert code |
| Train | Short path: preparation → architecture → search → **Train now** |
| Experiments | Live training, list of runs and comparison |
| Agent | Autonomous cycle: the LLM proposes, trains, diagnoses and iterates |
| Models | Registered models, champion/challenger, deployment and rollback |
| Monitoring | Models in use, drift, alerts and automatic retraining |
| LLM audit | What was sent to the LLM in each call and what it cost |

- **Copilot:** collapsible panel on the right ("Show or hide the copilot" button). Ask in natural
  language and it proposes changes.
- **Language and theme:** selectors at the top right (ES/EN; Light, Dark or System).

!!! note "What comes from the AI"
    Everything the LLM proposes is shown in violet with the **Suggested by AI** badge and **Accept** /
    **Dismiss** buttons. Nothing is applied until you accept it.

## Map of this guide

1. [Installation](instalacion.md) — desktop on Windows and Linux; pointer to the Team Server deployment.
2. [Your first model](primer-modelo.md) — full walkthrough with the churn case (UC-01).
3. [Data](datos.md) — sources, versions, profile and alerts, assisted labeling.
4. [Training](entrenamiento.md) — wizard, copilot, architectures, HPO, experiments and agent.
5. [Evaluation and use](evaluacion-y-uso.md) — metrics, explanations, report, export and playground.
6. [Monitoring](monitoreo.md) — deployments, drift, alerts, champion/challenger and retraining.
7. [LLM and privacy](llm-y-privacidad.md) — providers, L0–L3 levels, audit and costs.
8. [Team Server](team-server.md) — sign-in, roles, queue, connecting the desktop and administration.
9. [Glossary](glosario.md).
