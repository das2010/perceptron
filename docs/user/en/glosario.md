# Glossary

| Term | What it is |
|---|---|
| **Alert** | A monitoring notice (drift, degradation, retraining failures). It can be open, acknowledged or resolved |
| **ArchSpec** | Declarative (JSON) description of a network architecture as a graph of catalog blocks. It is validated before training and can be viewed as PyTorch code |
| **Architecture** | The structure of the neural network: which blocks it has and how they connect |
| **Autonomous agent** | Mode where the LLM acts as an ML engineer: it proposes, trains, diagnoses and iterates within limits enforced by the system. See [Training](entrenamiento.md#autonomous-agent) |
| **Catalog** | The set of allowed architecture blocks and templates, with their hyperparameter ranges and weight licenses |
| **Champion** | The project's model in production (stage **Production**) |
| **Challenger** | A candidate model (for example, a retrained one) that must beat the champion on the same data to be promoted |
| **Checkpoint** | Saved copy of the weights during training (the best and the last one) |
| **Copilot** | The LLM panel that talks with you and proposes changes to the project draft |
| **Data version** | Immutable snapshot of a dataset, identified by the hash of its content |
| **Dataset Profile Card** | Structured summary of the dataset (aggregated statistics, alerts). It is what the LLM sees at L1 |
| **Deployment (model in use)** | The champion, served and watched: it logs predictions, receives feedback and computes drift |
| **Diagnosis** | Analysis of a run's curves: overfitting, underfitting, divergence, learning rate, etc. |
| **Drift** | A change in the data compared with the training data (*data drift*) or in the relationship between data and outcome (*concept drift*) |
| **Engine** | The Python engine that runs data and ML work. It is the same on the desktop and on the Team Server |
| **Epoch** | One full pass over the training data |
| **Expert mode** | Defining the model as PyTorch code instead of an ArchSpec; it runs in a sandbox |
| **Export** | The model in a format usable outside Perceptron: ONNX, torch.export or TorchScript |
| **Feedback** | The real outcome of a prediction, sent later to measure performance in production |
| **HPO** | Hyperparameter search (optimization) |
| **Hyperparameter** | A parameter fixed before training (learning rate, layer size, dropout…) |
| **Leakage** | Information about the outcome that slips into the input variables; it makes the model look better than it is |
| **Lineage** | Which data version another one comes from and which transformation was applied |
| **LLM profile** | Which LLM model serves each purpose (copilot, architect, agent…) |
| **Model registry** | The project's **Models** list, with each model's stage (Candidate, Staging, Production, Archived) |
| **Model signature** | Description of inputs and outputs, version and hash, included in every export |
| **Pipeline (preparation)** | Data preparation steps (imputation, encoding, scaling, etc.), fitted on train only and packaged with the model |
| **Playground** | Screen to try the model on a case and see the prediction and its explanation |
| **Privacy level (L0–L3)** | What information about the dataset the LLM may receive in a project. See [LLM and privacy](llm-y-privacidad.md) |
| **Pruning** | Stopping badly performing trials early during a search (median, ASHA, Hyperband) |
| **Retraining policy** | Rules that trigger retraining (drift, cron, volume, degradation) and how the result is promoted |
| **Rollback** | Going back to the previous champion |
| **Run** | One training execution with a specific configuration; in a search, each trial is a run |
| **Sandbox** | Isolated environment (no network, no processes, no file access outside the run) where expert code runs |
| **Sealed test set** | The test split that is used for no decision; it is only opened for the final evaluation |
| **Splits** | Division of the data into train, validation and test |
| **Study** | A set of hyperparameter search trials under one strategy and budget |
| **Team Server** | On-premise team server: users, roles, worker queue and web UI |
| **Trial** | One hyperparameter configuration tried within a study |
| **Worker** | Team Server process that takes studies from the queue and trains them, on CPU or GPU |
| **Workspace** | Team Server space that groups projects, members and policies |
