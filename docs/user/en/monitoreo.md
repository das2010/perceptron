# Monitoring and retraining

A model that works today can stop working when the data changes. Perceptron watches the models in use,
warns you when something changes and can retrain them on its own, promoting the new model **only if it
improves**.

## Champion, challenger and stages

In **Models** each registered model has a **Stage**: **Candidate**, **Staging**, **Production** or
**Archived**. The model in **Production** is the **champion** (crown icon): the one in use.

| Action | Where | What it does |
|---|---|---|
| **Promote** | On a model that is not the champion | Makes it the champion directly |
| **Challenge the champion** | On a model that is not the champion | Evaluates both on the same data neither was trained on; the challenger becomes champion **only if it improves** the primary metric |
| **Roll back to previous champion** | Above the table | One-click rollback: the previous champion goes back to production |
| **Deploy** | On the champion | Creates a monitored model in use and takes you to **Monitoring** |

The outcome of a challenge is reported, for example: "The challenger did not improve (roc_auc: 0.88 vs
0.91 on 120 cases): the champion stays".

## Models in use (deployments)

**Deploy** serves the champion (its ONNX export) from the Engine or the Team Server and follows it: if
you promote another model, the deployment switches to the new champion.

Before deploying:

1. The model must be the champion: if the project doesn't have one yet, click **Promote** on the
   registered model.
2. It must have a verified **ONNX** export (run page → **Export the model**).

For now monitored deployments are for **tabular** models; image, text, series and audio are **coming
soon**.

- Applications request predictions from the deployment (`POST /api/v1/deployments/{id}/predict`).
- Predictions are **logged** with sampling. Only the variables the model uses are stored (plus a
  business key if you configured one), not the rest of the request.
- When you learn the real outcome (the customer left or not), send it as **feedback**
  (`POST /api/v1/deployments/{id}/feedback`), matched by prediction id or by the key.

On the desktop the Engine only listens on your machine: for other applications to use the deployment,
deploy on the Team Server. If you only need to serve the model without monitoring, use the
[exported inference server](evaluacion-y-uso.md#inference-server-docker).

In **Monitoring** you see each deployment with its status (**Active** / **Stopped**), **Stop** /
**Resume** and **Check now**.

## Drift

Checks run every N predictions in the background, or when you click **Check now**. Each one compares a
window of recent predictions ("Window from → to (N predictions)") with the champion's training data.

- **Data drift**, per feature: the table shows **Feature**, **Severity**, **p-value** and **Change**; for
  categorical features, also the percentage of new categories. PSI, KS, Jensen-Shannon and χ² are used.
- **Model output drift:** whether the distribution of its predictions changed.
- **Performance with feedback:** the metric on cases that already have a real outcome, compared with the
  champion's test metric.

| Severity | What it means | What to do |
|---|---|---|
| **No drift** | The data looks like the training data | Nothing |
| **Low** | Small, usual change | Keep watching |
| **Medium** | Noticeable change in one or more features | Look for the cause; consider retraining |
| **High** | The data changed a lot; the model may be failing | Retrain or investigate now |

A single feature with high drift, when it is a minority of the features, counts as medium overall.
Drift on internal representations for image, text and audio is **coming soon**.

## Alerts

**Alerts** show up in the app and, if you configure them, through:

- **Email alerts:** comma-separated recipients (requires a configured SMTP server).
- **Webhook (Teams / Slack):** the URL is write-only and stored in the keychain.

Click **Save channels**. To avoid flooding you, each alert type has a cooldown before it repeats. Filter
by **Open**, **Acknowledged**, **Resolved** or **All**, and use **Acknowledge** and **Resolve** to
manage them.

## Automatic retraining

In **Monitoring → Automatic retraining** you define a policy:

1. **Model in use to watch:** the deployment.
2. Triggers (you can combine several):
    - **On data drift:** from severity **Medium** or **High**.
    - **On N new rows:** when N new labeled rows accumulate (from streaming sources or feedback).
      Empty = don't use it.
    - **On a schedule (cron, UTC):** for example `0 3 * * 1` (Mondays at 3:00).
    - **Metric degradation** with feedback: available through the API; on screen it is **coming soon**.
3. **Ask for approval before promoting** (optional).
4. Check **Policy enabled** and click **Save policy**. **Retrain now** triggers it by hand.

What happens on each retraining:

1. A new data version is created: the champion's plus the new labeled rows. Part of the new rows is
   held out as test; the champion's test set does not move.
2. A **challenger** is trained with the same architecture and pipeline, with a reduced search. On the
   Team Server it goes to the worker queue.
3. Challenger and champion are compared on the new test set, which neither has seen.
4. If the challenger improves, it is promoted (or it stays **Awaiting approval**, with **Approve** /
   **Reject**). Otherwise it **Did not improve** and the champion stays.

The runs table shows **When**, **Trigger**, **Status** (Running, Awaiting approval, Promoted, Did not
improve, Rejected, No new data, Failed) and **Result**. If a promoted challenger turns out worse in
practice, use **Roll back to previous champion** in **Models**.

## Data versions: diff and lineage

Every retraining, labeling or load leaves a new data version with its **lineage** (which version it
comes from and what was done to it, for example "500 rows appended"). Between two versions you can get
a **diff**: added and removed rows, schema changes, per-column distribution changes and differing files.
For now diff and lineage are queried through the Engine API; the UI view is **coming soon**.
