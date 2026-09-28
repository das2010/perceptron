# Your first model: predicting churn (UC-01)

In this walkthrough you train a model that anticipates which customers will cancel, from a customer
table. You don't need to write code or have an LLM configured: without an LLM, every step uses
**rule-based** recommendations. On an ordinary machine it takes a few minutes.

## What you need

- Perceptron installed ([Installation](instalacion.md)) or access to a Team Server.
- A table with one row per customer and a column saying whether they left. You can use the sample in
  the repository, `fixtures/uc01_churn/churn.csv` (400 customers), with these columns:

| Column | What it is |
|---|---|
| `customer_id` | Customer identifier |
| `edad`, `region`, `plan`, `antiguedad_meses`, `cargo_mensual`, `tickets_90d` | Customer data (age, region, plan, tenure in months, monthly charge, tickets in the last 90 days) |
| `churn` | 1 if they cancelled, 0 otherwise (what you want to predict) |

An Excel file (`.xlsx`) with the same idea works too.

## 1. Create the project

1. On **Home**, click **New project**.
2. Fill in **Name** (for example, "Customer churn") and **Goal**: say it in words, for example
   "Anticipate which customers will cancel". The AI uses this text to make proposals.
3. Leave **Privacy towards the LLM** at **L1** (aggregated metadata only). See
   [LLM and privacy](llm-y-privacidad.md).
4. Click **Create**. The project opens on the **Data** tab.

## 2. Add the data

1. Under **Add data**, click **Upload file** and pick `churn.csv`.
2. Check the **Preview of churn.csv**: the first rows and the type Perceptron inferred for each column
   (Numeric, Categorical, Identifier, Date, etc.).
3. In **Target column** pick `churn`. If you leave it at "Detect automatically", Perceptron infers it.
4. Click **Create data version**.

An immutable **data version** is created: if you later change the file, you get another version and
earlier models keep pointing to theirs. See [Data](datos.md).

## 3. Review the profile

When the analysis finishes, the **Dataset profile** appears:

- **Splits:** how many rows go to train, validation and test. The **test set is sealed**: it is not used
  to pick the model, only for the final evaluation.
- **Quality alerts:** class imbalance, constant columns, possible leakage (for example, an identifier or
  a column almost equal to the target), too little data. Each alert has a severity: *Info*, *Warning* or
  *Important*.
- **Class distribution** and **Columns** (nulls, distinct values, summary).

With UC-01 you will see, for example, that `customer_id` is an identifier: the preparation drops it.

## 4. Data preparation

Click **Next: train**. The **Train** tab opens, with four steps.

**Step 1 — Data and preparation.** With the data version selected, click **Propose preparation**.
Perceptron builds a pipeline (null imputation, category encoding, scaling…) and lists *why* it chose
each step. The pipeline is fitted on the training data only, so no information leaks from the test set.
To see or change it, go to **Design → Data pipelines**.

## 5. Architecture proposals

**Step 2 — Architecture.** Click **Propose architectures**. You get 2 to 4 neural network proposals,
each with its rationale and an estimate ("parameters · MB · seconds per epoch").

- Without an LLM (or with privacy L0) proposals carry the **By rules** badge.
- With an LLM, they carry **Suggested by AI** and a natural-language rationale.
- The first one is selected; click **Choose** on another if you prefer. **Open in the visual editor**
  shows the block graph.

## 6. Strategy and budget

**Step 3 — Budget and hyperparameter search.**

1. In **Trials**, set how many configurations to try (for a first test, 3 to 10).
2. In **Max epochs per trial**, for example 5 to 20.
3. Click **Recommend strategy**. Perceptron (or the LLM) picks the **Search strategy** (for example, TPE
   with pruning) and the **Hyperparameters to tune**. You can dismiss it and ask again.

See [Training](entrenamiento.md) for the available strategies.

## 7. Train live

**Step 4 — Train.** If your desktop is connected to a Team Server, **Where to train** lets you choose
**On this machine** or **On "server"** (and **Use the server GPU**). Click **Train now**.

You land on **Experiments → Live training**: you see each epoch, the **Validation curves per trial** and
the job status. When all trials are done, the status says **Finished**.

## 8. Evaluate on the sealed test set

1. In the **Training runs** list, open the run (trials are named `t000`, `t001`, …).
2. Check the **Training curves** and the **Training diagnosis** (overfitting, underfitting, learning
   rate, etc.).
3. Click **Evaluate on test**. You see the metrics on the sealed test set (for churn: `roc_auc`,
   `accuracy`, precision, recall, F1…) and the **Confusion matrix**.

Since the sealed test set took part in no decision, this is the honest estimate of how the model will
perform on new customers.

## 9. Register the model

Click **Register model**. The button changes to **Model registered** and the model shows up on the
**Models** tab with its test metrics. From there you can deploy and monitor it
([Monitoring](monitoreo.md)).

## 10. Try it and export it

On the same run page:

- **Advanced evaluation:** errors, explanation, fairness, robustness and report.
- **Export the model:** leave **ONNX** checked and click **Export**. Perceptron verifies that the ONNX
  model gives the same results as PyTorch. Then you can download the **Inference server (Docker)** or
  the **Code project**.
- **Try the model** (playground, shown once there is a verified ONNX export): click **Fill with an
  example**, change values and click **Predict**. You see the prediction, the confidence and, with
  **Explain this prediction**, the contribution of each variable.

All of this is covered in [Evaluation and use](evaluacion-y-uso.md).

## What next?

- Try the project **Wizard**: the same path in 9 steps, with the copilot explaining each one.
- Launch the **Agent** to let it iterate on its own within a budget.
- Compare runs in **Experiments** by selecting two or more and clicking **Compare**.
