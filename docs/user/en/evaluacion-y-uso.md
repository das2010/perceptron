# Evaluation and use of the model

All of this lives on a run page (**Experiments** → click the run). The natural order is: evaluate on
test → analyze → register → export → try.

## Evaluation on the sealed test set

**Evaluate on test** computes the metrics on the sealed test set, which took part in no training
decision. It is the honest estimate of how the model will perform on new data.

| Task | Main metrics |
|---|---|
| Classification | accuracy, precision / recall / F1 (per class, macro and micro), ROC-AUC, PR-AUC, calibration, optimal threshold, **Confusion matrix** |
| Regression | MAE, RMSE, MAPE / sMAPE, R², residuals |
| Forecasting | MAE, RMSE, sMAPE, MASE (compared with a naive forecast), per horizon and per series |
| Anomalies | precision / recall / F1, ROC-AUC, score distribution |
| Object detection | mAP@.5 and mAP@[.5:.95], per class |
| Segmentation | IoU and Dice per class |
| OCR | CER and WER (character and word error rate) |
| Audio | classification metrics |

How to read them, briefly:

- **Imbalanced classes** (like churn): look at recall, F1 or ROC-AUC, not only accuracy. A model that
  always says "won't leave" can reach 90% accuracy and be useless.
- **MASE < 1** in forecasting means the model beats the naive forecast.
- **The confusion matrix** shows which class is confused with which (rows: actual; columns: predicted).

Sound event detection (with start and end) is **coming soon**.

## Advanced evaluation

After evaluating, **Advanced evaluation** appears with five tabs. Everything is computed on the sealed
test set already evaluated, except the global explanation, which uses validation.

### Errors

- Summary: how many errors out of how many cases.
- **Where it performs worst:** data slices (column = value) with the lowest performance.
- **Most frequent confusions:** for example, "“premium” predicted as “basic”: 12".
- **Mispredicted cases**, with the model confidence. Those marked "The label might be wrong" are
  candidate label errors: review them before blaming the model.

### Explanation

**Compute** estimates which variables weigh the most in the predictions (sampled Shapley values on
validation). For images, a heat map shows which areas the model looked at. Explanations for text, audio
and series are **coming soon**.

### Fairness

Mark the sensitive attributes (for example, region or age), pick the **Positive class** and click
**Compare groups**. Per **Group** you see the number of cases, the positive-class rate and the model
metrics, with demographic parity and equal opportunity indicators. If the gap between groups exceeds the
threshold, it is flagged as an alert.

### Robustness

**Test robustness** measures how much the model degrades under perturbations of increasing severity:
noise, swapped categories and missing values (tabular); noise, blur and JPEG compression (image). Text
and audio are **coming soon**.

### Report

**Generate report** (or the **Report** tab) produces a report with an executive summary, what was tried,
why the model won, metrics, limitations, risks and recommendations, plus a **model card**. With an LLM,
the LLM writes it; otherwise a template does. It includes what you already computed (explanation,
fairness, robustness). Download it as **PDF**, **HTML** or **Markdown**, with Preteco branding.

## Registering the model

**Register model** adds it to the **Models** tab with its test metrics. From there you deploy it and
manage the champion. See [Monitoring](monitoreo.md).

## Export

In **Export the model** you check the formats and click **Export**:

| Format | What for |
|---|---|
| **ONNX** | The most portable: ONNX Runtime in Python, C#, Java, JavaScript, etc. Options **ONNX fp16** (smaller, for GPU) and **ONNX INT8 (CPU)** (quantized) |
| **torch.export** | PyTorch exported program, to stay in the PyTorch ecosystem |
| **TorchScript** | Legacy format; use it only if your environment requires it |

Each format is **verified** against PyTorch on real validation data: the **Verification** column shows
the maximum difference and the tolerance (ONNX: 1e-4; fp16: 1e-2). If it fails, it is marked "failed".
All formats include the model **signature**: inputs, outputs, version and hash. The preprocessing
(pipeline) travels with the model.

### Inference server (Docker)

**Inference server (Docker)** (requires a verified ONNX export) downloads a ZIP with a REST API ready to
deploy:

- `Dockerfile.cpu` and `Dockerfile.cuda`, or run without Docker using `uv`.
- Endpoints: `/predict` (JSON), `/predict/batch` (CSV or image), `/signature`, `/metrics` (Prometheus) and
  `/docs` (OpenAPI).
- API key authentication: set it with the `PERCEPTRON_API_KEY` variable and send it in the `X-API-Key`
  header.

The ZIP's `README.md` has the exact commands.

### Code project

**Code project** downloads a standalone Python repository (it doesn't need Perceptron): the model as
PyTorch code generated from the ArchSpec, the pipeline, `config.yaml` with the best trial's
hyperparameters, the trained weights, train and validation data, and commands to retrain, predict, serve
and run a smoke test. Install it with `uv sync`. The sealed test set is not included.

License details for pretrained weights are in `LICENSES.md` inside the project.

## Playground

When there is a verified ONNX export, **Try the model** appears:

- **Tabular:** a form with one input per column. **Fill with an example** loads a real row; change values
  and click **Predict**.
- **Image:** pick the **Image to classify**.

You see the **Prediction**, the **confidence** and the probabilities. **Explain this prediction** shows
the **Contribution of each variable** (tabular) or the **Explanation heat map** (image). The playground
for audio and text is **coming soon**.
