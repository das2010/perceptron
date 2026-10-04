# Training

There are three ways to train, from most guided to most automatic:

| Path | Where | When it fits |
|---|---|---|
| **Wizard** | **Wizard** tab | First time with a problem; you want to understand each decision |
| **Train** | **Train** tab | You know what you want: preparation → architecture → search → **Train now** |
| **Agent** | **Agent** tab | You want the LLM to iterate on its own within a budget |

## Wizard and copilot

The wizard has nine steps: **Goal → Data → Quality → Labeling → Task and metric → Architecture →
Hyperparameter search → Budget and hardware → Review and launch**. Move with **Next** and **Back**. The
state is saved automatically: you can close it and continue another day.

- **Goal:** "What do you want to achieve?" Describe it as you would to a colleague.
- **Task and metric:** kind of problem, **Metric to optimize** (if you don't pick one, "Automatic
  (val_loss)") and an optional **Success threshold**.
- **Budget and hardware:** **Max time (minutes)**, **Device** ("The recommended one" or a specific one)
  and **Mode**: **Guided (you decide)** or **Autonomous agent**.
- **Review and launch:** summarizes everything before training.

Every step has a **Why?** button that asks the copilot why the recommendation fits your data.

**The copilot** is the panel on the right. Write to it in natural language ("I want to prioritize not
losing premium customers") and it answers in streaming. When it proposes changes to the draft (goal,
target column, task, metric, threshold, trials, epochs, time, autonomous mode), they show up under
**Suggested changes to the draft** with **Accept** / **Dismiss**.

Without an LLM configured, or with privacy L0, the copilot is unavailable and the wizard keeps working
with rule-based recommendations.

### Case sheet and adapted plan

In **Goal** there is the **Case sheet**: the kind of problem, what is predicted, which error is worse
(and by how much), whether there are dates or repeated entities, whether the inputs vary on their
own, whether you will predict outside the data range and where the result will be used. Fill it in
by hand or **tell the assistant about your case**: it proposes how to fill it in, asks what is
missing and nothing is applied until you press **Accept changes**.

With the sheet and the data, the wizard **adapts the plan**:

- it skips steps you don't need (for example **Labeling** when labels exist) and says why;
- it suggests the task, the metric and the kind of architecture, each with its **Why** and a
  **Use** button;
- it warns before training when something doesn't add up: inputs that vary together when you want
  to discover a rule, a random split with time-ordered data or repeated entities, too few rows.

If the case doesn't fit what Perceptron solves, the wizard says so and continues with the standard
steps.

Also:

- In **Quality**, **Check the case sheet** compares what you said with what the data show (for
  example, inputs declared independent that actually vary together) and proposes corrections to
  accept or discard.
- If the goal is to discover a rule, the **Suggested formula** step appears before training.
- If one error is worse than the other in a two-class classification, **Decision threshold**
  appears: you say how many times worse it is, and the evaluation picks on validation the
  threshold that minimizes the cost and shows it on the test set next to the 50 % one
  (**Threshold by error cost**, on the run page).
- When the plan changes, a **The plan changed** notice says which steps were added or removed and
  which warnings are new or resolved.

## Architectures

### Proposals: by rules or by LLM

**Propose architectures** generates 2 to 4 proposals. Each comes with a rationale, parameter count,
memory and estimated time per epoch.

- **By rules:** taken from the Perceptron catalog according to modality, task and data size. Used
  without an LLM, with L0, when the LLM budget is exhausted or when the LLM failed validation three
  times. The reason is shown ("Rules were used: …").
- **By LLM** (**Suggested by AI**): the architect reads the profile (according to your privacy level),
  the goal and the hardware, and composes catalog blocks.

Every proposal is **validated** before it is shown: schema, shape compatibility, memory against what is
available, and availability and license of pretrained weights.

The catalog covers, among others: MLP, ResNet-MLP and FT-Transformer (tabular); a compact CNN or curated
pretrained models such as EfficientNet (image); detection, U-Net and CRNN for OCR (advanced vision);
TextCNN, BiLSTM and pretrained encoders (text); N-BEATS, LSTM/GRU, TCN, PatchTST and autoencoders
(series); CRNN over spectrograms (audio).

### Visual editor

In **Design → Architectures** (or **Open in the visual editor** from **Train**) you see the architecture
as a block graph:

- **Add block** → **Pick a catalog block**. Connect blocks by dragging from a handle; **Delete** removes
  the selected one.
- Click a block to edit its parameters; mark **Tunable by HPO** on those the search should explore.
- Validation is live: **Valid** or the list of **Validation issues**, plus the parameter and memory
  summary.
- **View as code** shows the equivalent PyTorch code.
- **Save as new** creates another architecture (origin *manual*); the original is untouched.

In the wizard's **Architecture** step, **Define step by step** walks you through **Family → Backbone or
size → Head and loss → Regularization**, with the **Recommended** option marked, **Explain the options**
(asks the copilot), **Fill in with the recommended** and **Create this architecture**.

### Expert mode (code)

From the editor, **Expert mode (code)** opens an editor where you write (or the LLM proposes) the model
as PyTorch code with a `build_model(config)` function.

- The code is **not declarative**: Perceptron cannot explain or validate it like an ArchSpec.
- It runs in a **sandbox** with no network, no processes and no file access outside the run, with
  memory and time limits. The linter flags issues live.
- To save it, check "I understand the code is not declarative and runs in a sandbox" and click **Test
  and save**: the model is built in the sandbox and its parameter count is reported.
- The run is marked **Expert code (not declarative)**. To change it, **Edit as new**.

## Hyperparameter search (HPO)

**Recommend strategy** analyzes your scenario (data size, cost per trial, budget, hardware) and returns a
**Search strategy**, the **pruning** and the **Hyperparameters to tune**. With an LLM, the strategist
recommends it; without one, the rules do. You can dismiss it.

| Strategy | When it fits |
|---|---|
| Single evaluation | Very low budget or a known recipe |
| Random search | Large spaces with little budget; baseline |
| Grid search | Few discrete hyperparameters |
| **TPE** (default) | General case |
| CMA-ES | Continuous spaces, medium or high budget |
| NSGA-II (multi-objective) | Balance quality against size or latency |
| Pruning: median, ASHA, Hyperband | Long trainings where the first epochs already predict the outcome |

**Budget:** **Trials**, **Max epochs per trial** and, in the wizard, **Max time** and the target metric.
Whichever comes first stops the search. An interrupted study can be resumed.

By default every training uses mixed precision when the hardware allows it, early stopping, best and
last checkpoints, fixed seeds and automatic batch size.

## Experiments

In **Experiments**:

**Studies** lists each search with its status: training, queued, stopped, interrupted (for example, when the worker restarted), finished or failed. **Stop** halts a study keeping the finished trials; **Resume** continues where it left off.

- **Live training:** epoch, metrics and **Validation curves per trial** while it runs.
- **Training runs:** each run with its **Status** (Queued, Training, Paused, Finished, Failed, Cancelled)
  and **Started**.
- **Compare:** select two or more runs and click **Compare**. You get the overlaid curves (pick the
  **Curve metric**) and a table highlighting the hyperparameters that differ.

On each run page, the **Training diagnosis** detects overfitting, underfitting, divergence or a badly
tuned learning rate and suggests actions (by LLM or **rule-based**).

## Autonomous agent

In **Agent** the LLM acts as an ML engineer: it proposes architecture and strategy, trains, diagnoses and
proposes again, until the goal is reached or the budget runs out.

1. Set the limits: **Max trials (total)**, **Max iterations** and **Max LLM cost (USD)**.
2. Pick the **Human approval**: **Never**, **Before each iteration**, **When the architecture family
   changes** or **After 50% of the budget**.
3. Click **Launch agent**.

While it works you see the live **Log** (decisions, suggestions, limits, approvals), iterations, trials
and cost. When it needs permission, **The agent asks for your approval** appears. **Stop** ends it,
keeping the best model so far.

- The system, not the LLM, enforces the limits.
- The agent **never sees the sealed test set**: it is opened only at the end, under **Result on the
  sealed test set**. **Open the best run** takes you there to register it.
- If the LLM fails, the agent runs a rule-based iteration ("Used rules") or closes safely.

The architecture mini-tournament (train each proposal on a short budget and continue with the best) is
available through the CLI and the API; in the UI it is **coming soon**.

## Where training runs

- **On your machine:** on the recommended device (NVIDIA CUDA GPU, AMD ROCm on Linux, Intel XPU or CPU)
  or the one you pick in **Device**. Each run runs in a separate process; if it runs out of memory or
  crashes, it is reported with a diagnosis.
- **On a Team Server from the desktop:** connect the server in **Settings → Team servers**. In the
  **Train** step, **Where to train** offers **On this machine** or **On "server"**; check **Use the
  server GPU** to go to a GPU worker. The project is uploaded to the server and you see live progress
  just like locally.
- **In the Team Server web UI:** trainings go straight to the worker queue. See
  [Team Server](team-server.md#training-queue).

Multi-GPU on a single machine and one trial per GPU in parallel are **coming soon**.
