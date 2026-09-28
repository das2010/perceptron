# Data

Everything starts on the project's **Data** tab: you load a source, pick the target column and create a
**data version**. Perceptron profiles it, warns you about quality issues and splits it into train,
validation and test.

## Sources

### Files and folders

Under **Add data**:

- **Upload file:** CSV, TSV, Excel (`.xlsx`), Parquet, JSON / JSON Lines or a ZIP.
- **Upload folder:** a folder with **one subfolder per class**, for example `defects/ok/*.png` and
  `defects/fault/*.png`. It works for images, audio (WAV, FLAC, MP3, OGG) and text (`.txt`).

Perceptron recognizes the source type and shows it in the preview:

| Type | Example |
|---|---|
| Table | CSV or Excel with one row per case |
| Image folder | Subfolders per class, or images with COCO, Pascal VOC or YOLO annotations (detection) |
| Text folder | Subfolders per class with `.txt` files (or a CSV with a text column) |
| Audio folder | Subfolders per class with audio files |
| Images with masks | Images and PNG masks (segmentation) |
| Images with text | Images with a CSV of transcriptions (OCR) |

The audio events CSV (start, end, label) is **coming soon**.

In the **Preview** you check the first rows and the inferred type of each column (Numeric, Categorical,
Yes/No, Date, Text, Identifier, File). In **Target column** you pick what the model must predict; if
you leave it at "Detect automatically", it is inferred. Then click **Create data version**.

### Databases

**From a database** connects to **PostgreSQL**, **MySQL / MariaDB**, **SQL Server** or **SQLite
(file)**:

1. Pick the **Engine** and fill in **Host**, **Port**, **Database**, **User** and **Password**.
2. Write the **SQL query**. It is read-only; it is read in batches and the result is saved as a copy
   inside the project.
3. Click **Run and preview** and continue as with a file.

The password is stored in the system keychain, never in the project.

### Public datasets (Hugging Face / Kaggle)

**Public dataset (Hugging Face / Kaggle)**: pick the **Source**, type the **Dataset** name (and the
**Split** on Hugging Face) and click **Download and preview**. The **Hugging Face token** is optional;
Kaggle needs your **Kaggle credentials (user:key)**. Credentials go to the system keychain.

### Server sources (Team Server)

The Team Server web UI has no access to the folders on your machine: you upload files or use **Server
sources**, folders the Admin mounts on the server. Browse the **Path** and click **Use**. See
[Team Server](team-server.md).

### APIs and streaming

To feed [automatic retraining](monitoreo.md#automatic-retraining), a project can have sources that
accumulate new data:

- **REST API** with pagination (offset, page, cursor or "next" link) and header or token
  authentication; the JSON is flattened into a table.
- **WebSocket** with JSON messages.
- **Growing JSON Lines file** (for example, a log): read from the last position.

New data accumulates in a buffer and is polled at an interval. For now these sources are configured
through the Engine API; the screen to create them is **coming soon**. **Kafka** and **MQTT** are also
**coming soon**.

## Data versions

Every load creates an immutable **data version**, identified by the hash of its content. The **Data
versions** table shows the version, modality, number of samples, target and date; **Show profile** opens
its profile.

- If the data changes, another version is created. Earlier runs and models keep pointing to theirs, so a
  result can always be reproduced.
- Each version keeps its **lineage** (which version it comes from and with which transformation, for
  example when applying labels or appending new rows for retraining).
- Two versions can be **compared**: added and removed rows, schema changes, per-column distribution
  changes and differing files. For now, through the Engine API; the UI view is **coming soon**.

## Splits and the sealed test set

When the version is created, the data is split into **train**, **validation** and **test**:

- By default, random **stratified** (keeps the class proportions).
- If the target is numeric and there is a date column, **temporal** (the test set is the most recent
  data).
- Time series are split by time within each series.
- From the CLI or the API you can request a **group** split (so the same customer is never in both train
  and test) or **k-fold**.

!!! important "The test set is sealed"
    The test set is **not used** to pick the architecture, the hyperparameters or the best trial; the
    autonomous agent doesn't see it either. It is only opened when you click **Evaluate on test** (or
    when the agent closes), so the final metric is an honest estimate.

## Profile and quality alerts

The **Dataset profile** summarizes:

- **Splits:** the size of each one.
- **Class distribution** and the minority/majority ratio.
- **Columns:** type, nulls, distinct values and a summary (mean and std for numeric columns).
- Depending on the modality: resolution, channels, corrupt files and near-duplicates (images); language,
  length and duplicates (text); frequency, gaps, seasonality and trend (series); duration, sample rate,
  silence and clipping (audio).

**Quality alerts** have three severities: **Info**, **Warning** and **Important**. The most common:

| Alert | What it means | What to do |
|---|---|---|
| Imbalance | One class has far fewer cases than another | Training can compensate (class weights, oversampling); look at recall or F1, not only accuracy |
| Possible leakage | A column is almost equal to the target, is an identifier or carries future dates | Remove it: the model would look perfect and fail in production |
| Constant column | It has a single value | It is dropped during preparation |
| Insufficient data | Too few cases for the task | Get more data or lower expectations |

This profile is also what the LLM sees at privacy level L1: aggregated statistics only, never
individual values. See [LLM and privacy](llm-y-privacidad.md).

## Preparation (pipeline)

**Propose preparation** (in **Train** or in the wizard) builds a pipeline from the profile: imputation,
category encoding, scaling, tokenization, image resizing, spectrograms, series windows, etc., with a
rationale per step. The pipeline is fitted on train only and travels with the model when you export it.

In **Design → Data pipelines** you can open it in the visual editor: **Add step**, **Move up** / **Move
down**, remove, change **Columns** and **Parameters (JSON)**, and see the **Preview on train** after each
step (**Show result after "step"**) before you **Save**.

## Assisted labeling

If your data has no labels (or few), use the **Labeling** tab:

1. Pick the **Data version**, the **Type** (**Class**, **Multi-label** or **Boxes**) and the comma
   separated **Classes**. Click **New set**.
2. In the **Review queue** you label sample by sample. Pick the **Queue order**: **Most uncertain first**
   (active learning), **Diverse** or **Random**. Shortcuts: **Accept (Enter)**, **Skip**, **Undo**.
3. Once you have trained a first model, pick it in **Model to pre-label with** and click **Pre-label**:
   each sample gets a suggestion with its confidence.
4. With **Minimum confidence** and **Accept confident suggestions** you accept the safe suggestions in
   bulk and only review the uncertain ones by hand.
5. **Apply labels (new version)** creates a new data version with the labels. From there, **Retrain** and
   repeat the cycle.

You can also export labels to CSV or JSONL (and COCO, YOLO or VOC for boxes). Perceptron reports the
human–model agreement and flags possible label errors.

Pre-labeling with the LLM (for text, at privacy L2 or L3) and the LLM-generated labeling guide are
available through the API. Brush masks, temporal segments for audio and series, and pre-labeling with
local zero-shot models are **coming soon**.
