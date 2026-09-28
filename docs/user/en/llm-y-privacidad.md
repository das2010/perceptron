# LLM and privacy

Perceptron uses an LLM as copilot, architect, hyperparameter strategist, diagnostician, agent, report
writer and labeling assistant. **It is optional:** without an LLM (or with privacy L0) everything works
with rule-based recommendations.

Perceptron never calls the provider directly from each feature: all calls go through a single point (the
*gateway*) that applies your privacy level, validates the answers, controls the cost and records
everything in the audit log.

## Providers

| Provider | Runs on | Key |
|---|---|---|
| **Anthropic Claude** (default profile) | Cloud | Yes |
| OpenAI | Cloud | Yes |
| Google Gemini | Cloud | Yes |
| Kimi (Moonshot) | Cloud | Yes |
| **Ollama** | Local (`http://127.0.0.1:11434`) | No |
| **OpenAI-compatible** (LM Studio, vLLM, llama.cpp server) | Local (by default `http://127.0.0.1:1234/v1`) | No |

### Setting it up

In **Settings**:

1. **Providers** lists each provider, where it **Runs on** (Local / Cloud), the **Catalog models** and the
   **Key** status (No key needed / Key stored / No key).
2. Paste the provider key ("Paste the key (it is never shown again)") and click **Save**.
3. Under **LLM profile → Active profile** pick the profile to use. A profile defines which model serves
   each purpose: Copilot, Architect, HPO strategist, Diagnosis, Agent, Report and Labeling.
4. Click **Test connection**. You will see, for example, "Connected: provider/model in 2 s".

Models and their prices live in a **configuration catalog** that is updated without reinstalling (the
Admin can replace it with `PERCEPTRON_LLM__CATALOG_FILE`). Each model declares what it can do; if a model
does not accept images, for example, Perceptron does not send it images.

### Local LLM (Ollama or LM Studio)

With a local model, data never leaves your machine or your network:

1. Install Ollama or LM Studio and download a model. Prefer one that follows structured JSON output.
2. Keep it running at the address in the table.
3. In Perceptron, pick the local provider's profile and click **Test connection**.

A small model on CPU works, but it can take minutes per call with long prompts.

## Privacy levels (L0–L3)

Each project has its own level, chosen when you create it under **Privacy towards the LLM**. The level
shows as a badge next to the project name.

| Level | What the LLM sees | Example with the churn table | When to use it |
|---|---|---|---|
| **L0 — No LLM** | Nothing. Rules only | — | Highly sensitive data and no local LLM |
| **L1 — Metadata** (default) | Schema, aggregated statistics, class distribution, sizes, metrics and curves. **Never individual values** | "column `edad`: numeric, mean 47, 0% nulls" | Most projects |
| **L2 — Anonymized samples** | L1 + a few rows (5 by default) with personal data masked | L1 + rows like "age 52, plan premium, email ‹EMAIL›" | When the LLM needs to understand the content (categories, examples) |
| **L3 — Raw samples** | L2 without anonymization, including images or audio if the model is multimodal | Rows as they are | Non-sensitive data or a local LLM |

**What L2 masks:** emails, URLs, IP addresses, IBANs, cards, phone numbers and Argentine IDs (DNI,
CUIT/CUIL), plus any custom rules configured. To avoid leaking people's names, **free-text fields are not
sent at L2** unless the Admin configures a name detector (NER) with a compatible license; this is
recorded in the audit log. If you need the LLM to read free text, use L3 with a local LLM.

At every level, classes with very few cases are sent under a pseudonym (translated back on your
machine) and file paths are replaced.

Some features depend on the level: for example, pre-labeling text with the LLM needs L2 or L3. Also:

- Your data content is always sent to the LLM marked as **data**, never as instructions, and answers only
  act through validated schemas.
- Every LLM proposal that changes something (architecture, strategy, draft changes) is validated and
  shown as **Suggested by AI**, for you to accept or dismiss.
- On the Team Server the Admin sets a **maximum level** per workspace (and, optionally, a higher one for
  local LLMs) and the **allowed providers**. No project goes above that level. See
  [Team Server](team-server.md#policies).

## LLM audit

The project's **LLM audit** tab shows **every** LLM call, with a summary ("12 calls · total cost
$0.18"):

| Column | What it shows |
|---|---|
| When | Date and time |
| Purpose | Copilot, Architect, HPO strategist, Diagnosis, Agent, Report or Labeling |
| Provider/model | Where it was sent |
| Level | Privacy level applied |
| Status / Cost | Outcome and estimated cost |

**Show what was sent** shows the **exact** content that left, after filtering, and the **Privacy
filtering** section with what was masked or removed. This lets you check, for example, that no
individual value left at L1. From the CLI: `perceptron llm audit`.

## Costs and budget

- Each call estimates its cost from the catalog prices; the audit log shows the running total.
- **Per-project budget:** USD 5 by default (configurable with `PERCEPTRON_LLM__PROJECT_BUDGET_USD`).
- **Agent budget:** **Max LLM cost (USD)** when launching it.
- The budget is checked before each call: if it is not enough, the call is not sent and Perceptron
  continues **by rules** (you will see the reason in "Rules were used: …").
- Answers are **cached**: the same request with the same model is not paid for again and gives the same
  result.

Per-user and per-workspace LLM quotas on the Team Server are **coming soon**.

## Where keys are stored

- **Desktop:** in the operating system keychain (Credential Manager on Windows, Secret Service on
  Linux). They never end up in the project, the logs or the exports.
- **Team Server:** encrypted on the server with a master key set by the Admin.
- The same applies to database passwords, Hugging Face or Kaggle tokens and alert webhooks.
