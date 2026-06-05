# CoverPath — Rare Disease Coverage Intelligence

CoverPath automates pharmaceutical companies' responses to draft Medicare Local Coverage Determinations (LCDs) for rare disease drugs. It fetches the draft LCD, gathers external evidence in parallel, runs an LLM reasoning loop to build a formal response brief, and scores the output with four LLM-as-Judge evaluators.

Built as a Databricks App (Flask + React SPA).

---

## Architecture

```
Flask backend (app/)
├── agent/
│   ├── orchestrator.py   — 6-step ReAct workflow
│   ├── tools.py          — CMS, openFDA, ClinicalTrials, PubMed, Databricks SQL
│   └── prompts.py        — System prompts for each step + 4 judge prompts
├── static/index.html     — React SPA (CDN-loaded, no build step)
├── fixtures/             — Demo LCD HTML document
└── app.yaml              — Databricks Apps runtime config

data/
└── generate_data.py      — Synthetic Unity Catalog data generator

databricks.yml            — DABs bundle definition (variables, app resource)
```

**Workflow steps:**

| Step | Description |
|------|-------------|
| 1 | Load & parse draft LCD; extract restrictions, ICD codes, documentation requirements |
| 2 | Parallel external evidence: CMS precedent, FDA label, ClinicalTrials, PubMed, Databricks SQL |
| 3 | Synthesize evidence into structured rebuttal arguments |
| 4 | Draft restriction-by-restriction rebuttals with citations |
| 5 | Generate formal 9-section response brief |
| 6 | Parallel LLM-as-Judge scoring (Completeness, Evidence Strength, Regulatory Tone, Threat Mitigation) |

---

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/databricks-cli.html) v0.221+
- Python 3.10+
- A Databricks workspace with:
  - Unity Catalog enabled
  - A Serverless SQL Warehouse
  - Access to `databricks-claude-sonnet-4-6` (or another Claude model via Model Serving)
  - Permissions to create schemas and tables in a catalog

---

## Setup

### 1. Configure Databricks CLI

```bash
databricks configure
# Enter your workspace URL and personal access token when prompted
```

Verify with:
```bash
databricks current-user me
```

### 2. Find your Warehouse ID

In the Databricks UI: **SQL Warehouses → your warehouse → Connection Details → HTTP Path**

The warehouse ID is the last segment after `/sql/1.0/warehouses/`.

### 3. Generate demo data

```bash
cd data
pip install databricks-sdk
python generate_data.py \
    --warehouse-id <YOUR_WAREHOUSE_ID> \
    --catalog <YOUR_CATALOG> \
    --schema coverpath_demo
```

This creates 5 synthetic Unity Catalog tables with ~1,100 rows total, representing a pivotal complement-inhibitor trial and real-world patient registry.

**Options:**

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--warehouse-id` | `COVERPATH_WAREHOUSE_ID` | *(required)* | SQL Warehouse ID |
| `--catalog` | `COVERPATH_CATALOG` | `coverpath_demo` | UC catalog name |
| `--schema` | `COVERPATH_SCHEMA` | `coverpath_demo` | UC schema name |
| `--profile` | `DATABRICKS_PROFILE` | *(ambient)* | Databricks CLI profile |

### 4. Deploy the app

```bash
cd ..  # repo root
BUNDLE_VAR_warehouse_id=<YOUR_WAREHOUSE_ID> \
BUNDLE_VAR_catalog=<YOUR_CATALOG> \
make deploy
```

`make deploy` does three things in sequence:
1. Runs `databricks bundle validate` to resolve variables, then renders `app/app.yaml` from the template
2. Runs `databricks bundle deploy` to sync the app code and register the resource
3. Runs `databricks apps deploy` with the correct workspace path (derived automatically)

**Bundle variables** — set as `BUNDLE_VAR_<name>=value` env vars or override in `databricks.yml` under `targets.dev.variables`:

| Variable | Default | Description |
|----------|---------|-------------|
| `warehouse_id` | *(required)* | SQL Warehouse ID |
| `catalog` | `coverpath_demo` | UC catalog name |
| `schema` | `coverpath_demo` | UC schema name |
| `model` | `databricks-claude-sonnet-4-6` | LLM serving endpoint |

**Makefile variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFILE` | `prod` | Databricks CLI profile |
| `TARGET` | `dev` | Bundle target |

### 5. Open the app

```bash
databricks apps get coverpath --output json | python -c "import sys,json; d=json.load(sys.stdin); print(d['url'])"
```

---

## Running locally

```bash
cd app
pip install -r requirements.txt

export COVERPATH_WAREHOUSE_ID=<YOUR_WAREHOUSE_ID>
export COVERPATH_CATALOG=<YOUR_CATALOG>
export COVERPATH_SCHEMA=coverpath_demo
export DATABRICKS_HOST=https://<YOUR_WORKSPACE>.cloud.databricks.com
export DATABRICKS_TOKEN=<YOUR_PAT>

python app.py
# → http://localhost:8000
```

---

## Demo flow

1. Open the app — the LCD document is pre-loaded on the Dashboard
2. Fill in **MAC Medical Director** contact details in Settings (optional)
3. Click **Run Analysis** on the Dashboard
4. Watch the 6-step workflow animate in real time
5. After completion, the **Quality Review** tab opens automatically with judge scores
6. Click **Brief** to read the generated response document
7. Click **Mark for Counsel Review** to escalate

---

## Tables

| Table | Rows | Description |
|-------|------|-------------|
| `trial_enrollment` | 127 | Pivotal-trial patients (89 outpatient, 38 inpatient) |
| `trial_outcomes` | ~630 | Per-patient outcomes at weeks 4, 8, 12, 26, 52 |
| `trial_adverse_events` | ~210 | AEs (zero meningococcal infections by design) |
| `registry_patients` | 43 | Post-approval real-world registry (31 outpatient) |
| `registry_outcomes` | 172 | Registry outcomes at months 1, 3, 6, 12 |
