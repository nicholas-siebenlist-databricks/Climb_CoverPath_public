"""
Synthetic data generator for CoverPath demo tables.

Usage:
  python data/generate_data.py \\
      --warehouse-id <WAREHOUSE_ID> \\
      --catalog <CATALOG> \\
      --schema <SCHEMA> \\
      [--profile <DATABRICKS_PROFILE>]

Env vars (override defaults, overridden by CLI flags):
  COVERPATH_WAREHOUSE_ID  — required
  COVERPATH_CATALOG       — default: coverpath_demo
  COVERPATH_SCHEMA        — default: coverpath_demo
  DATABRICKS_PROFILE      — optional; omit to use ambient SDK credentials

Creates 5 tables with data matching CoverPath PRD constraints exactly.
Random seed=42 ensures deterministic output.
"""
import argparse
import os
import random
import sys
import time
from datetime import date, timedelta

random.seed(42)


# ── Databricks SDK connection ─────────────────────────────────────────────────

def get_client(profile=None):
    from databricks.sdk import WorkspaceClient
    if profile:
        return WorkspaceClient(profile=profile)
    return WorkspaceClient()


def run_sql(w, stmt: str, warehouse_id: str, catalog: str, schema: str, label: str = ""):
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Running SQL...", end=" ", flush=True)
    t0 = time.time()
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=stmt,
        catalog=catalog,
        schema=schema,
        wait_timeout="50s",
    )
    state = resp.status.state.value if resp.status and resp.status.state else "UNKNOWN"
    elapsed = round(time.time() - t0, 1)
    if state == "SUCCEEDED":
        print(f"✓ ({elapsed}s)")
    else:
        err = resp.status.error.message if resp.status and resp.status.error else "unknown error"
        print(f"✗ {state}: {err}")
        raise Exception(f"SQL failed ({state}): {err}")
    return resp


def insert_batch(w, table: str, columns: list, rows: list, warehouse_id: str, catalog: str, schema: str, batch_size: int = 200):
    total = len(rows)
    inserted = 0
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        vals = []
        for row in batch:
            formatted = []
            for v in row:
                if v is None:
                    formatted.append("NULL")
                elif isinstance(v, bool):
                    formatted.append("TRUE" if v else "FALSE")
                elif isinstance(v, (int, float)):
                    formatted.append(str(v))
                else:
                    escaped = str(v).replace("'", "''")
                    formatted.append(f"'{escaped}'")
            vals.append(f"({', '.join(formatted)})")
        col_list = ", ".join(columns)
        stmt = f"INSERT INTO {table} ({col_list}) VALUES {', '.join(vals)}"
        run_sql(w, stmt, warehouse_id, catalog, schema,
                f"INSERT {table} rows {inserted+1}-{min(inserted+len(batch), total)}/{total}")
        inserted += len(batch)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def rand_normal_int(mean: float, sd: float, lo: int, hi: int) -> int:
    v = mean + random.gauss(0, 1) * sd
    return max(lo, min(hi, round(v)))


# ── Table 1: trial_enrollment (127 rows) ──────────────────────────────────────

def gen_trial_enrollment():
    n = 127
    rows = []

    settings = ["Outpatient"] * 89 + ["Inpatient"] * 38
    random.shuffle(settings)

    pe_dist = [0] * 101 + [1] * 19 + [2] * 7
    random.shuffle(pe_dist)

    apoc3_flags = [True] * 41 + [False] * 86
    random.shuffle(apoc3_flags)

    severity_pool = (["Moderate"] * 57 + ["Severe"] * 51 + ["Critical"] * 19)
    random.shuffle(severity_pool)

    country_pool = (["US"] * 99 + ["Germany"] * 10 + ["UK"] * 8 + ["Japan"] * 6 + ["Canada"] * 4)
    random.shuffle(country_pool)

    site_assignments = []
    for site in range(1, 19):
        site_assignments.append(f"SITE-{site:02d}")
    while len(site_assignments) < 127:
        site_assignments.append(f"SITE-{random.randint(1,18):02d}")
    random.shuffle(site_assignments[:127])

    enroll_start = date(2021, 4, 1)
    enroll_end = date(2022, 9, 30)

    for i in range(n):
        pid = f"SUV-{i+1:04d}"
        rows.append((
            pid,
            site_assignments[i],
            rand_date(enroll_start, enroll_end).isoformat(),
            rand_normal_int(42, 14, 18, 78),
            "Female" if i < 74 else "Male",
            severity_pool[i],
            settings[i],
            pe_dist[i],
            apoc3_flags[i],
            country_pool[i],
        ))
    return rows


# ── Table 2: trial_outcomes (635 rows = 127 × 5) ─────────────────────────────

def gen_trial_outcomes(enrollment_rows):
    WEEKS = [4, 8, 12, 26, 52]
    rows = []
    out_id = 1

    outpatient_pids = [r[0] for r in enrollment_rows if r[6] == "Outpatient"]
    inpatient_pids  = [r[0] for r in enrollment_rows if r[6] == "Inpatient"]

    all_pids = [r[0] for r in enrollment_rows]
    disc_pids = set(random.sample(all_pids, 6))
    disc_reasons = {}
    disc_pids_list = list(disc_pids)
    for i, pid in enumerate(disc_pids_list):
        if i < 3:
            disc_reasons[pid] = "Adverse Event"
        elif i < 5:
            disc_reasons[pid] = "Withdrawal"
        else:
            disc_reasons[pid] = "Loss to Follow-up"
    disc_week = {pid: random.choice([4, 8, 12]) for pid in disc_pids}

    op_responders_w26 = set(random.sample(outpatient_pids, 76))
    ip_responders_w26 = set(random.sample(inpatient_pids, 31))
    all_responders_w26 = op_responders_w26 | ip_responders_w26

    responders_w52 = set(random.sample(list(all_responders_w26), 95))
    dialysis_baseline = set(random.sample(all_pids, 28))
    severity_map = {r[0]: r[5] for r in enrollment_rows}

    def plat(pid, week, is_responder):
        if week == 4:
            base = 80000 if pid in dialysis_baseline else random.randint(150000, 220000)
            return int(base + random.gauss(0, 12000))
        if is_responder and week >= 12:
            return int(random.gauss(230000, 20000))
        return int(random.gauss(90000, 15000))

    def ldh(pid, week, is_responder):
        sev = severity_map.get(pid, "Moderate")
        base = {"Critical": 420, "Severe": 320, "Moderate": 240}.get(sev, 280)
        if week == 4:
            return round(base + random.gauss(0, 40), 1)
        if is_responder and week >= 12:
            return round(random.gauss(155, 20), 1)
        return round(base * 0.9 + random.gauss(0, 30), 1)

    def creat(pid, week, is_responder):
        sev = severity_map.get(pid, "Moderate")
        base = {"Critical": 2.4, "Severe": 1.8, "Moderate": 1.2}.get(sev, 1.5)
        if week == 4:
            return round(base + random.gauss(0, 0.3), 2)
        if is_responder and week >= 26:
            return round(random.gauss(1.1, 0.2), 2)
        return round(base * 0.85 + random.gauss(0, 0.25), 2)

    for r in enrollment_rows:
        pid = r[0]
        for week in WEEKS:
            if pid in disc_pids and week > disc_week[pid]:
                continue

            is_discontinued = pid in disc_pids and week == disc_week[pid]

            if week == 26:
                tma_response = pid in all_responders_w26
            elif week == 52:
                tma_response = pid in responders_w52
            elif week == 4:
                tma_response = None
            else:
                if pid in all_responders_w26:
                    tma_response = week >= 12
                else:
                    tma_response = False

            on_dialysis = (pid in dialysis_baseline and week == 4) or (
                week <= 8 and pid in dialysis_baseline and not (pid in all_responders_w26)
            )
            if week == 26 and tma_response:
                on_dialysis = False

            rows.append((
                f"OUT-{out_id:05d}",
                pid,
                week,
                tma_response,
                plat(pid, week, pid in all_responders_w26),
                ldh(pid, week, pid in all_responders_w26),
                creat(pid, week, pid in all_responders_w26),
                on_dialysis,
                is_discontinued,
                disc_reasons.get(pid) if is_discontinued else None,
            ))
            out_id += 1

    return rows


# ── Table 3: trial_adverse_events (~210 rows) ─────────────────────────────────

def gen_adverse_events(enrollment_rows):
    all_pids = [r[0] for r in enrollment_rows]
    rows = []
    ae_id = 1

    mild_terms = [
        ("Headache", 0.25), ("Nasopharyngitis", 0.20), ("Back pain", 0.15),
        ("Nausea", 0.15), ("Hypertension", 0.10), ("Fatigue", 0.15),
    ]

    patients_with_ae = random.sample(all_pids, 95)
    for pid in patients_with_ae:
        n_aes = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_aes):
            term = random.choices([t for t, _ in mild_terms], weights=[w for _, w in mild_terms])[0]
            grade = random.choices([1, 2], weights=[0.44, 0.56])[0]
            rows.append((
                f"AE-{ae_id:05d}", pid, term, grade, False, False,
                random.randint(1, 365), True,
            ))
            ae_id += 1

    serious_patients = random.sample(all_pids, 12)
    serious_distribution = (
        [("Hypertension", 3)] * 5 + [("Urinary tract infection", 4)] * 4 +
        [("Infusion-related reaction", 3)] * 3
    )
    for i, pid in enumerate(serious_patients):
        term, grade = serious_distribution[i]
        rows.append((
            f"AE-{ae_id:05d}", pid, term, grade, True, True,
            random.randint(14, 90), random.random() < 0.95,
        ))
        ae_id += 1

    # CRITICAL: zero meningococcal infection rows

    target = 210
    if len(rows) > target:
        rows = rows[:target]

    return rows


# ── Table 4: registry_patients (43 rows) ─────────────────────────────────────

def gen_registry_patients():
    n = 43
    rows = []

    settings = ["Outpatient"] * 31 + ["Inpatient"] * 12
    random.shuffle(settings)

    cgs_states = ["MO", "MO", "MO", "KS", "KS", "NE", "NE", "IA"]
    other_states = (
        ["CA"] * 6 + ["NY"] * 5 + ["TX"] * 5 + ["FL"] * 4 + ["PA"] * 3 +
        ["MA"] * 3 + ["IL"] * 3 + ["OH"] * 2 + ["WA"] * 2 + ["CO"] * 2
    )
    non_cgs = random.sample(other_states, min(35, len(other_states)))
    while len(non_cgs) < 35:
        non_cgs.append(random.choice(other_states))
    all_states = cgs_states + non_cgs
    random.shuffle(all_states)

    pe_dist = [0] * 32 + [1] * 9 + [2] * 2
    random.shuffle(pe_dist)

    enroll_start = date(2024, 3, 1)
    enroll_end = date(2025, 12, 31)
    sex_pool = ["Female"] * 26 + ["Male"] * 17
    random.shuffle(sex_pool)

    for i in range(n):
        enroll_date = rand_date(enroll_start, enroll_end)
        treat_start = enroll_date + timedelta(days=random.randint(0, 14))
        rows.append((
            f"REG-{i+1:03d}",
            enroll_date.isoformat(),
            rand_normal_int(44, 16, 18, 80),
            sex_pool[i],
            all_states[i],
            settings[i],
            treat_start.isoformat(),
            pe_dist[i],
        ))
    return rows


# ── Table 5: registry_outcomes (172 rows = 43 × 4) ───────────────────────────

def gen_registry_outcomes(registry_rows):
    MONTHS = [1, 3, 6, 12]
    rows = []
    rgo_id = 1

    all_pids = [r[0] for r in registry_rows]
    responders_12mo = set(random.sample(all_pids, 38))
    discontinued_pids = random.sample(all_pids, 2)

    for r in registry_rows:
        pid = r[0]
        is_outpatient = r[5] == "Outpatient"

        for month in MONTHS:
            if month == 12:
                sustained = pid in responders_12mo
            elif month in (3, 6):
                sustained = pid in responders_12mo
            else:
                sustained = None

            on_treatment = True
            if month == 12 and pid in discontinued_pids:
                on_treatment = False

            if is_outpatient:
                # CRITICAL: all outpatient patients — zero hospitalizations, zero serious AEs
                any_serious_ae = False
                hospitalization_required = False
            else:
                any_serious_ae = random.random() < 0.1
                hospitalization_required = random.random() < 0.08

            dialysis_free = None
            if month >= 6 and sustained is True:
                dialysis_free = True
            elif month >= 6 and sustained is False:
                dialysis_free = False

            rows.append((
                f"RGO-{rgo_id:03d}",
                pid,
                month,
                sustained,
                on_treatment,
                any_serious_ae,
                hospitalization_required,
                dialysis_free,
            ))
            rgo_id += 1

    return rows


# ── DDL ───────────────────────────────────────────────────────────────────────

def get_ddl(catalog, schema):
    return {
        "trial_enrollment": f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.trial_enrollment (
  patient_id         STRING    NOT NULL,
  site_id            STRING    NOT NULL,
  enrollment_date    DATE      NOT NULL,
  age_at_enrollment  INT       NOT NULL,
  sex                STRING    NOT NULL,
  disease_severity   STRING    NOT NULL,
  initiation_setting STRING    NOT NULL,
  prior_pe_courses   INT       NOT NULL,
  apoc3_variant      BOOLEAN   NOT NULL,
  country            STRING    NOT NULL
) USING DELTA
COMMENT 'Pivotal-trial enrollment roster, one row per patient (n=127). Synthetic.'
""",
        "trial_outcomes": f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.trial_outcomes (
  outcome_id             STRING    NOT NULL,
  patient_id             STRING    NOT NULL,
  visit_week             INT       NOT NULL,
  complete_tma_response  BOOLEAN,
  platelet_count_k_ul    DOUBLE,
  ldh_iu_l               DOUBLE,
  serum_creatinine_mg_dl DOUBLE,
  on_dialysis            BOOLEAN,
  discontinued           BOOLEAN   NOT NULL,
  discontinuation_reason STRING
) USING DELTA
COMMENT 'Pivotal-trial outcomes, one row per patient per timepoint. Timepoints: weeks 4, 8, 12, 26, 52.'
""",
        "trial_adverse_events": f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.trial_adverse_events (
  ae_id              STRING    NOT NULL,
  patient_id         STRING    NOT NULL,
  ae_term            STRING    NOT NULL,
  ae_grade           INT       NOT NULL,
  serious            BOOLEAN   NOT NULL,
  drug_related       BOOLEAN   NOT NULL,
  onset_day_relative INT       NOT NULL,
  resolved           BOOLEAN   NOT NULL
) USING DELTA
COMMENT 'Pivotal-trial adverse events. Critical: zero meningococcal infection rows.'
""",
        "registry_patients": f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.registry_patients (
  registry_id          STRING    NOT NULL,
  enrollment_date      DATE      NOT NULL,
  age                  INT       NOT NULL,
  sex                  STRING    NOT NULL,
  treating_state       STRING    NOT NULL,
  initiation_setting   STRING    NOT NULL,
  treatment_start_date DATE      NOT NULL,
  prior_pe_courses     INT       NOT NULL
) USING DELTA
COMMENT 'Post-approval real-world patient registry. One row per patient (n=43).'
""",
        "registry_outcomes": f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.registry_outcomes (
  registry_outcome_id      STRING    NOT NULL,
  registry_id              STRING    NOT NULL,
  months_from_start        INT       NOT NULL,
  sustained_tma_response   BOOLEAN,
  on_treatment             BOOLEAN   NOT NULL,
  any_serious_ae           BOOLEAN   NOT NULL,
  hospitalization_required BOOLEAN   NOT NULL,
  dialysis_free            BOOLEAN
) USING DELTA
COMMENT 'Post-approval registry outcomes, one row per patient per timepoint. Timepoints: months 1, 3, 6, 12.'
""",
    }


TABLE_COLUMNS = {
    "trial_enrollment": [
        "patient_id", "site_id", "enrollment_date", "age_at_enrollment", "sex",
        "disease_severity", "initiation_setting", "prior_pe_courses", "apoc3_variant", "country",
    ],
    "trial_outcomes": [
        "outcome_id", "patient_id", "visit_week", "complete_tma_response",
        "platelet_count_k_ul", "ldh_iu_l", "serum_creatinine_mg_dl",
        "on_dialysis", "discontinued", "discontinuation_reason",
    ],
    "trial_adverse_events": [
        "ae_id", "patient_id", "ae_term", "ae_grade", "serious",
        "drug_related", "onset_day_relative", "resolved",
    ],
    "registry_patients": [
        "registry_id", "enrollment_date", "age", "sex", "treating_state",
        "initiation_setting", "treatment_start_date", "prior_pe_courses",
    ],
    "registry_outcomes": [
        "registry_outcome_id", "registry_id", "months_from_start",
        "sustained_tma_response", "on_treatment", "any_serious_ae",
        "hospitalization_required", "dialysis_free",
    ],
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CoverPath synthetic demo data in Unity Catalog.")
    parser.add_argument("--warehouse-id", default=os.environ.get("COVERPATH_WAREHOUSE_ID", ""),
                        help="Databricks SQL Warehouse ID (env: COVERPATH_WAREHOUSE_ID)")
    parser.add_argument("--catalog", default=os.environ.get("COVERPATH_CATALOG", "coverpath_demo"),
                        help="Unity Catalog catalog name (env: COVERPATH_CATALOG)")
    parser.add_argument("--schema", default=os.environ.get("COVERPATH_SCHEMA", "coverpath_demo"),
                        help="Unity Catalog schema name (env: COVERPATH_SCHEMA)")
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", ""),
                        help="Databricks CLI profile (env: DATABRICKS_PROFILE; omit for ambient credentials)")
    args = parser.parse_args()

    warehouse_id = args.warehouse_id
    catalog = args.catalog
    schema = args.schema
    profile = args.profile or None

    if not warehouse_id:
        print("ERROR: --warehouse-id is required (or set COVERPATH_WAREHOUSE_ID env var).")
        sys.exit(1)

    print("=" * 60)
    print("CoverPath — Synthetic Data Generator")
    print(f"Target: {catalog}.{schema}")
    print(f"Warehouse: {warehouse_id}")
    print("=" * 60)

    w = get_client(profile)
    DDL = get_ddl(catalog, schema)

    print("\n1. Creating schema...")
    run_sql(w, f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}", warehouse_id, catalog, schema, "schema")

    print("\n2. Dropping existing tables (if any)...")
    for table in TABLE_COLUMNS:
        run_sql(w, f"DROP TABLE IF EXISTS {catalog}.{schema}.{table}", warehouse_id, catalog, schema, f"drop {table}")

    print("\n3. Creating tables...")
    for table, ddl in DDL.items():
        run_sql(w, ddl, warehouse_id, catalog, schema, f"create {table}")

    print("\n4. Generating synthetic data...")
    enrollment_rows = gen_trial_enrollment()
    outcomes_rows = gen_trial_outcomes(enrollment_rows)
    ae_rows = gen_adverse_events(enrollment_rows)
    registry_rows = gen_registry_patients()
    registry_outcomes_rows = gen_registry_outcomes(registry_rows)

    print(f"  trial_enrollment:      {len(enrollment_rows):>4} rows")
    print(f"  trial_outcomes:        {len(outcomes_rows):>4} rows")
    print(f"  trial_adverse_events:  {len(ae_rows):>4} rows")
    print(f"  registry_patients:     {len(registry_rows):>4} rows")
    print(f"  registry_outcomes:     {len(registry_outcomes_rows):>4} rows")

    print("\n5. Inserting rows...")
    insert_batch(w, f"{catalog}.{schema}.trial_enrollment",    TABLE_COLUMNS["trial_enrollment"],    enrollment_rows,         warehouse_id, catalog, schema)
    insert_batch(w, f"{catalog}.{schema}.trial_outcomes",      TABLE_COLUMNS["trial_outcomes"],      outcomes_rows,           warehouse_id, catalog, schema)
    insert_batch(w, f"{catalog}.{schema}.trial_adverse_events",TABLE_COLUMNS["trial_adverse_events"],ae_rows,                 warehouse_id, catalog, schema)
    insert_batch(w, f"{catalog}.{schema}.registry_patients",   TABLE_COLUMNS["registry_patients"],   registry_rows,           warehouse_id, catalog, schema)
    insert_batch(w, f"{catalog}.{schema}.registry_outcomes",   TABLE_COLUMNS["registry_outcomes"],   registry_outcomes_rows,  warehouse_id, catalog, schema)

    print("\n6. Validating constraints...")
    checks = [
        (f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.trial_enrollment",
         lambda r: int(r[0].get("n", 0)) == 127, "trial_enrollment = 127 rows"),
        (f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.trial_enrollment WHERE initiation_setting='Outpatient'",
         lambda r: int(r[0].get("n", 0)) == 89, "89 outpatient patients"),
        (f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.trial_adverse_events WHERE LOWER(ae_term)='meningococcal infection'",
         lambda r: int(r[0].get("n", 0)) == 0, "ZERO meningococcal infections ✓"),
        (f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.registry_patients WHERE treating_state IN ('MO','KS','NE','IA')",
         lambda r: int(r[0].get("n", 0)) == 8, "8 CGS-jurisdiction registry patients"),
        (f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.registry_outcomes ro JOIN {catalog}.{schema}.registry_patients rp ON ro.registry_id=rp.registry_id WHERE rp.initiation_setting='Outpatient' AND ro.hospitalization_required=TRUE",
         lambda r: int(r[0].get("n", 0)) == 0, "ZERO outpatient hospitalizations ✓"),
    ]

    all_ok = True
    for sql, check_fn, label in checks:
        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id, statement=sql, catalog=catalog,
            schema=schema, wait_timeout="50s",
        )
        result_rows = []
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            result_rows = [dict(zip(cols, row)) for row in resp.result.data_array]
        ok = check_fn(result_rows)
        status = "✓" if ok else "✗ FAILED"
        print(f"  {status} {label}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅  All constraints validated. Data generation complete.")
    else:
        print("⚠️  Some constraints failed — review output above.")
    print(f"Tables live at: {catalog}.{schema}")
    print("=" * 60)


if __name__ == "__main__":
    main()
