import os
import time
import json
import requests
from bs4 import BeautifulSoup

WAREHOUSE_ID = os.environ.get("COVERPATH_WAREHOUSE_ID", "")
CATALOG = os.environ.get("COVERPATH_CATALOG", "coverpath_demo")
SCHEMA = os.environ.get("COVERPATH_SCHEMA", "coverpath_demo")


def _retry(fn, retries=3, backoff=1.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


# ── CMS Coverage Database ─────────────────────────────────────────────────────

def cms_fetch_lcd(lcd_id: str) -> dict:
    """Return structured LCD content. Loads the static fixture for MCD-A-XXXXX."""
    fixture_path = os.path.join(os.path.dirname(__file__), "../fixtures/mcd_a_xxxxx.html")
    with open(fixture_path, "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return {
        "lcd_id": lcd_id,
        "html": html,
        "title": soup.find("title").get_text(strip=True) if soup.find("title") else lcd_id,
        "source": "CMS Coverage Database (fixture)",
    }


def cms_search_precedent(drug_class: str) -> list:
    """Search CMS Coverage Database for prior LCDs in the same drug class."""
    # Returns hard-coded class-analogue results for the demo
    return [
        {
            "lcd_id": "L38291",
            "mac": "Noridian Healthcare Solutions",
            "title": "Eculizumab (Soliris) for Atypical Hemolytic Uremic Syndrome",
            "drug_class": "C5 complement inhibitor",
            "step_therapy_required": False,
            "inpatient_only": False,
            "duration_limit_weeks": None,
            "effective_date": "2019-04-07",
            "url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdId=38291",
            "note": "No step therapy or inpatient requirement for eculizumab in aHUS under Noridian jurisdiction.",
        },
        {
            "lcd_id": "L39104",
            "mac": "Palmetto GBA",
            "title": "Complement Inhibitor Therapy for Complement-Mediated Thrombotic Microangiopathy",
            "drug_class": "C5 complement inhibitor",
            "step_therapy_required": False,
            "inpatient_only": False,
            "duration_limit_weeks": None,
            "effective_date": "2021-11-15",
            "url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdId=39104",
            "note": "Outpatient initiation permitted; no prior plasma exchange requirement documented.",
        },
    ]


# ── openFDA ───────────────────────────────────────────────────────────────────

def openfda_get_label(brand_name: str = "Soliris") -> dict:
    """Fetch FDA drug label from openFDA. Uses eculizumab/Soliris as class analogue."""
    def _call():
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:%22{brand_name}%22&limit=1"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data["results"][0]
        return {
            "brand_name": brand_name,
            "indications_and_usage": result.get("indications_and_usage", [""])[0][:2000],
            "clinical_studies": result.get("clinical_studies", [""])[0][:2000],
            "boxed_warning": result.get("boxed_warning", [""])[0][:1000],
            "warnings_and_precautions": result.get("warnings_and_precautions", [""])[0][:1000],
            "source": "openFDA Drug Labels API",
            "note": "Class analogue (eculizumab) used as public-label reference for Suvaxilumab (same mechanism class).",
        }
    return _retry(_call)


# ── ClinicalTrials.gov ────────────────────────────────────────────────────────

def clinicaltrials_get_study(nct_id: str = "NCT01844856") -> dict:
    """Fetch study record from ClinicalTrials.gov v2 API."""
    def _call():
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        protocol = data.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        design = protocol.get("designModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        outcomes = protocol.get("outcomesModule", {})
        return {
            "nct_id": nct_id,
            "title": ident.get("officialTitle", ident.get("briefTitle", "")),
            "status": design.get("studyType", ""),
            "phase": str(design.get("phases", [])),
            "eligibility_criteria": eligibility.get("eligibilityCriteria", "")[:1500],
            "primary_outcomes": [o.get("measure", "") for o in outcomes.get("primaryOutcomes", [])[:3]],
            "enrollment": design.get("enrollmentInfo", {}).get("count"),
            "source": "ClinicalTrials.gov v2 API",
            "note": "RECOVER trial (NCT01844856) — pivotal eculizumab aHUS study used as class precedent.",
        }
    return _retry(_call)


# ── PubMed ────────────────────────────────────────────────────────────────────

DEMO_PMIDS = ["22553501", "25908597", "31395980"]

def pubmed_fetch(pmids: list = None) -> list:
    """Fetch abstracts from PubMed E-utilities for fixed demo PMIDs."""
    if pmids is None:
        pmids = DEMO_PMIDS

    def _fetch_abstract(pmid):
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={pmid}&retmode=json"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        article = data.get("result", {}).get(pmid, {})
        return {
            "pmid": pmid,
            "title": article.get("title", ""),
            "authors": [a.get("name", "") for a in article.get("authors", [])[:3]],
            "journal": article.get("fulljournalname", article.get("source", "")),
            "pub_date": article.get("pubdate", ""),
            "source": f"PubMed PMID {pmid}",
        }

    results = []
    for pmid in pmids:
        try:
            results.append(_retry(lambda p=pmid: _fetch_abstract(p)))
            time.sleep(0.35)  # NCBI courtesy rate limit
        except Exception as e:
            results.append({"pmid": pmid, "error": str(e), "source": f"PubMed PMID {pmid}"})
    return results


# ── Databricks SQL ─────────────────────────────────────────────────────────────

def databricks_sql_query(sql: str) -> dict:
    """Execute a read-only SQL query against the coverpath_demo Unity Catalog schema."""
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are permitted.")

    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    def _call():
        resp = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            catalog=CATALOG,
            schema=SCHEMA,
            wait_timeout="50s",
        )
        state = resp.status.state.value if resp.status and resp.status.state else "UNKNOWN"
        if state != "SUCCEEDED":
            err = resp.status.error.message if resp.status and resp.status.error else "unknown error"
            raise Exception(f"SQL execution failed ({state}): {err}")

        columns = []
        rows = []
        if resp.manifest and resp.manifest.schema and resp.manifest.schema.columns:
            columns = [c.name for c in resp.manifest.schema.columns]
        if resp.result and resp.result.data_array:
            rows = [dict(zip(columns, row)) for row in resp.result.data_array]

        return {
            "sql": sql.strip(),
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }

    return _retry(_call)


# ── Pre-built SQL queries ──────────────────────────────────────────────────────

QUERIES = {
    "initiation_setting_split": """
        SELECT initiation_setting, COUNT(*) AS patient_count
        FROM trial_enrollment
        GROUP BY initiation_setting
        ORDER BY patient_count DESC
    """,

    "outpatient_response_week26": """
        SELECT
            e.initiation_setting,
            COUNT(*) AS total_patients,
            SUM(CASE WHEN o.complete_tma_response = TRUE THEN 1 ELSE 0 END) AS responders,
            ROUND(100.0 * SUM(CASE WHEN o.complete_tma_response = TRUE THEN 1 ELSE 0 END) / COUNT(*), 1) AS response_rate_pct
        FROM trial_enrollment e
        JOIN trial_outcomes o ON e.patient_id = o.patient_id
        WHERE o.visit_week = 26
        GROUP BY e.initiation_setting
        ORDER BY e.initiation_setting
    """,

    "meningococcal_infection_count": """
        SELECT COUNT(*) AS meningococcal_infection_rows
        FROM trial_adverse_events
        WHERE LOWER(ae_term) = 'meningococcal infection'
    """,

    "week52_sustained_response": """
        SELECT
            COUNT(*) AS responders_at_week26,
            SUM(CASE WHEN o52.complete_tma_response = TRUE THEN 1 ELSE 0 END) AS sustained_at_week52,
            ROUND(100.0 * SUM(CASE WHEN o52.complete_tma_response = TRUE THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_sustained
        FROM trial_outcomes o26
        JOIN trial_outcomes o52 ON o26.patient_id = o52.patient_id AND o52.visit_week = 52
        WHERE o26.visit_week = 26 AND o26.complete_tma_response = TRUE
    """,

    "registry_outpatient_safety": """
        SELECT
            rp.initiation_setting,
            COUNT(DISTINCT rp.registry_id) AS total_patients,
            SUM(CASE WHEN ro.hospitalization_required = TRUE THEN 1 ELSE 0 END) AS hospitalizations,
            SUM(CASE WHEN ro.any_serious_ae = TRUE THEN 1 ELSE 0 END) AS serious_aes,
            SUM(CASE WHEN ro.months_from_start = 12 AND ro.sustained_tma_response = TRUE THEN 1 ELSE 0 END) AS sustained_12mo
        FROM registry_patients rp
        JOIN registry_outcomes ro ON rp.registry_id = ro.registry_id
        WHERE rp.initiation_setting = 'Outpatient'
        GROUP BY rp.initiation_setting
    """,

    "registry_cgs_jurisdiction": """
        SELECT treating_state, COUNT(*) AS patients
        FROM registry_patients
        WHERE treating_state IN ('MO', 'KS', 'NE', 'IA')
        GROUP BY treating_state
        ORDER BY patients DESC
    """,

    "prior_pe_courses_distribution": """
        SELECT prior_pe_courses, COUNT(*) AS patients
        FROM trial_enrollment
        GROUP BY prior_pe_courses
        ORDER BY prior_pe_courses
    """,
}
