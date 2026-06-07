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

_CLINTRIALS_MCP_PATH = "/api/2.0/mcp/external/climb_clintrials_v2"


def _mcp_clintrials(tool_name: str, arguments: dict) -> dict:
    """Call the climb_clintrials_v2 MCP server on the current workspace."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    host = w.config.host.rstrip("/")
    token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")
    resp = requests.post(
        f"{host}{_CLINTRIALS_MCP_PATH}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": tool_name, "arguments": arguments}},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")
    return data["result"]["structuredContent"]


def _clintrials_via_mcp(nct_id: str) -> dict:
    """Fetch trial metadata + outcomes via the internal MCP server."""
    trial = _mcp_clintrials("clinicaltrials_get_trial", {"nct_id": nct_id})
    outcomes = _mcp_clintrials("clinicaltrials_get_trial_outcomes", {"nct_id": nct_id})
    return {
        "nct_id": trial["nct_id"],
        "title": trial["title"],
        "status": trial.get("status", ""),
        "phase": trial.get("phase", ""),
        "conditions": trial.get("conditions", []),
        "interventions": [i["name"] for i in trial.get("interventions", [])],
        "enrollment": trial.get("enrollment"),
        "start_date": trial.get("start_date"),
        "completion_date": trial.get("completion_date"),
        "sponsors": trial.get("sponsors", []),
        # MCP doesn't expose eligibility_criteria; brief_summary is the closest proxy
        "eligibility_criteria": trial.get("brief_summary", "")[:1500],
        "primary_outcomes": [o["measure"] for o in outcomes.get("primary", [])[:3]],
        "source": "climb_clintrials_v2 MCP",
    }


def _clintrials_via_rest(nct_id: str) -> dict:
    """Fetch trial metadata from the public ClinicalTrials.gov v2 REST API (fallback)."""
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
        "source": "ClinicalTrials.gov v2 REST API (fallback)",
    }


def clinicaltrials_get_study(nct_id: str) -> dict:
    """Fetch a specific study by NCT ID — MCP-first, REST fallback."""
    def _call():
        try:
            return _clintrials_via_mcp(nct_id)
        except Exception:
            return _clintrials_via_rest(nct_id)
    return _retry(_call)


def clinicaltrials_find_for_drug(drug_name: str) -> dict:
    """Search for the most relevant trial for a drug name, then fetch full details.

    Uses MCP clinicaltrials_search_trials → top result → clinicaltrials_get_study.
    Falls back to the public ClinicalTrials.gov search API if MCP is unavailable.
    """
    def _call():
        try:
            structured = _mcp_clintrials("clinicaltrials_search_trials", {
                "query": drug_name,
                "max_results": 3,
            })
            # MCP wraps list results: {"result": [...]}
            trials = structured.get("result", structured) if isinstance(structured, dict) else structured
            if not trials:
                raise ValueError(f"No trials found for '{drug_name}'")
            nct_id = trials[0].get("nct_id", "")
            if not nct_id:
                raise ValueError("Search returned trial with no NCT ID")
            return _clintrials_via_mcp(nct_id)
        except Exception:
            # Fallback: public search API
            url = (
                f"https://clinicaltrials.gov/api/v2/studies"
                f"?query.term={requests.utils.quote(drug_name)}&pageSize=1"
                f"&fields=NCTId"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            studies = resp.json().get("studies", [])
            if not studies:
                return {"nct_id": "", "title": "", "source": "no trials found", "eligibility_criteria": ""}
            nct_id = studies[0].get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
            return _clintrials_via_rest(nct_id) if nct_id else {}
    return _retry(_call)


# ── PubMed ────────────────────────────────────────────────────────────────────

_PUBMED_MCP_PATH = "/api/2.0/mcp/external/climb_pubmed"


def _mcp_pubmed(tool_name: str, arguments: dict) -> dict:
    """Call the climb_pubmed MCP server on the current workspace."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    host = w.config.host.rstrip("/")
    token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")
    resp = requests.post(
        f"{host}{_PUBMED_MCP_PATH}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": tool_name, "arguments": arguments}},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")
    result = data["result"]
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    return result["structuredContent"]


def _normalize_pubmed_article(article: dict, source: str) -> dict:
    """Normalize a PubMed article from MCP or REST into a consistent shape."""
    return {
        "pmid": article.get("pmid", ""),
        "title": article.get("title", ""),
        "abstract": article.get("abstract", ""),
        "authors": article.get("authors", [])[:3],
        "journal": article.get("journal", article.get("fulljournalname", article.get("source", ""))),
        "pub_date": article.get("pub_date", article.get("pubdate", "")),
        "source": source,
    }


def pubmed_fetch(pmids: list) -> list:
    """Fetch article metadata for a list of PMIDs — MCP-first, NCBI REST fallback."""

    def _fetch_via_mcp(pmid):
        article = _mcp_pubmed("pubmed_get_article", {"pmid": pmid})
        return _normalize_pubmed_article(article, f"climb_pubmed MCP PMID {pmid}")

    def _fetch_via_rest(pmid):
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={pmid}&retmode=json"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        article = data.get("result", {}).get(str(pmid), {})
        return _normalize_pubmed_article(article, f"PubMed NCBI REST PMID {pmid}")

    results = []
    for pmid in pmids:
        def _call(p=pmid):
            try:
                return _fetch_via_mcp(p)
            except Exception:
                time.sleep(0.35)  # NCBI courtesy rate limit
                return _fetch_via_rest(p)
        try:
            results.append(_retry(_call))
        except Exception as e:
            results.append({"pmid": pmid, "error": str(e), "source": f"PubMed PMID {pmid}"})
    return results


def pubmed_search(query: str, max_results: int = 3) -> list:
    """Search PubMed by query string — MCP-first, NCBI REST fallback."""

    def _search_via_mcp():
        data = _mcp_pubmed("pubmed_search_articles", {
            "query": query,
            "max_results": max_results,
        })
        articles = data.get("result", data) if isinstance(data, dict) else data
        return [_normalize_pubmed_article(a, "climb_pubmed MCP") for a in articles]

    def _search_via_rest():
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={requests.utils.quote(query)}"
            f"&retmax={max_results}&retmode=json&sort=relevance"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        return pubmed_fetch(pmids)

    def _call():
        try:
            return _search_via_mcp()
        except Exception:
            return _search_via_rest()

    return _retry(_call)


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
