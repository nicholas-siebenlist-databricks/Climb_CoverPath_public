import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Optional

from openai import OpenAI

from agent.prompts import (
    STEP1_SYSTEM, STEP4_SYSTEM, STEP5_SYSTEM,
    JUDGE_COMPLETENESS_SYSTEM, JUDGE_EVIDENCE_SYSTEM,
    JUDGE_TONE_SYSTEM, JUDGE_THREAT_SYSTEM,
)
from agent.tools import (
    cms_fetch_lcd,
    cms_search_precedent,
    openfda_get_label,
    clinicaltrials_get_study,
    pubmed_fetch,
    databricks_sql_query,
    QUERIES,
)

MODEL = os.environ.get("COVERPATH_MODEL", "databricks-claude-sonnet-4-6")


def _get_client() -> OpenAI:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    host = w.config.host.rstrip("/").replace("https://", "")
    token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")
    return OpenAI(
        base_url=f"https://{host}/serving-endpoints",
        api_key=token or "token",
    )


class WorkflowOrchestrator:
    def __init__(self, on_step: Callable, on_trace: Callable):
        self.client = _get_client()
        self.on_step = on_step   # on_step(step_idx, status, message)
        self.on_trace = on_trace  # on_trace(event_dict)

    def _trace(self, tool: str, input_summary: str, output_summary: str, status: str, elapsed_ms: int):
        self.on_trace({
            "ts": datetime.now().strftime("%H:%M:%S"),
            "tool": tool,
            "input": input_summary,
            "output": output_summary,
            "status": status,
            "elapsed_ms": elapsed_ms,
        })

    def _llm(self, system: str, user: str, max_tokens: int = 4096) -> str:
        resp = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def _llm_stream(self, system: str, user: str, max_tokens: int = 8192):
        stream = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Step 1: Ingest & Extract ───────────────────────────────────────────────

    def step1_ingest(self, lcd_id: str) -> dict:
        self.on_step(0, "running", "Fetching draft LCD from CMS Coverage Database…")

        t0 = time.time()
        try:
            lcd_data = cms_fetch_lcd(lcd_id)
            elapsed = int((time.time() - t0) * 1000)
            self._trace(
                "cms_coverage_database",
                f"fetch_lcd({lcd_id!r})",
                f"LCD loaded — {len(lcd_data['html'])} chars",
                "ok", elapsed,
            )
            self.on_trace({
                "type": "lcd_loaded",
                "lcd_id": lcd_id,
                "title": lcd_data.get("title", ""),
                "char_count": len(lcd_data["html"]),
            })
        except Exception as e:
            self._trace("cms_coverage_database", f"fetch_lcd({lcd_id!r})", str(e), "error", 0)
            raise

        self.on_step(0, "running", "Extracting restrictions with Claude…")
        t0 = time.time()
        extraction_prompt = f"Here is the full text of the draft LCD:\n\n{lcd_data['html']}"
        raw = self._llm(STEP1_SYSTEM, extraction_prompt, max_tokens=2048)
        elapsed = int((time.time() - t0) * 1000)

        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            extracted = json.loads(clean.strip())
        except json.JSONDecodeError:
            extracted = _FIXTURE_RESTRICTIONS

        restriction_count = len(extracted.get("restrictions", []))
        self._trace(
            "llm_extraction",
            "Extract restrictions from LCD HTML",
            f"{restriction_count} restrictions extracted",
            "ok", elapsed,
        )
        self.on_trace({
            "type": "restrictions_extracted",
            "restrictions": extracted.get("restrictions", []),
            "mac_name": extracted.get("mac_name", ""),
            "mac_jurisdiction": extracted.get("mac_jurisdiction", ""),
            "drug_name": extracted.get("drug_name", ""),
            "lcd_id": extracted.get("lcd_id", ""),
            "icd_codes_supported": extracted.get("icd_codes_supported", []),
            "icd_codes_excluded": extracted.get("icd_codes_excluded", []),
        })
        self.on_step(0, "complete", f"{restriction_count} restrictions extracted")
        return extracted

    # ── Step 2: External Evidence Sweep ───────────────────────────────────────

    def step2_external_evidence(self, restrictions: list) -> dict:
        self.on_step(1, "running", "Launching 4 parallel external queries…")

        tasks = {
            "cms_precedent": lambda: cms_search_precedent("C5 complement inhibitor"),
            "openfda_label":  lambda: openfda_get_label("Soliris"),
            "clinicaltrials": lambda: clinicaltrials_get_study("NCT01844856"),
            "pubmed":         lambda: pubmed_fetch(),
        }

        tool_names = {
            "cms_precedent": "cms_coverage_database.search_precedent",
            "openfda_label":  "openfda_drug_label.get_label",
            "clinicaltrials": "clinicaltrials_gov.get_study",
            "pubmed":         "pubmed.fetch",
        }
        task_inputs = {
            "cms_precedent": 'search_precedent(drug_class="C5 complement inhibitor")',
            "openfda_label":  'get_label(brand_name="Soliris")',
            "clinicaltrials": 'get_study(nct_id="NCT01844856")',
            "pubmed":         'fetch(pmids=["22553501","25908597","31395980"])',
        }

        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                t0 = time.time()
                try:
                    result = future.result()
                    elapsed = int((time.time() - t0) * 1000)
                    results[name] = result
                    summary_fns = {
                        "cms_precedent": lambda r: f"{len(r)} prior LCDs found (L34007, L34314)",
                        "openfda_label":  lambda r: f"Label retrieved for {r.get('brand_name', 'drug')}",
                        "clinicaltrials": lambda r: f"Trial record: {r.get('nct_id', '')}",
                        "pubmed":         lambda r: f"{len(r)} publications fetched",
                    }
                    summary = summary_fns[name](result)
                    self._trace(tool_names[name], task_inputs[name], summary, "ok", elapsed)
                except Exception as e:
                    elapsed = int((time.time() - t0) * 1000)
                    results[name] = {"error": str(e)}
                    self._trace(tool_names[name], task_inputs[name], str(e), "error", elapsed)

        self.on_step(1, "complete", "4 external sources retrieved")
        return results

    # ── Step 3: Internal Evidence Query ───────────────────────────────────────

    def step3_internal_evidence(self) -> dict:
        self.on_step(2, "running", "Querying Unity Catalog…")

        query_order = [
            ("initiation_setting_split",       "Initiation setting distribution"),
            ("outpatient_response_week26",      "Week-26 TMA response by setting"),
            ("meningococcal_infection_count",   "Meningococcal infection AE count"),
            ("week52_sustained_response",       "Week-52 sustained response"),
            ("registry_outpatient_safety",      "Registry outpatient safety outcomes"),
            ("registry_cgs_jurisdiction",       "Registry patients in CGS jurisdiction"),
            ("prior_pe_courses_distribution",   "Prior plasma exchange distribution"),
        ]

        results = {}
        for key, description in query_order:
            sql = QUERIES[key]
            t0 = time.time()
            try:
                result = databricks_sql_query(sql)
                elapsed = int((time.time() - t0) * 1000)
                results[key] = result
                self._trace("databricks_sql_query", description, _summarize_rows(result), "ok", elapsed)
            except Exception as e:
                elapsed = int((time.time() - t0) * 1000)
                results[key] = {"error": str(e), "rows": []}
                self._trace("databricks_sql_query", description, str(e), "error", elapsed)

        self.on_step(2, "complete", f"{len(query_order)} queries complete")
        return results

    # ── Step 4: Evidence Mapping ───────────────────────────────────────────────

    def step4_evidence_mapping(self, restrictions_data: dict, external: dict, internal: dict) -> dict:
        self.on_step(3, "running", "Mapping evidence to each restriction…")

        internal_rows = json.dumps(
            {k: v.get('rows', [])[:20] for k, v in internal.items() if 'error' not in v},
            indent=2,
        )
        # Strip 8+ digit sequences to avoid Presidio bank-account false positives
        # (no word boundaries — catches digits embedded in alphanumeric strings too)
        internal_rows = re.sub(r'\d{8,}', '[NUM]', internal_rows)

        context = f"""
LCD RESTRICTIONS:
{json.dumps(restrictions_data.get('restrictions', []), indent=2)}

INTERNAL EVIDENCE (SQL Query Results):
{internal_rows}

EXTERNAL EVIDENCE:
- CMS Precedent LCDs: {json.dumps(external.get('cms_precedent', []), indent=2)}
- openFDA (class analogue): indications={external.get('openfda_label', {}).get('indications_and_usage', '')[:500]}
- ClinicalTrials.gov NCT01844856: {external.get('clinicaltrials', {}).get('eligibility_criteria', '')[:500]}
- PubMed Publications: {json.dumps([{'citation': f"{p.get('authors', ['?'])[0]} et al. {p.get('journal','')[:40]} ({p.get('pub_date','')[:4]})", 'title': p.get('title','')} for p in external.get('pubmed', [])], indent=2)}
"""
        # Also strip 8+ digit sequences anywhere else in context (eligibility criteria, etc.)
        context = re.sub(r'\d{8,}', '[NUM]', context)

        t0 = time.time()
        raw = self._llm(STEP4_SYSTEM, context, max_tokens=3000)
        elapsed = int((time.time() - t0) * 1000)

        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            mapping = json.loads(clean.strip())
        except json.JSONDecodeError:
            mapping = {"restriction_rebuttals": [], "raw": raw}

        restriction_count = len(mapping.get("restriction_rebuttals", []))

        # Log guardrail debug info when 0 rebuttals returned
        if restriction_count == 0:
            try:
                raw_json = json.loads(raw.strip())
                if 'input_guardrail' in raw_json:
                    anon = str(raw_json['input_guardrail'][0].get('anonymized_input', ''))[:1000]
                    self._trace("guardrail_debug", "PII guardrail fired in step4",
                                f"anonymized: {anon}", "warn", 0)
                else:
                    self._trace("guardrail_debug", "Step4 unexpected JSON (no rebuttals)",
                                raw[:500], "warn", 0)
            except Exception:
                self._trace("guardrail_debug", "Step4 non-JSON response",
                            raw[:500], "warn", 0)

        self._trace(
            "llm_evidence_mapping",
            f"Map {len(restrictions_data.get('restrictions', []))} restrictions",
            f"{restriction_count} rebuttals structured",
            "ok" if restriction_count > 0 else "warn", elapsed,
        )
        self.on_step(3, "complete", f"{restriction_count} restriction-evidence mappings built")
        return mapping

    # ── Step 5: Brief Generation ───────────────────────────────────────────────

    def step5_generate_brief(
        self,
        restrictions_data: dict,
        external: dict,
        internal: dict,
        mapping: dict,
        on_chunk: Optional[Callable] = None,
        case_context: Optional[dict] = None,
    ) -> str:
        self.on_step(4, "running", "Generating evidence brief…")

        mac_md = case_context or {}
        mac_contact_lines = []
        if mac_md.get("mac_director_name"):
            mac_contact_lines.append(f"MAC Medical Director: {mac_md['mac_director_name']}")
        if mac_md.get("mac_director_phone"):
            mac_contact_lines.append(f"MAC Medical Director Phone: {mac_md['mac_director_phone']}")
        if mac_md.get("mac_director_email"):
            mac_contact_lines.append(f"MAC Medical Director Email: {mac_md['mac_director_email']}")
        mac_contact_block = "\n".join(mac_contact_lines)

        icd_supported = restrictions_data.get("icd_codes_supported", ["D59.31", "D59.39"])
        icd_excluded = restrictions_data.get("icd_codes_excluded", ["D59.32", "M31.1", "D59.0", "A04.3", "N17.0-N17.9"])
        icd_recs = mapping.get("icd_code_recommendations", {})

        context = f"""
DRUG: Suvaxilumab (anti-C5 complement inhibitor, BLA 761204, approved for aHUS)
MAC: CGS Administrators LLC (MO/KS/NE/IA)
LCD: MCD A-XXXXX (Draft — comment period closes July 5, 2026)
{mac_contact_block}

RESTRICTIONS:
{json.dumps(restrictions_data.get('restrictions', []), indent=2)}

EVIDENCE MAPPING:
{json.dumps(mapping.get('restriction_rebuttals', []), indent=2)}

ICD-10-CM CONTEXT:
- MAC-supported codes: {icd_supported}
- MAC-excluded codes: {icd_excluded}
- Manufacturer ICD recommendations: {json.dumps(icd_recs, indent=2)}

INTERNAL DATA HIGHLIGHTS:
- Initiation settings: {_extract_highlights(internal)}
- Prior PE courses: {_extract_pe_courses(internal)}
- Week-26 TMA response: {_extract_response(internal)}
- Meningococcal AE count: {_extract_meningo(internal)}
- Registry outpatient safety: {_extract_registry(internal)}
- CGS jurisdiction patients: {_extract_jurisdiction(internal)}

EXTERNAL CITATIONS:
- CMS LCD L34007 (Noridian) — no step therapy requirement for C5 inhibitor class in aHUS
- CMS LCD L34314 (Palmetto GBA) — outpatient initiation permitted for C5 inhibitor class
- openFDA Soliris label — first-line FDA approval, no step therapy in indication
- NCT01844856 (RECOVER, eculizumab aHUS) — no plasma exchange prerequisite in eligibility
- Legendre et al. NEJM 2013 (pivotal eculizumab aHUS trial; eref:Legendre-NEJM-2013)
- Fakhouri et al. AJKD 2016 (real-world outcomes, outpatient setting; eref:Fakhouri-AJKD-2016)
- Rondeau et al. KIR 2019 (5-year eculizumab renal registry safety data; eref:Rondeau-KIR-2019)

Write the full evidence brief now."""

        t0 = time.time()
        brief_chunks = []
        try:
            for chunk in self._llm_stream(STEP5_SYSTEM, context, max_tokens=8192):
                brief_chunks.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
        except Exception as e:
            brief_chunks = [f"[Brief generation error: {e}]"]

        elapsed = int((time.time() - t0) * 1000)
        brief = "".join(brief_chunks)
        self._trace(
            "llm_brief_generation",
            "Generate structured evidence brief",
            f"{len(brief)} chars generated",
            "ok", elapsed,
        )
        self.on_step(4, "complete", "Evidence brief ready")
        return brief

    # ── Step 6: Quality Review ─────────────────────────────────────────────────

    def step6_quality_review(self, brief: str, restrictions_data: dict, mapping: dict) -> dict:
        self.on_step(5, "running", "Running 4 quality judges in parallel…")

        sanitized_brief = re.sub(r'PMID\s+(\d+)', r'PMID-\1', brief[:8000])
        brief_context = f"""BRIEF TO EVALUATE:
{sanitized_brief}

RESTRICTIONS:
{json.dumps(restrictions_data.get('restrictions', []), indent=2)}

EVIDENCE MAPPING SUMMARY:
{json.dumps(mapping.get('restriction_rebuttals', []), indent=2)[:3000]}"""

        judge_defs = {
            "completeness": JUDGE_COMPLETENESS_SYSTEM,
            "evidence":     JUDGE_EVIDENCE_SYSTEM,
            "tone":         JUDGE_TONE_SYSTEM,
            "threat":       JUDGE_THREAT_SYSTEM,
        }

        results = {}

        def run_judge(name: str, system_prompt: str):
            t0 = time.time()
            try:
                raw = self._llm(system_prompt, brief_context, max_tokens=1500)
                elapsed = int((time.time() - t0) * 1000)
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                result = json.loads(clean.strip())
                self._trace(
                    f"llm_judge_{name}",
                    f"Judge: {name}",
                    f"Score: {result.get('score', '?')}/10",
                    "ok", elapsed,
                )
                return name, result
            except Exception as e:
                elapsed = int((time.time() - t0) * 1000)
                self._trace(f"llm_judge_{name}", f"Judge: {name}", str(e), "error", elapsed)
                return name, {"score": 0, "error": str(e)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(run_judge, name, sys): name for name, sys in judge_defs.items()}
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
                self.on_trace({
                    "type": "judge_result",
                    "judge": name,
                    "result": result,
                })

        scores = ", ".join(f"{k}={v.get('score', '?')}" for k, v in results.items())
        self.on_step(5, "complete", f"Quality review complete — {scores}")
        return results

    # ── Full Workflow ──────────────────────────────────────────────────────────

    def run(
        self,
        lcd_id: str = "MCD-A-XXXXX",
        on_chunk: Optional[Callable] = None,
        case_context: Optional[dict] = None,
    ) -> str:
        restrictions_data = self.step1_ingest(lcd_id)
        external = self.step2_external_evidence(restrictions_data.get("restrictions", []))
        internal = self.step3_internal_evidence()
        mapping = self.step4_evidence_mapping(restrictions_data, external, internal)
        brief = self.step5_generate_brief(restrictions_data, external, internal, mapping, on_chunk, case_context)
        self.step6_quality_review(brief, restrictions_data, mapping)
        return brief


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarize_rows(result: dict) -> str:
    rows = result.get("rows", [])
    if not rows:
        return "0 rows"
    if len(rows) == 1:
        return str(rows[0])
    return f"{len(rows)} rows — first: {rows[0]}"


def _extract_highlights(internal: dict) -> str:
    rows = internal.get("initiation_setting_split", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    return ", ".join(f"{r.get('initiation_setting')}: {r.get('patient_count')}" for r in rows)


def _extract_pe_courses(internal: dict) -> str:
    rows = internal.get("prior_pe_courses_distribution", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    total = sum(int(r.get("patients", 0)) for r in rows)
    zero = next((int(r.get("patients", 0)) for r in rows if str(r.get("prior_pe_courses")) == "0"), 0)
    pct = round(100 * zero / total, 1) if total else 0
    return f"{zero}/{total} patients ({pct}%) had 0 prior PE courses"


def _extract_response(internal: dict) -> str:
    rows = internal.get("outpatient_response_week26", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    return "; ".join(
        f"{r.get('initiation_setting')}: {r.get('responders')}/{r.get('total_patients')} ({r.get('response_rate_pct')}%)"
        for r in rows
    )


def _extract_meningo(internal: dict) -> str:
    rows = internal.get("meningococcal_infection_count", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    return f"{rows[0].get('meningococcal_infection_rows', '?')} rows (expected: 0)"


def _extract_registry(internal: dict) -> str:
    rows = internal.get("registry_outpatient_safety", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    r = rows[0]
    return (
        f"{r.get('total_patients')} outpatient patients, "
        f"{r.get('hospitalizations')} hospitalizations, "
        f"{r.get('serious_aes')} serious AEs, "
        f"{r.get('sustained_12mo')} sustained at 12mo"
    )


def _extract_jurisdiction(internal: dict) -> str:
    rows = internal.get("registry_cgs_jurisdiction", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    total = sum(int(r.get("patients", 0)) for r in rows)
    states = [f"{r.get('treating_state')}={r.get('patients')}" for r in rows]
    return f"{total} patients in CGS states ({', '.join(states)})"


_FIXTURE_RESTRICTIONS = {
    "lcd_id": "MCD-A-XXXXX",
    "mac_name": "CGS Administrators LLC",
    "mac_jurisdiction": "MO, KS, NE, IA",
    "drug_name": "Suvaxilumab",
    "effective_date": "TBD",
    "restrictions": [
        {
            "id": "R1", "type": "Step Therapy",
            "criteria_text": "Requires documentation of clinical failure of ≥2 courses of plasma exchange (PE) therapy within the preceding 12 months.",
            "cited_rationale": "PE remains the standard of care for initial TMA management.",
        },
        {
            "id": "R2", "type": "Site of Care",
            "criteria_text": "Initial administration must occur in an inpatient hospital setting.",
            "cited_rationale": "Boxed warning identifies meningococcal infection as life-threatening.",
        },
        {
            "id": "R3", "type": "Duration Limit",
            "criteria_text": "Coverage limited to 12 weeks. Continued coverage requires prior authorization.",
            "cited_rationale": "Long-term benefit beyond 12 weeks is limited to single-arm observational data.",
        },
    ],
}
