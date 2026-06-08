import json
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Generator, Optional

from openai import OpenAI

from agent.prompts import (
    STEP1_SYSTEM, STEP2A_CLASS_INFERENCE_SYSTEM,
    STEP4_SYSTEM, STEP5_SYSTEM,
    JUDGE_COMPLETENESS_SYSTEM, JUDGE_EVIDENCE_SYSTEM,
    JUDGE_TONE_SYSTEM, JUDGE_THREAT_SYSTEM,
)
from agent.tools import (
    cms_fetch_lcd,
    cms_search_precedent,
    openfda_get_label,
    clinicaltrials_find_for_drug,
    pubmed_search,
    databricks_sql_query,
    chembl_drug_profile,
    chembl_verify_analogue,
    QUERIES,
)

MODEL = os.environ.get("COVERPATH_MODEL", "databricks-claude-sonnet-4-6")

# JSON generation steps (extraction, mapping): large limit to avoid truncation.
_OUTPUT_LIMIT = 50000
# Prose brief (step 5): 4 restrictions × 3 paragraphs + 6 additional sections ≈ 14k tokens.
_BRIEF_TOKEN_LIMIT = 16000
# Quality judges output small JSON (~300-1500 tokens); keep well under serving endpoint limits.
_JUDGE_TOKEN_LIMIT = 4000
# Drug class inference micro-call: small JSON output.
_ANALOGUE_TOKEN_LIMIT = 600
# Hard wall-clock cap on brief streaming — per-chunk timeout on the HTTP client
# does NOT bound total stream duration; this does.
_BRIEF_MAX_SECONDS = 300
# Emit a Tool Trace update every this many seconds during brief streaming.
_BRIEF_TRACE_INTERVAL = 30
# debug_heartbeat events for the Generation Log (finer grain than Tool Traces).
_BRIEF_HEARTBEAT_SECONDS = 15

# Regex to strip ```json ... ``` or ``` ... ``` code fences from LLM output
_FENCE_RE = re.compile(r'^```(?:json)?\s*\n?(.*?)(?:\n?```\s*)?$', re.DOTALL)


def _stream_with_timeout(gen: Generator, timeout_secs: float) -> Generator:
    """Wrap a generator so the total wall-clock time (including first-chunk wait) is bounded.

    The underlying generator runs in a daemon thread.  The main thread yields
    items from a queue and returns (exhausts the generator) once timeout_secs
    have elapsed — even if no chunks have arrived yet.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def _run():
        try:
            for item in gen:
                q.put(('item', item))
        except Exception as e:
            q.put(('error', e))
        finally:
            q.put(('done', _DONE))

    threading.Thread(target=_run, daemon=True).start()
    deadline = time.time() + timeout_secs

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        try:
            kind, val = q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue  # check deadline on next iteration
        if kind == 'done':
            return
        if kind == 'error':
            raise val
        yield val


def _parse_llm_json(raw: str, fallback: dict) -> tuple[dict, bool]:
    """Parse JSON from an LLM response, handling code-fence wrapping.

    Returns (result, was_truncated).
    was_truncated=True means the output ended before its closing delimiter —
    the caller should emit a 'warn' trace so it's visible in the event stream.
    """
    clean = raw.strip()
    m = _FENCE_RE.match(clean)
    if m:
        clean = m.group(1).strip()
    try:
        return json.loads(clean), False
    except json.JSONDecodeError:
        truncated = bool(clean) and not clean.rstrip().endswith(('}', ']'))
        return fallback, truncated


_openai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    host = w.config.host.rstrip("/").replace("https://", "")
    token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")
    _openai_client = OpenAI(
        base_url=f"https://{host}/serving-endpoints",
        api_key=token or "token",
        timeout=180.0,
    )
    return _openai_client


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

    def _llm(self, system: str, user: str, max_tokens: int = _OUTPUT_LIMIT) -> str:
        resp = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def _llm_stream(self, system: str, user: str, max_tokens: int = _OUTPUT_LIMIT):
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
        raw = self._llm(STEP1_SYSTEM, extraction_prompt)
        elapsed = int((time.time() - t0) * 1000)

        extracted, truncated = _parse_llm_json(raw, _FIXTURE_RESTRICTIONS)
        if truncated:
            self._trace("llm_extraction", "Extract restrictions from LCD HTML",
                        f"Output truncated at {len(raw)} chars — fell back to fixture", "warn", elapsed)

        restriction_count = len(extracted.get("restrictions", []))
        self._trace(
            "llm_extraction",
            "Extract restrictions from LCD HTML",
            f"{restriction_count} restrictions extracted",
            "ok" if not truncated else "warn", elapsed,
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

    # ── Step 2a: Drug Class Profiling (ChEMBL + LLM inference) ───────────────

    def step2a_drug_class_profile(self, restrictions_data: dict) -> dict:
        self.on_step(1, "running", "Profiling drug class with ChEMBL…")

        drug_name = restrictions_data.get("drug_name", "")
        search_name = re.sub(r'\s*\(.*', '', drug_name).strip() or drug_name

        # 1. ChEMBL molecule lookup — confirm drug exists, get type/phase/synonyms
        t0 = time.time()
        try:
            profile = chembl_drug_profile(search_name)
            elapsed = int((time.time() - t0) * 1000)
            if profile.get("chembl_id"):
                self._trace(
                    "chembl.search_molecules",
                    f"search_molecules({search_name!r})",
                    f"ChEMBL ID: {profile['chembl_id']} · type: {profile.get('molecule_type') or 'n/a'} · phase: {profile.get('max_phase') or 'n/a'} · indication_class: {profile.get('indication_class') or 'null (biologic)'}",
                    "ok", elapsed,
                )
            else:
                self._trace("chembl.search_molecules", f"search_molecules({search_name!r})",
                            "No molecule found in ChEMBL", "warn", elapsed)
                self.on_step(1, "complete", f"No ChEMBL record for '{search_name}' — proceeding without analogue data")
                return {"profile": {}, "drug_class": None, "analogues": [], "primary_analogue": None}
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            self._trace("chembl.search_molecules", f"search_molecules({search_name!r})", str(e), "error", elapsed)
            self.on_step(1, "complete", "ChEMBL unavailable — proceeding without analogue data")
            return {"profile": {}, "drug_class": None, "analogues": [], "primary_analogue": None}

        # 2. LLM micro-call: infer drug class and approved analogues from name + ChEMBL metadata.
        # ChEMBL's indication_class is null for biologics/antibodies (no biochemical assay data),
        # so class inference requires an LLM reasoning step for this drug type.
        t0 = time.time()
        llm_context = (
            f"Drug: {drug_name}\n"
            f"Molecule type: {profile.get('molecule_type') or 'unknown'}\n"
            f"Max phase (approval status): {profile.get('max_phase') or 'unknown'}\n"
            f"ChEMBL indication_class: {profile.get('indication_class') or 'null (not indexed for biologics)'}\n"
            f"Known synonyms: {', '.join(profile.get('synonyms', [])[:5])}"
        )
        try:
            raw = self._llm(STEP2A_CLASS_INFERENCE_SYSTEM, llm_context, max_tokens=_ANALOGUE_TOKEN_LIMIT)
            elapsed = int((time.time() - t0) * 1000)
            class_data, truncated = _parse_llm_json(raw, {})
            drug_class    = class_data.get("drug_class") or ""
            primary_raw   = class_data.get("primary_analogue") or ""
            analogues_raw = class_data.get("approved_analogues", [])
            self._trace(
                "llm.class_inference",
                f"Infer class + analogues for {search_name!r}",
                f"Class: {drug_class} · primary: {primary_raw} · {len(analogues_raw)} candidates",
                "warn" if truncated else "ok", elapsed,
            )
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            self._trace("llm.class_inference", f"Class inference for {search_name!r}", str(e), "error", elapsed)
            self.on_step(1, "complete", "Class inference failed — proceeding without analogue data")
            return {"profile": profile, "drug_class": None, "analogues": [], "primary_analogue": None}

        # 3. ChEMBL verification: confirm each LLM-suggested analogue is phase ≥ 3
        verified = []
        for name in analogues_raw:
            t0 = time.time()
            try:
                mol = chembl_verify_analogue(name)
                elapsed = int((time.time() - t0) * 1000)
                if mol:
                    verified.append(mol)
                    self._trace(
                        "chembl.verify_analogue",
                        f"verify({name!r})",
                        f"Confirmed: {mol['name']} | {mol['chembl_id']} | phase={mol['max_phase']}",
                        "ok", elapsed,
                    )
                else:
                    self._trace("chembl.verify_analogue", f"verify({name!r})",
                                "Not found or below phase 3", "warn", elapsed)
            except Exception as e:
                elapsed = int((time.time() - t0) * 1000)
                self._trace("chembl.verify_analogue", f"verify({name!r})", str(e), "error", elapsed)

        # Primary analogue: prefer ChEMBL-verified version of the LLM's primary pick
        primary_name_clean = re.sub(r'\s*\(.*', '', primary_raw).strip().upper()
        primary_analogue = next(
            (v for v in verified if v["name"].upper() == primary_name_clean),
            verified[0] if verified else None,
        )
        primary_name = primary_analogue["name"].title() if primary_analogue else None

        self.on_step(
            1, "complete",
            f"Class: {drug_class or 'unknown'} · primary analogue: {primary_name or 'none'} · {len(verified)} verified",
        )
        return {
            "profile":          profile,
            "drug_class":       drug_class,
            "analogues":        verified,
            "primary_analogue": primary_analogue,
        }

    # ── Step 2b: External Evidence Sweep ──────────────────────────────────────

    def step2b_external_evidence(self, restrictions_data: dict, drug_profile: dict) -> dict:
        self.on_step(2, "running", "Launching 4 parallel external queries…")

        drug_name = restrictions_data.get("drug_name", "")
        # LLM may extract the full generic suffix, e.g. "Ultomiris (ravulizumab-cwvz)".
        # APIs index by brand name only — strip anything in parentheses for searches.
        search_name = re.sub(r'\s*\(.*', '', drug_name).strip() or drug_name

        # Use ChEMBL-verified + LLM-identified primary analogue for cms/fda queries.
        # Eculizumab is selected by data (ChEMBL phase=4 confirmed) rather than LLM training-data.
        primary = drug_profile.get("primary_analogue")
        analogue_name = primary["name"].title() if primary else None
        indication_class = drug_profile.get("drug_class", "")

        # openFDA indexes labels by brand name (e.g. "Soliris"), not INN/generic name
        # ("Eculizumab"). Extract the brand name from ChEMBL synonyms — it's typically
        # a single mixed-case alphabetic word, unlike the all-caps INN or alphanumeric
        # compound codes (e.g. "BNJ441").
        synonyms = primary.get("synonyms", []) if primary else []
        brand_name = next(
            (s for s in synonyms
             if len(s.split()) == 1 and s[0].isupper() and not s.isupper() and s.isalpha()),
            None,
        )

        cms_query   = analogue_name or search_name
        fda_query   = brand_name or analogue_name or search_name
        pubmed_query = (
            f"{search_name} {indication_class} clinical trial outcomes"
            if indication_class
            else f"{search_name} atypical hemolytic uremic syndrome clinical trial"
        )

        tasks = {
            "cms_precedent": lambda: cms_search_precedent(cms_query),
            "openfda_label":  lambda: openfda_get_label(fda_query),
            "clinicaltrials": lambda: clinicaltrials_find_for_drug(search_name),
            "pubmed":         lambda: pubmed_search(pubmed_query),
        }

        tool_names = {
            "cms_precedent": "cms_coverage_database.search_precedent",
            "openfda_label":  "openfda_drug_label.get_label",
            "clinicaltrials": "clinicaltrials_gov.find_for_drug",
            "pubmed":         "pubmed.search",
        }
        task_inputs = {
            "cms_precedent": f'search_precedent(drug_name={cms_query!r})',
            "openfda_label":  f'get_label(brand_name={fda_query!r})',
            "clinicaltrials": f'find_for_drug(drug_name={search_name!r})',
            "pubmed":         f'search(query={pubmed_query!r})',
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

                    def _pubmed_summary(r):
                        ok = [p for p in r if p.get("title") and not p.get("error")]
                        titles = "; ".join(p["title"][:60] for p in ok)
                        status = f"{len(ok)}/{len(r)} fetched"
                        return f"{status} — {titles}" if titles else f"{status} (no titles returned)"

                    summary_fns = {
                        "cms_precedent": lambda r: f"{len(r)} precedent LCDs found",
                        "openfda_label":  lambda r: f"Label retrieved for {r.get('brand_name', drug_name)}",
                        "clinicaltrials": lambda r: f"Trial: {r.get('nct_id', 'n/a')} — {r.get('title', '')[:60]} via {r.get('source', '?')}",
                        "pubmed":         _pubmed_summary,
                    }
                    summary = summary_fns[name](result)
                    ok_count = len([p for p in result if p.get("title")]) if name == "pubmed" else None
                    status = "warn" if (name == "pubmed" and ok_count == 0) else "ok"
                    self._trace(tool_names[name], task_inputs[name], summary, status, elapsed)
                except Exception as e:
                    elapsed = int((time.time() - t0) * 1000)
                    results[name] = {"error": str(e)}
                    self._trace(tool_names[name], task_inputs[name], str(e), "error", elapsed)

        analogue_note = f" (via ChEMBL analogue: {analogue_name})" if analogue_name else ""
        self.on_step(2, "complete", f"4 external sources retrieved{analogue_note}")
        return results

    # ── Step 3: Internal Evidence Query ───────────────────────────────────────

    def step3_internal_evidence(self) -> dict:
        self.on_step(3, "running", "Querying Unity Catalog…")

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

        self.on_step(3, "complete", f"{len(query_order)} queries complete")
        return results

    # ── Step 4: Evidence Mapping ───────────────────────────────────────────────

    def step4_evidence_mapping(self, restrictions_data: dict, external: dict, internal: dict) -> dict:
        self.on_step(4, "running", "Mapping evidence to each restriction…")

        internal_rows = json.dumps(
            {k: v.get('rows', [])[:20] for k, v in internal.items() if 'error' not in v},
            indent=2,
        )
        # Strip 8+ digit sequences to avoid Presidio bank-account false positives
        # (no word boundaries — catches digits embedded in alphanumeric strings too)
        internal_rows = re.sub(r'\d{8,}', '[NUM]', internal_rows)

        ct = external.get('clinicaltrials', {})
        ct_nct = ct.get('nct_id', 'n/a')
        ct_eligibility = ct.get('eligibility_criteria', ct.get('brief_summary', ''))[:500]

        context = f"""
LCD RESTRICTIONS:
{json.dumps(restrictions_data.get('restrictions', []), indent=2)}

INTERNAL EVIDENCE (SQL Query Results):
{internal_rows}

EXTERNAL EVIDENCE:
- CMS Precedent LCDs: {json.dumps(external.get('cms_precedent', []), indent=2)}
- openFDA (class analogue): indications={external.get('openfda_label', {}).get('indications_and_usage', '')[:500]}
- ClinicalTrials.gov {ct_nct}: {ct_eligibility}
- PubMed Publications: {json.dumps([{'citation': f"{p.get('authors', ['?'])[0]} et al. {p.get('journal','')[:40]} ({p.get('pub_date','')[:4]})", 'title': p.get('title','')} for p in external.get('pubmed', [])], indent=2)}
"""
        # Also strip 8+ digit sequences anywhere else in context (eligibility criteria, etc.)
        context = re.sub(r'\d{8,}', '[NUM]', context)

        t0 = time.time()
        raw = self._llm(STEP4_SYSTEM, context)
        elapsed = int((time.time() - t0) * 1000)

        mapping, truncated = _parse_llm_json(raw, {"restriction_rebuttals": []})
        if truncated:
            self._trace("llm_evidence_mapping", "Map restrictions",
                        f"Output truncated at {len(raw)} chars", "warn", elapsed)

        restriction_count = len(mapping.get("restriction_rebuttals", []))

        # If 0 rebuttals and no truncation, check for PII guardrail interception
        if restriction_count == 0 and not truncated:
            grd, _ = _parse_llm_json(raw, {})
            if 'input_guardrail' in grd:
                anon = str(grd['input_guardrail'][0].get('anonymized_input', ''))[:500]
                self._trace("guardrail_debug", "PII guardrail fired in step4",
                            f"anonymized: {anon}", "warn", 0)
            else:
                self._trace("guardrail_debug", "Step4 returned no rebuttals",
                            raw[:300], "warn", 0)

        self._trace(
            "llm_evidence_mapping",
            f"Map {len(restrictions_data.get('restrictions', []))} restrictions",
            f"{restriction_count} rebuttals structured",
            "ok" if restriction_count > 0 else "warn", elapsed,
        )
        self.on_step(4, "complete", f"{restriction_count} restriction-evidence mappings built")
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
        self.on_step(5, "running", "Generating evidence brief…")

        mac_md = case_context or {}
        mac_contact_lines = []
        if mac_md.get("mac_director_name"):
            mac_contact_lines.append(f"MAC Medical Director: {mac_md['mac_director_name']}")
        if mac_md.get("mac_director_phone"):
            mac_contact_lines.append(f"MAC Medical Director Phone: {mac_md['mac_director_phone']}")
        if mac_md.get("mac_director_email"):
            mac_contact_lines.append(f"MAC Medical Director Email: {mac_md['mac_director_email']}")
        mac_contact_block = "\n".join(mac_contact_lines)

        drug_name = restrictions_data.get("drug_name", "Drug")
        mac_name = restrictions_data.get("mac_name", "MAC")
        mac_jurisdiction = restrictions_data.get("mac_jurisdiction", "")
        lcd_id_val = restrictions_data.get("lcd_id", "LCD")
        effective_date = restrictions_data.get("effective_date", "TBD")
        icd_supported = restrictions_data.get("icd_codes_supported", [])
        icd_excluded = restrictions_data.get("icd_codes_excluded", [])
        icd_recs = mapping.get("icd_code_recommendations", {})

        # Build dynamic external citations from live-fetched data
        citation_lines = []
        for lcd_ref in external.get('cms_precedent', []):
            lcd_ref_id = lcd_ref.get('lcd_id', '')
            lcd_mac = lcd_ref.get('mac', '')
            lcd_note = lcd_ref.get('note', '')
            if lcd_ref_id:
                citation_lines.append(f"- CMS LCD {lcd_ref_id} ({lcd_mac}) — {lcd_note}")
        fda_label = external.get('openfda_label', {})
        fda_brand = fda_label.get('brand_name', 'class analogue')
        if fda_brand:
            citation_lines.append(f"- openFDA {fda_brand} label — class analogue reference; first-line FDA approval, no step therapy in indication")
        ct = external.get('clinicaltrials', {})
        ct_nct = ct.get('nct_id', '')
        ct_title = ct.get('title', '')
        if ct_nct:
            citation_lines.append(f"- ClinicalTrials.gov {ct_nct} ({ct_title[:80]}) — eligibility and outcomes data")
        for pub in external.get('pubmed', []):
            if pub.get('title') and not pub.get('error'):
                try:
                    authors = pub.get('authors') or ['?']
                    first_author = str(authors[0]).split()[-1] if authors else '?'
                    journal = str(pub.get('journal') or '')[:30]
                    year = str(pub.get('pub_date') or '')[:4]
                    title = str(pub.get('title') or '')[:80]
                    citation_lines.append(f"- {first_author} et al. {journal} {year}: {title} (eref:{first_author}-{journal[:10].replace(' ','')}-{year})")
                except Exception:
                    pass
        external_citations_block = "\n".join(citation_lines) if citation_lines else "- No external citations retrieved"

        mac_display = f"{mac_name} ({mac_jurisdiction})" if mac_jurisdiction else mac_name
        lcd_display = f"{lcd_id_val} (Draft — effective {effective_date})"

        t0 = time.time()
        brief_chunks = []
        timed_out = False
        first_chunk_received = False
        _chars_ref = [0]  # mutable reference shared with heartbeat thread

        _hb_stop = threading.Event()
        _hb_count = [0]
        hb_thread = None  # set once thread is started; checked in finally

        _last_trace_at = [0.0]  # tracks when the last Tool Trace was emitted

        def _emit_heartbeat(n, chars, elapsed_s, first_seen):
            try:
                rate = round(chars / elapsed_s, 1) if elapsed_s > 0 else 0.0
                # Generation Log only — no Tool Trace here
                self.on_trace({
                    "type":        "debug_heartbeat",
                    "ts":          datetime.now().strftime("%H:%M:%S"),
                    "hb":          n,
                    "chars":       chars,
                    "elapsed_s":   round(elapsed_s, 1),
                    "rate_cps":    rate,
                    "first_chunk": first_seen,
                })
                # Emit a Tool Trace update every _BRIEF_TRACE_INTERVAL seconds
                if n > 0 and elapsed_s - _last_trace_at[0] >= _BRIEF_TRACE_INTERVAL:
                    _last_trace_at[0] = elapsed_s
                    status_note = " — waiting for first token" if not first_seen else ""
                    self._trace(
                        "llm_brief_generation",
                        "Brief in progress",
                        f"{chars:,} chars · {elapsed_s:.0f}s elapsed · {rate} chars/sec{status_note}",
                        "warn" if not first_seen and elapsed_s > 30 else "ok",
                        int(elapsed_s * 1000),
                    )
            except Exception as exc:
                try:
                    self._trace("llm_brief_generation", "Heartbeat error", str(exc), "error", 0)
                except Exception:
                    pass

        def _heartbeat():
            while not _hb_stop.wait(_BRIEF_HEARTBEAT_SECONDS):
                _hb_count[0] += 1
                _emit_heartbeat(
                    _hb_count[0], _chars_ref[0],
                    time.time() - t0, first_chunk_received,
                )

        try:
            # Context building is inside try so any extraction/serialization failure
            # produces a clean error brief rather than a workflow crash.
            context = f"""
DRUG: {drug_name}
MAC: {mac_display}
LCD: {lcd_display}
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
{external_citations_block}

Write the full evidence brief now."""

            context_chars = len(context)
            self._trace(
                "llm_brief_generation",
                "Brief generation starting",
                f"context={context_chars} chars · max_tokens={_BRIEF_TOKEN_LIMIT} · timeout={_BRIEF_MAX_SECONDS}s",
                "ok", 0,
            )

            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()

            # Heartbeat #0 — fires immediately to confirm the event pipeline end-to-end.
            _emit_heartbeat(0, 0, 0.0, False)

            raw_stream = self._llm_stream(STEP5_SYSTEM, context, max_tokens=_BRIEF_TOKEN_LIMIT)
            for chunk in _stream_with_timeout(raw_stream, _BRIEF_MAX_SECONDS):
                if not first_chunk_received:
                    first_chunk_received = True
                    self._trace(
                        "llm_brief_generation",
                        "First chunk received",
                        f"{time.time() - t0:.1f}s to first token",
                        "ok", int((time.time() - t0) * 1000),
                    )

                brief_chunks.append(chunk)
                if on_chunk:
                    on_chunk(chunk)

                _chars_ref[0] = sum(len(c) for c in brief_chunks)

        except Exception as e:
            brief_chunks = [f"[Brief generation error: {e}]"]
            self._trace("llm_brief_generation", "Brief generation exception", str(e), "error",
                        int((time.time() - t0) * 1000))
        finally:
            _hb_stop.set()
            if hb_thread is not None:
                hb_thread.join(timeout=2)

        elapsed = int((time.time() - t0) * 1000)

        if not first_chunk_received:
            timed_out = True
            self._trace(
                "llm_brief_generation",
                "Brief timed out — no first chunk",
                f"Serving endpoint returned no tokens in {_BRIEF_MAX_SECONDS}s — likely endpoint overload or auth issue",
                "error", elapsed,
            )
        elif elapsed >= _BRIEF_MAX_SECONDS * 1000:
            timed_out = True
            self._trace(
                "llm_brief_generation",
                "Brief streaming timed out",
                f"Reached {_BRIEF_MAX_SECONDS}s wall-clock limit at {sum(len(c) for c in brief_chunks)} chars",
                "warn", elapsed,
            )

        brief = "".join(brief_chunks)
        last_char = brief.rstrip()[-1:] if brief.rstrip() else ''
        brief_truncated = timed_out or last_char not in {'.', '!', '?', '\n', '#', '*', '|', '-', '—'}
        self._trace(
            "llm_brief_generation",
            "Brief generation complete",
            f"{len(brief)} chars" + (" — timed out" if timed_out else " — may be truncated" if brief_truncated else " — ok"),
            "warn" if brief_truncated else "ok", elapsed,
        )
        self.on_step(5, "complete", "Evidence brief ready" + (f" (partial — timed out after {_BRIEF_MAX_SECONDS}s)" if timed_out else ""))
        return brief

    # ── Step 6: Quality Review ─────────────────────────────────────────────────

    def step6_quality_review(self, brief: str, restrictions_data: dict, mapping: dict) -> dict:
        self.on_step(6, "running", "Running 4 quality judges in parallel…")

        # Sanitize and truncate brief — with 4 restrictions + 6 sections, R4 starts
        # at ~14k chars; use 30k to ensure all restriction responses reach judges.
        sanitized_brief = re.sub(r'\d{8,}', '[NUM]', brief)
        if len(sanitized_brief) > 30000:
            sanitized_brief = sanitized_brief[:30000] + "\n\n[Brief truncated for evaluation — first 30,000 chars shown]"
        brief_context = f"""BRIEF TO EVALUATE:
{sanitized_brief}

RESTRICTIONS:
{json.dumps(restrictions_data.get('restrictions', []), indent=2)}

EVIDENCE MAPPING SUMMARY:
{json.dumps(mapping.get('restriction_rebuttals', []), indent=2)}"""

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
                raw = self._llm(system_prompt, brief_context, max_tokens=_JUDGE_TOKEN_LIMIT)
                elapsed = int((time.time() - t0) * 1000)
                result, truncated = _parse_llm_json(raw, {"score": 0, "error": "parse_failed"})
                if truncated:
                    self._trace(f"llm_judge_{name}", f"Judge: {name}",
                                f"Output truncated at {len(raw)} chars", "warn", elapsed)
                    result = {"score": 0, "error": f"Output truncated at {len(raw)} chars"}
                else:
                    self._trace(f"llm_judge_{name}", f"Judge: {name}",
                                f"Score: {result.get('score', '?')}/10", "ok", elapsed)
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
        self.on_step(6, "complete", f"Quality review complete — {scores}")
        return results

    # ── Full Workflow ──────────────────────────────────────────────────────────

    def run(
        self,
        lcd_id: str = "MCD-A-XXXXX",
        on_chunk: Optional[Callable] = None,
        case_context: Optional[dict] = None,
    ) -> str:
        restrictions_data = self.step1_ingest(lcd_id)
        drug_profile      = self.step2a_drug_class_profile(restrictions_data)
        external          = self.step2b_external_evidence(restrictions_data, drug_profile)
        internal          = self.step3_internal_evidence()
        mapping           = self.step4_evidence_mapping(restrictions_data, external, internal)
        brief             = self.step5_generate_brief(restrictions_data, external, internal, mapping, on_chunk, case_context)
        try:
            self.step6_quality_review(brief, restrictions_data, mapping)
        except Exception as e:
            self._trace("llm_judge", "Quality review failed", str(e), "error", 0)
            self.on_step(6, "complete", f"Quality review error: {e}")
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
    try:
        total = sum(int(r.get("patients") or 0) for r in rows)
        zero = next((int(r.get("patients") or 0) for r in rows if str(r.get("prior_pe_courses")) == "0"), 0)
        pct = round(100 * zero / total, 1) if total else 0
        return f"{zero}/{total} patients ({pct}%) had 0 prior PE courses"
    except Exception:
        return "data unavailable"


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
    try:
        r = rows[0]
        return (
            f"{r.get('total_patients', '?')} outpatient patients, "
            f"{r.get('hospitalizations', '?')} hospitalizations, "
            f"{r.get('serious_aes', '?')} serious AEs, "
            f"{r.get('sustained_12mo', '?')} sustained at 12mo"
        )
    except Exception:
        return "data unavailable"


def _extract_jurisdiction(internal: dict) -> str:
    rows = internal.get("registry_cgs_jurisdiction", {}).get("rows", [])
    if not rows:
        return "data unavailable"
    try:
        total = sum(int(r.get("patients") or 0) for r in rows)
        states = [f"{r.get('treating_state')}={r.get('patients')}" for r in rows]
        return f"{total} patients in CGS states ({', '.join(states)})"
    except Exception:
        return "data unavailable"


_FIXTURE_RESTRICTIONS = {
    "lcd_id": "MCD-A-XXXXX",
    "mac_name": "CGS Administrators LLC",
    "mac_jurisdiction": "MO, KS, NE, IA",
    "drug_name": "Ultomiris",
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
