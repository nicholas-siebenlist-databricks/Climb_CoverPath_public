import json
import os
import threading

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Global workflow state ─────────────────────────────────────────────────────
_lock = threading.Lock()
_events: list = []
_state = {
    "status": "idle",   # idle | running | done | error
    "brief": "",
    "threat_status": "Workflow Pending",
    "judges": {},
    "case_context": {},
    "extracted_data": None,
}


def _push(event: dict):
    with _lock:
        _events.append(event)


def _on_step(idx: int, status: str, message: str):
    _push({"type": "step", "idx": idx, "status": status, "message": message})


def _on_trace(ev: dict):
    event = {"type": "trace", **ev}
    t = event.get("type")
    if t == "judge_result":
        with _lock:
            _state["judges"][ev["judge"]] = ev["result"]
    elif t == "restrictions_extracted":
        with _lock:
            _state["extracted_data"] = ev
    _push(event)


def _run_workflow(lcd_id: str, case_context: dict):
    from agent.orchestrator import WorkflowOrchestrator

    with _lock:
        _events.clear()
        _state["status"] = "running"
        _state["brief"] = ""
        _state["judges"] = {}
        _state["extracted_data"] = None

    try:
        orch = WorkflowOrchestrator(on_step=_on_step, on_trace=_on_trace)

        def on_chunk(chunk: str):
            _push({"type": "brief_chunk", "content": chunk})
            with _lock:
                _state["brief"] += chunk

        orch.run(lcd_id=lcd_id, on_chunk=on_chunk, case_context=case_context)

        with _lock:
            _state["status"] = "done"
            _state["threat_status"] = "Review Required"

        _push({"type": "done", "threat_status": "Review Required"})

    except Exception as e:
        with _lock:
            _state["status"] = "error"
        _push({"type": "error", "message": str(e)})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def get_status():
    with _lock:
        return jsonify({
            "status": _state["status"],
            "threat_status": _state["threat_status"],
            "event_count": len(_events),
        })


@app.route("/api/events")
def get_events():
    cursor = int(request.args.get("cursor", 0))
    with _lock:
        batch = _events[cursor:]
        status = _state["status"]
        count = len(_events)
    return jsonify({"events": batch, "status": status, "cursor": count})


@app.route("/api/run", methods=["POST"])
def run_workflow():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Workflow already running"}), 409

    data = request.get_json(force=True) or {}
    lcd_id = data.get("lcd_id", "MCD-A-XXXXX")
    case_context = data.get("case_context", {})
    with _lock:
        _state["case_context"] = case_context

    thread = threading.Thread(target=_run_workflow, args=(lcd_id, case_context), daemon=True)
    thread.start()
    return jsonify({"ok": True, "lcd_id": lcd_id})


@app.route("/api/brief")
def get_brief():
    with _lock:
        return jsonify({"brief": _state["brief"], "status": _state["status"]})


@app.route("/api/reset", methods=["POST"])
def reset_workflow():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Cannot reset while running"}), 409
        _events.clear()
        _state["status"] = "idle"
        _state["brief"] = ""
        _state["threat_status"] = "Workflow Pending"
        _state["judges"] = {}
        _state["case_context"] = {}
        _state["extracted_data"] = None
    return jsonify({"ok": True})


@app.route("/api/lcd-source")
def get_lcd_source():
    import os as _os
    from bs4 import BeautifulSoup
    fixture_path = _os.path.join(_os.path.dirname(__file__), "fixtures/mcd_a_xxxxx.html")
    try:
        with open(fixture_path, "r") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"text": "", "error": str(e)})


@app.route("/api/judges")
def get_judges():
    with _lock:
        return jsonify({"judges": _state["judges"]})


@app.route("/api/mark-counsel-review", methods=["POST"])
def mark_counsel():
    with _lock:
        _state["threat_status"] = "Counsel Review"
    return jsonify({"ok": True})


@app.route("/api/debug")
def get_debug():
    """Diagnostic endpoint: event type counts, heartbeats, and last error."""
    with _lock:
        total = len(_events)
        type_counts: dict = {}
        heartbeats = []
        errors = []
        for ev in _events:
            t = ev.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            if t == "debug_heartbeat":
                heartbeats.append(ev)
            elif t == "error":
                errors.append(ev)
        status = _state["status"]
    return jsonify({
        "status": status,
        "total_events": total,
        "type_counts": type_counts,
        "heartbeat_count": len(heartbeats),
        "heartbeats": heartbeats,
        "errors": errors,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
