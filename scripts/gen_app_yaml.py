#!/usr/bin/env python3
"""Generate app/app.yaml from app/app.yaml.tmpl using resolved DAB variables.

Usage:
    databricks bundle validate --target <target> --profile <profile> --output json \
        | python3 scripts/gen_app_yaml.py
"""
import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = os.path.join(ROOT, "app", "app.yaml.tmpl")
OUT  = os.path.join(ROOT, "app", "app.yaml")

data = json.load(sys.stdin)
vars_ = {k: v["value"] for k, v in data.get("variables", {}).items() if "value" in v}

template = open(TMPL).read()
rendered = template.format(**vars_)

with open(OUT, "w") as f:
    f.write(rendered)

print(f"wrote {OUT}", file=sys.stderr)
for k, v in sorted(vars_.items()):
    print(f"  {k} = {v}", file=sys.stderr)
