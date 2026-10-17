#!/usr/bin/env python3
"""Exercise ABI v1 through Godwit's real plugin-host, with a real engine and model.

    smoke.py LIBRARY --godwit GODWIT_CHECKOUT --engine ENGINE_DIR --model MODEL_DIR

ENGINE_DIR holds the pinned llama.cpp libraries, MODEL_DIR model.gguf and
decision_config.json, as Godwit lays out verified assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("library", type=Path)
parser.add_argument("--godwit", type=Path, required=True)
parser.add_argument("--engine", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
library = args.library.resolve()
digest = hashlib.sha256(library.read_bytes()).hexdigest()
host = root / "tmp" / ("godwit.exe" if sys.platform == "win32" else "godwit")
host.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["go", "build", "-o", str(host), "./cmd/godwit"], cwd=args.godwit, check=True,
               env={**__import__("os").environ, "CGO_ENABLED": "0"})

assets = {"engine": str(args.engine.resolve()), "model": str(args.model.resolve())}
bill = '<untrusted_email source="message:1">\nSubject: Your October invoice\nFrom: Acme <billing@acme.example>\n\nAmount due: 42.00 USD.\n</untrusted_email>'
questions = [{"name": "bill", "type": "predicate", "instructions": "Answer yes if this email is a bill, receipt or invoice."},
             {"name": "otp", "type": "predicate", "instructions": "Answer yes if this email carries a one-time sign-in code."}]
judge = {"modelId": "decision-2b-q8", "input": bill, "questions": questions}
requests = [
    ("describe", {}, True),
    ("configure", {"config": {}}, True),
    ("models.list", {}, True),
    ("models.judge", judge, False),  # nothing loaded yet: model_not_ready
]
first = "".join(json.dumps({"id": i, "method": m, "params": p}) + "\n" for i, (m, p, _) in enumerate(requests, 1))
# Loading waits for the answers before it; judging after it, eight at once, shares one model.
later = [("models.load", {"modelId": "decision-2b-q8", "assets": assets}, True)] + [("models.judge", judge, True)] * 8
cmd = [str(host), "plugin-host", "--library", str(library), "--sha256", digest]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")


def ask(lines, start):
    for i, (m, p, _) in enumerate(lines, start):
        proc.stdin.write(json.dumps({"id": i, "method": m, "params": p}) + "\n")
    proc.stdin.flush()
    got = [json.loads(proc.stdout.readline()) for _ in lines]
    return sorted(got, key=lambda r: r["id"])


out = ask(requests, 1)
out += ask(later[:1], len(requests) + 1)
out += ask(later[1:], len(requests) + 2)
proc.stdin.close()
assert proc.wait(timeout=60) == 0, proc.stderr.read()
for (method, params, ok), response in zip(requests + later, out):
    assert response["ok"] is ok, (method, response)
    if method == "describe":
        assert response["result"]["methods"] == ["models.list", "models.load", "models.judge"], response
    if method == "models.judge" and not ok:
        assert response["error"]["code"] == "model_not_ready", response
    if method == "models.judge" and ok:
        a = {x["name"]: x for x in response["result"]["answers"]}
        assert a["bill"]["probability"] > 0.5 > a["otp"]["probability"], response
print(json.dumps(out[-1]["result"]))
denied = subprocess.run([str(host), "plugin-host", "--library", str(library), "--sha256", "0" * 64], capture_output=True,
                        encoding="utf-8", timeout=30)
assert denied.returncode == 3 and "DENIED_BEFORE_DLOPEN" in denied.stderr, denied
print(f"Godwit ABI smoke test passed ({len(out)} calls, 8 concurrent judges, SHA-256 rejection).")
