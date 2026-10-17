#!/usr/bin/env python3
"""Reference answers for the parity tests, from the StartLux reference
implementation (startlux_decide) and a stock llama-server.

    reference.py --tokenizer MODEL_DIR --server URL CASES.json OUT.json

MODEL_DIR is the model's Hugging Face folder (tokenizer, decision_config.json);
the server runs the same model's GGUF with the settings of
docs/parity.md. For each case (a Godwit question about a state) it writes the
prompt as the reference renders it, its token ids, the letter
log-probabilities the server returns and the option probabilities, which the
Go tests compare with the plugin's own.
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("STARTLUX_LLAMACPP_BIDIR", "server")  # stock server, non-causal context

from startlux_decide import prompt, readout  # noqa: E402
from startlux_decide.backends.llamacpp import BIAS, NEUTRAL  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def level_text(level):
    label, desc = level.get("label", ""), level.get("description", "")
    if not desc:
        return label
    if not label:
        return desc
    return f"{label}: {desc}"


def systemone_spec(q):
    """A Godwit question as Godwit's System One client sends it, choices in the question's order."""
    if q["type"] == "predicate":
        return {"type": "noul", "instructions": q["instructions"]}
    if q["type"] == "choice":
        return {"type": "choice", "instructions": q["instructions"],
                "criteria": {c["value"]: c.get("description", "") for c in q["choices"]}}
    return {"type": "score", "instructions": q["instructions"], "criteria": [level_text(lv) for lv in q["levels"]]}


def completion(server, ids, letter_ids):
    k = len(letter_ids)
    body = dict(NEUTRAL, prompt=ids, n_predict=1, n_probs=k, post_sampling_probs=True, top_k=k,
                logit_bias=[[t, BIAS] for t in letter_ids], cache_prompt=False, seed=0)
    req = urllib.request.Request(server.rstrip("/") + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    got = {c["id"]: (math.log(c["prob"]) if c["prob"] > 0 else -1e9)
           for c in out["completion_probabilities"][0]["top_probs"]}
    return [got[t] for t in letter_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("cases", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = json.loads((args.tokenizer / "decision_config.json").read_text())
    out = []
    for case in json.loads(args.cases.read_text()):
        q = case["question"]
        row = prompt.from_systemone(case["state"], systemone_spec(q))
        text = prompt.plain_prompt(tok, prompt.messages(row))
        ids = tok.encode(text, add_special_tokens=False)
        z = completion(args.server, ids, cfg["letter_token_ids"][:len(row["options"])])
        p = readout.probabilities(z, cfg["temperature_by_type"][row["type"]])
        out.append({"name": case["name"], "text": text, "ids": ids, "logprobs": z, "probabilities": p})
        print(f"{case['name']}: {len(ids)} tokens", file=sys.stderr)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
