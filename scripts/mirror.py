#!/usr/bin/env python3
"""Mirror the engine and models to dl.godwit.email and print the registry assets.

    mirror.py WORK_DIR [--upload]

scripts/engine.py lays out the pinned llama.cpp build (third_party/llama/VERSION)
per platform, as Godwit stores the engine asset. The models come from startlux-models on Hugging Face at
pinned revisions and must have the hashes below. Without --upload nothing is
written to the bucket. With it, each file goes to s3://godwit-downloads/plugins/
startlux/... once: a path that already holds other bytes fails the run, since a
published release's files never change. WORK_DIR/assets.json is the release's
"assets" for registry.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

BUCKET = "godwit-downloads"
PUBLIC = "https://dl.godwit.email"
PLATFORMS = ["darwin-arm64", "darwin-amd64", "linux-amd64", "linux-arm64", "windows-amd64", "windows-arm64"]
GPU = ["darwin-arm64"]  # only Apple silicon has a GPU back end; elsewhere only 2B is offered (CPU)

HF = "https://huggingface.co/startlux-models/StartLux-Decision-v1.1-"
MODELS = {
    "4b-q4": {
        "model.gguf": (HF + "4B-Instant-GGUF/resolve/a9ab4199fdff586f86e3baf8c9557f6c5ca0fdc0/StartLux-Decision-v1.1-4B-Instant-Q4_K_M.gguf",
                       "5fcfe9771db996a868f78215c08955c21cf5e8188609e17e9c653183d946e84d"),
        "decision_config.json": (HF + "4B-Instant/resolve/416088f77cfce09633906a3675c7cd2e5dabcf10/decision_config.json",
                                 "9ca014791b5c71996c212c7aed6275405c4d9b682a6b6eba325a3ab17bcccc08"),
    },
    "2b-q8": {
        "model.gguf": (HF + "2B-Instant-GGUF/resolve/afb95dbba67e648f4e1b5ad3f770a91f799f9496/StartLux-Decision-v1.1-2B-Instant-Q8_0.gguf",
                       "170cf86482a3340c7bd2000f921aa482820a992c14587f496f43fc57009f9529"),
        "decision_config.json": (HF + "2B-Instant/resolve/f1e6cd147dda82c9ffa3e8e725d5c1f175566fb1/decision_config.json",
                                 "3bd3c0c3854b18be967c7a660d38b8340aa2b4c14648d8ce0260cd32f6e89c1a"),
    },
}
# Physical memory each needs, measured (docs/parity.md): Windows commits the
# 4K-token compute buffer that macOS and Linux only reserve.
MEMORY = {"4b-q4": {"default": 8}, "2b-q8": {"default": 6, "windows-*": 12}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, want):
    if not (dest.exists() and sha256(dest) == want):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    if (got := sha256(dest)) != want:
        sys.exit(f"{url}: sha256 {got}, not {want}")


def aws(*args, check=True):
    return subprocess.run(["aws", *args, "--endpoint-url", os.environ["R2"]], check=check, capture_output=True, text=True)


def publish(local, key, digest, upload):
    """The registry file for local at key: uploaded once, never replaced."""
    if upload:
        head = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key, check=False)
        if head.returncode == 0:
            have = json.loads(head.stdout).get("Metadata", {}).get("sha256")
            if have != digest:
                sys.exit(f"{key} already holds other bytes ({have}); a published file never changes")
            print(f"{key}: already there")
        else:
            aws("s3", "cp", str(local), f"s3://{BUCKET}/{key}", "--metadata", f"sha256={digest}",
                "--content-type", "application/octet-stream", "--cache-control", "public, max-age=31536000, immutable")
            print(f"{key}: uploaded")
    return {"name": local.name, "url": f"{PUBLIC}/{key}", "sha256": digest, "size": local.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if args.upload and not os.environ.get("R2"):
        sys.exit("--upload needs R2 (the bucket's endpoint) and the AWS_* credentials")
    root = Path(__file__).resolve().parent.parent
    version = (root / "third_party" / "llama" / "VERSION").read_text().strip()

    assets = []
    for platform in PLATFORMS:
        out = args.work / "engine" / platform
        if not out.exists():
            subprocess.run([sys.executable, str(root / "scripts" / "engine.py"), str(out), "--platform", platform], check=True)
        files = [publish(p, f"plugins/startlux/engine/{version}/{platform}/{p.name}", sha256(p), args.upload)
                 for p in sorted(out.iterdir())]
        assets.append({"name": "engine", "platforms": [platform], "variants": [{"id": version, "files": files}]})

    variants = {}
    for variant, files in MODELS.items():
        published = []
        for name, (url, want) in files.items():
            local = args.work / "models" / variant / name
            download(url, local, want)
            published.append(publish(local, f"plugins/startlux/models/v1.1/{variant}/{name}", want, args.upload))
        variants[variant] = {"id": variant, "minMemoryGB": MEMORY[variant], "files": published}
    cpu = [p for p in PLATFORMS if p not in GPU]
    assets.append({"name": "model", "platforms": GPU, "variants": [variants["4b-q4"], variants["2b-q8"]]})
    assets.append({"name": "model", "platforms": cpu, "variants": [variants["2b-q8"]]})

    (args.work / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    print(f"assets: {args.work / 'assets.json'}")


if __name__ == "__main__":
    main()
