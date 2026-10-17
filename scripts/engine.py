#!/usr/bin/env python3
"""Fetch the pinned llama.cpp build's libraries for a platform, laid out as Godwit stores the engine asset.

    engine.py OUT_DIR [--platform darwin-arm64]

Only the libraries the plugin loads go in OUT_DIR, flat and named as they find each other: macOS by
install name (libllama.0.dylib, through @loader_path), Linux by soname (libllama.so.0, through
$ORIGIN) with the CPU backend modules, Windows as the package names its DLLs. No executables.
"""
import argparse
import io
import platform as host
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = (ROOT / "third_party" / "llama" / "VERSION").read_text().strip()
PACKAGES = {
    "darwin-arm64": "macos-arm64.tar.gz",
    "darwin-amd64": "macos-x64.tar.gz",
    "linux-amd64": "ubuntu-x64.tar.gz",
    "linux-arm64": "ubuntu-arm64.tar.gz",
    "windows-amd64": "win-cpu-x64.zip",
    "windows-arm64": "win-cpu-arm64.zip",
}
MAC = ["llama", "ggml", "ggml-base", "ggml-cpu", "ggml-blas", "ggml-metal", "ggml-rpc"]


def this_platform():
    os_ = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}[host.system()]
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "AMD64": "amd64", "ARM64": "arm64"}[host.machine()]
    return f"{os_}-{arch}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--platform", default=this_platform(), choices=sorted(PACKAGES))
    args = ap.parse_args()
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{VERSION}/llama-{VERSION}-bin-{PACKAGES[args.platform]}"
    data = urllib.request.urlopen(url, timeout=300).read()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if url.endswith(".zip"):
            zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        else:
            safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}  # Python 3.12+: no paths out of tmp
            tarfile.open(fileobj=io.BytesIO(data)).extractall(tmp, **safe)
        files = {p.name: p for p in tmp.rglob("*") if p.is_file() or p.is_symlink()}
        args.out.mkdir(parents=True, exist_ok=True)
        picked = {}
        if args.platform.startswith("darwin"):
            for lib in MAC:
                versioned = [n for n in files if n.startswith(f"lib{lib}.0.") and n.count(".") == 4 and files[n].is_file() and not files[n].is_symlink()]
                if versioned:
                    picked[f"lib{lib}.0.dylib"] = files[versioned[0]]
        elif args.platform.startswith("linux"):
            for n, p in files.items():
                if n in ("libllama.so.0", "libggml.so.0", "libggml-base.so.0") or (n.startswith("libggml-") and n.endswith(".so") and n != "libggml-base.so"):  # backend modules
                    picked[n] = p
        else:
            for n, p in files.items():
                if n in ("llama.dll", "libomp.dll") or (n.startswith("ggml") and n.endswith(".dll")):
                    picked[n] = p
        for name in ("libllama.0.dylib", "libllama.so.0", "llama.dll"):
            if name in picked:
                break
        else:
            raise SystemExit(f"no libllama in {url}")
        for name, src in sorted(picked.items()):
            shutil.copyfile(src.resolve(), args.out / name)  # a symlink's target, as a plain file
            print(name)


if __name__ == "__main__":
    main()
