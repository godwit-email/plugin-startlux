#!/usr/bin/env python3
"""Build the current platform's plugin library into dist/<platform>/."""
import os
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent.parent
os.chdir(root)
goos, goarch = subprocess.check_output(["go", "env", "GOOS", "GOARCH"], text=True).split()
extension = {"darwin": "dylib", "linux": "so", "windows": "dll"}[goos]
dest = root / "dist" / f"{goos}-{goarch}"
dest.mkdir(parents=True, exist_ok=True)
library = dest / f"plugin.{extension}"
subprocess.run(["go", "build", "-trimpath", "-buildmode=c-shared", "-ldflags=-s -w", "-o", str(library), "."], check=True)
size = library.stat().st_size
assert 0 < size <= 64 << 20, "the library exceeds Godwit's native artifact limit"
for name in ["plugin.json", "LICENSE"]:
    shutil.copyfile(root / name, dest / name)
shutil.copyfile(root / "third_party" / "llama" / "LICENSE", dest / "LICENSE.llama.cpp")
print(f"{library.relative_to(root)}: {size} bytes")
