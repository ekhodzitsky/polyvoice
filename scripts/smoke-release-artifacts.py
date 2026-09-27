#!/usr/bin/env python3
"""Test release files in isolated consumer directories; never use source models.

Python 3.12+, Cargo, a C compiler, and platform loader inspection tools required.
The report is written only after every requested surface passes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import venv
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = ROOT / "scripts/artifact-consumers"
LINUX_LIBS = re.compile(r"^(lib(c|m|gcc_s|pthread|dl|rt|util)\.so(?:\.\d+)*|ld-linux[^/]*\.so(?:\.\d+)*)$")
WINDOWS_LIBS = re.compile(r"^(api-ms-win-[\w-]+|ext-ms-win-[\w-]+|kernel32|advapi32|bcrypt|bcryptprimitives|crypt32|ntdll|user32|userenv|ws2_32|ole32|combase|shell32|secur32|synchronization|ucrtbase|vcruntime140(?:_1)?|msvcp140)\.dll$", re.I)


def run(args, cwd, env=None):
    args = [str(x) for x in args]
    # Windows executable lookup does not use the child environment's PATH.
    if os.name == "nt" and env:
        args[0] = shutil.which(args[0], path=env.get("PATH")) or args[0]
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_result(data, duration):
    if not isinstance(data.get("num_speakers"), int) or data["num_speakers"] < 1:
        raise ValueError("real speech fixture must produce at least one speaker")
    turns = data.get("turns", [])
    if not turns:
        raise ValueError("real speech fixture must produce nonempty turns")
    for turn in turns:
        start, end = turn["time"]["start"], turn["time"]["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.1):
            raise ValueError(f"invalid time range: {turn}")
        if not isinstance(turn["speaker"], int) or not 0 <= turn["speaker"] < data["num_speakers"]:
            raise ValueError(f"invalid speaker: {turn}")
    return {"num_speakers": data["num_speakers"], "turns": turns}


def check_library(name, system, wheel=False):
    lower = name.lower()
    if any(x in lower for x in ("onnxruntime", "openblas", "torch", "tensorflow", "mkl", "libomp")):
        raise ValueError(f"undeclared inference/BLAS runtime: {name}")
    if system == "Linux":
        ok = LINUX_LIBS.fullmatch(Path(name).name) or (wheel and re.fullmatch(r"libpython3\.\d+\.so(?:\.\d+)*", Path(name).name))
    elif system == "Darwin":
        ok = name.startswith(("/usr/lib/", "/System/Library/Frameworks/"))
    else:
        ok = WINDOWS_LIBS.fullmatch(name) or (wheel and re.fullmatch(r"python3\d*\.dll", lower))
    if not ok:
        raise ValueError(f"undocumented native dependency: {name}")


def native_imports(path, env, wheel=False):
    system = platform.system()
    if system == "Linux":
        direct = run(["readelf", "-d", path], path.parent, env)
        names = re.findall(r"\(NEEDED\).*\[(.*?)\]", direct)
        closure = run(["ldd", path], path.parent, env)
        if "not found" in closure:
            raise ValueError(f"unresolved loader dependency:\n{closure}")
        names += re.findall(r"^\s*(\S+)\s+=>", closure, re.M)
    elif system == "Darwin":
        text = run(["otool", "-L", path], path.parent, env)
        names = [line.strip().split(" (")[0] for line in text.splitlines()[1:]]
        # The library's install name can differ from its installed filename.
        identity = run(["otool", "-D", path], path.parent, env)
        own_names = {line.strip() for line in identity.splitlines()[1:]}
        names = [n for n in names if n not in own_names]
    else:
        text = run(["dumpbin", "/DEPENDENTS", path], path.parent, env)
        names = re.findall(r"^\s+([\w.-]+\.dll)\s*$", text, re.M | re.I)
    if not names:
        raise ValueError(f"no native dependencies found for {path}")
    print(f"{path.name} native imports: {sorted(set(names))}", flush=True)
    for name in set(names):
        check_library(name, system, wheel)
    return sorted(set(names))


def compiler_env():
    env = {k.upper(): v for k, v in os.environ.items()} if os.name == "nt" else os.environ.copy()
    if os.name == "nt" and not shutil.which("cl"):
        vswhere = Path(env["PROGRAMFILES(X86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
        install = run([vswhere, "-latest", "-products", "*", "-requires",
                       "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], ROOT).strip()
        vcvars = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
        # Keep batch syntax out of subprocess's Windows argv quoting.
        with tempfile.TemporaryDirectory(prefix="polyvoice-msvc-") as directory:
            script = Path(directory) / "environment.cmd"
            script.write_text(f'@call "{vcvars}" >nul\n@if errorlevel 1 exit /b 1\n@set\n')
            output = run(["cmd", "/d", "/c", script], ROOT)
        for line in output.splitlines():
            if "=" in line and not line.startswith("="):
                key, value = line.split("=", 1)
                env[key.upper()] = value
    return env


def assets(destination, source=None):
    manifest = tomllib.loads((ROOT / "src/models/manifest.toml").read_text())
    ids = ["powerset_int8", "resnet34_int8"] + [k for k in manifest["models"] if k.startswith("vbx_plda_")]
    hashes = {}
    destination.mkdir(parents=True, exist_ok=True)
    for key in ids:
        entry = manifest["models"][key]
        path = destination / entry["filename"]
        if source:
            shutil.copyfile(source / path.name, path)
        else:
            with urllib.request.urlopen(entry["url"], timeout=120) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
        if sha(path) != entry["sha256"]:
            raise ValueError(f"asset checksum mismatch: {key}")
        hashes[key] = entry["sha256"]
    return hashes


def unpack_crate(archive, directory):
    with tarfile.open(archive) as tar:
        tar.extractall(directory, filter="data")
    roots = list(directory.glob("*/Cargo.toml"))
    if len(roots) != 1:
        raise ValueError("crate must contain exactly one root Cargo.toml")
    return roots[0].parent


def rust_consumer(archive, kernel, work, models, wav, env, report, target_dir):
    root = unpack_crate(archive, work / "crate")
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    dep = manifest["dependencies"]["polyvoice-kernels"]
    if "path" in dep:
        raise ValueError("packaged Rust dependency still points to a local workspace")
    kernel_root = unpack_crate(kernel, work / "kernel") if kernel else None
    consumer = work / "rust-consumer"
    (consumer / "src").mkdir(parents=True)
    base = ('[package]\nname="artifact-consumer"\nversion="0.0.0"\nedition="2024"\n'
            '[workspace]\n[features]\nlocal=[]\n[dependencies]\nserde_json="1"\n')
    rust_env = os.environ.copy()
    if target_dir:
        rust_env["CARGO_TARGET_DIR"] = str(target_dir)
    for label, features, source in [("byo", [], "byo.rs"), ("byo-vbx", ["clusterer", "vbx"], "byo.rs"),
                                     ("native", ["pipeline-native", "vbx"], "native.rs"),
                                     ("local", ["pipeline-local"], "native.rs")]:
        config = base + f'polyvoice={{path={json.dumps(root.as_posix())}, default-features=false, features={json.dumps(features)}}}\n'
        if kernel_root:
            config += f'[patch.crates-io]\npolyvoice-kernels={{path={json.dumps(kernel_root.as_posix())}}}\n'
        (consumer / "Cargo.toml").write_text(config)
        shutil.copyfile(CONSUMERS / source, consumer / "src/main.rs")
        extra = ["--features", "local"] if label == "local" else []
        metadata = json.loads(run(["cargo", "metadata", "--format-version", "1", *extra], consumer, rust_env))
        if label in ("native", "local"):
            selected = next(p for p in metadata["packages"] if p["name"] == "polyvoice-kernels")
            if not kernel and not (selected["source"] or "").startswith("registry+"):
                raise ValueError("release consumer did not resolve published kernels")
            report["kernel_resolution"] = {k: selected[k] for k in ["name", "version", "source"]}
        run(["cargo", "build", "--locked", "--release", *extra], consumer, rust_env)
        binary = Path(metadata["target_directory"]) / "release" / ("artifact-consumer.exe" if os.name == "nt" else "artifact-consumer")
        output = run([binary, models, wav], work, env)
        report["imports"]["rust-" + label] = native_imports(binary, env)
        report["surfaces"]["rust-" + label] = json.loads(output) if label in ("native", "local") else "passed"
    report["rust_dependency_mode"] = "staged archives (not publication-ready)" if kernel else "published registry"


def smoke(args):
    args.report.unlink(missing_ok=True)
    env = compiler_env()
    env = {k: v for k, v in env.items() if not k.startswith("POLYVOICE_") and k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")}
    expected_version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    report = {"version": expected_version, "revision": run(["git", "rev-parse", "HEAD"], ROOT).strip(), "platform": platform.platform(),
              "python": platform.python_version(), "worktree_status": run(["git", "status", "--porcelain", "--untracked-files=no"], ROOT).strip(), "artifacts": {}, "surfaces": {}, "imports": {}}
    for label in ("cli", "ffi", "wheel", "crate", "staged_kernel"):
        path = getattr(args, label)
        if path:
            report["artifacts"][label] = {"file": path.name, "sha256": sha(path)}
    with tempfile.TemporaryDirectory(prefix="polyvoice-consumer-") as directory:
        work = Path(directory)
        models = work / "models"
        report["models"] = assets(models, args.assets_dir)
        wav = work / "speech.wav"
        shutil.copyfile(ROOT / "tests/data/e2e-smoke/audio/fuzfh.wav", wav)
        with wave.open(str(wav)) as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16000:
                raise ValueError("fixture must be mono PCM16 at 16 kHz")
            duration = audio.getnframes() / 16000
            samples = [v[0] / 32768.0 for v in struct.iter_unpack("<h", audio.readframes(audio.getnframes()))]
        if not 1 <= duration <= 60:
            raise ValueError("fixture duration must be bounded")
        report["fixture"] = {"sha256": sha(wav), "duration_secs": duration}
        pcm = work / "speech.f32"
        pcm.write_bytes(struct.pack(f"<{len(samples)}f", *samples))
        env.update(POLYVOICE_VBX_PLDA_DIR=str(models), XDG_CACHE_HOME=str(work / "empty-cache"),
                   LOCALAPPDATA=str(work / "empty-cache"), HF_HOME=str(work / "empty-cache"),
                   HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="")
        if args.cli:
            cli = work / args.cli.name
            shutil.copyfile(args.cli, cli)
            cli.chmod(0o755)
            if run([cli, "--version"], work, env).strip() != "polyvoice " + expected_version:
                raise ValueError("CLI version does not match release")
            report["imports"]["cli"] = native_imports(cli, env)
            report["surfaces"]["cli"] = json.loads(run([cli, "diarize", wav, "--models-cache", models, "--json"], work, env))
        if args.ffi:
            ffi = work / "ffi"
            with zipfile.ZipFile(args.ffi) as archive:
                archive.extractall(ffi)
            system = platform.system()
            library = ffi / {"Linux": "libpolyvoice.so", "Darwin": "libpolyvoice.dylib", "Windows": "polyvoice.dll"}[system]
            report["imports"]["ffi"] = native_imports(library, env)
            shutil.copyfile(CONSUMERS / "ffi.c", ffi / "consumer.c")
            executable = ffi / ("consumer.exe" if os.name == "nt" else "consumer")
            if os.name == "nt":
                run(["cl", "/nologo", "/W4", "/WX", "/D_CRT_SECURE_NO_WARNINGS", "/I.", "consumer.c", "polyvoice.dll.lib", "/Fe:consumer.exe"], ffi, env)
            else:
                run(["cc", "-Wall", "-Wextra", "-Werror", "-I", ffi, ffi / "consumer.c", "-L", ffi,
                     "-lpolyvoice", "-o", executable], ffi, env)
                env["DYLD_LIBRARY_PATH" if system == "Darwin" else "LD_LIBRARY_PATH"] = str(ffi)
            report["surfaces"]["ffi"] = json.loads(run([executable, models, pcm], ffi, env))
        if args.wheel:
            venv.EnvBuilder(with_pip=True).create(work / "venv")
            python = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            run([python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", args.wheel], work, env)
            code = ('import json,struct,pathlib,polyvoice; '
                    f'samples=[x[0] for x in struct.iter_unpack("<f",pathlib.Path({str(pcm)!r}).read_bytes())]; '
                    f'p=polyvoice.Pipeline.balanced({str(models)!r},vbx_plda_dir={str(models)!r}); '
                    'print(p.run_result(samples,16000).to_json())')
            report["surfaces"]["wheel"] = json.loads(run([python, "-I", "-c", code], work, env))
            extensions = list((work / "venv").rglob("*.pyd" if os.name == "nt" else "*.so"))
            extensions = [p for p in extensions if "_polyvoice" in p.name]
            if len(extensions) != 1:
                raise ValueError("expected one installed polyvoice extension")
            report["imports"]["wheel"] = native_imports(extensions[0], env, wheel=True)
        if args.crate:
            rust_consumer(args.crate, args.staged_kernel, work, models, wav, env, report, args.cargo_target_dir)
        for label, result in list(report["surfaces"].items()):
            if isinstance(result, dict):
                if result.get("provenance", {}).get("version") != expected_version:
                    raise ValueError(f"{label} result version does not match release")
                report["surfaces"][label] = validate_result(result, duration)
        native_results = [r for r in report["surfaces"].values() if isinstance(r, dict)]
        if any(r != native_results[0] for r in native_results[1:]):
            raise ValueError("packaged front doors disagree on the deterministic fixture")
        report["status"] = "passed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"artifact smoke passed: {args.report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "ffi", "wheel", "crate", "staged-kernel", "assets-dir", "cargo-target-dir"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve())
    parser.add_argument("--report", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if not any((args.cli, args.ffi, args.wheel, args.crate)) or (args.staged_kernel and not args.crate):
        parser.error("provide an artifact; --staged-kernel requires --crate")
    smoke(args)


if __name__ == "__main__":
    main()
