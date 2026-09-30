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


def run(args, cwd, env=None, reject=None):
    args = [str(x) for x in args]
    # Windows executable lookup does not use the child environment's PATH.
    if os.name == "nt" and env:
        args[0] = shutil.which(args[0], path=env.get("PATH")) or args[0]
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    if reject is not None:
        if result.returncode == 0 or re.search(reject, result.stderr, re.I) is None:
            raise ValueError(f'expected rejection {reject!r}: exit={result.returncode}, stderr={result.stderr}')
        return result.stderr
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_result(data, duration, speech=True, require_tail=False):
    if type(data.get('num_speakers')) is not int:
        raise ValueError('speaker count must be an integer')
    if not speech:
        if data.get("num_speakers") != 0 or data.get("turns") != []:
            raise ValueError("silence/short input must produce zero speakers and no turns")
        return {"num_speakers": 0, "turns": []}
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
    if require_tail:
        if not any(t["time"]["end"] > duration - 60 for t in turns):
            raise ValueError("speech at the end of the hour was not processed")
        if not any(t["time"]["start"] < 60 for t in turns):
            raise ValueError("speech at the beginning of the hour was not processed")
    return {"num_speakers": data["num_speakers"], "turns": turns}


def scenario_fixtures(work, speech):
    # Sparse files keep the one-hour boundary practical without allocating a
    # Python list of 57.6 million floats. Speech anchors both ends of the hour.
    values = [v[0] for v in struct.iter_unpack('<h', speech)]
    half = len(values) // 2
    mixed = struct.pack(f'<{half}h', *(int((a + b) / 2) for a, b in zip(values[:half], values[half:])))
    cases = {}
    for name, frames, content in [('speech', len(values), speech), ('empty', 0, b''),
                                  ('short', 100, b''), ('silence', 16000, b''),
                                  ('overlap', half, mixed), ('hour', 16000 * 3600, speech),
                                  ('too-long', 16000 * 3600 + 1, b'')]:
        wav, pcm = work / (name + '.wav'), work / (name + '.f32')
        header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + frames * 2, b'WAVE',
                             b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, b'data', frames * 2)
        floats = b''.join(struct.pack('<f', v[0] / 32768) for v in struct.iter_unpack('<h', content))
        for path, prefix, body, width in [(wav, header, content, 2), (pcm, b'', floats, 4)]:
            with path.open('wb') as out:
                out.write(prefix + body)
                out.truncate(len(prefix) + frames * width)
                if name == 'hour':
                    out.seek(len(prefix) + frames * width - len(body))
                    out.write(body)
        cases[name] = {'wav': wav, 'pcm': pcm, 'duration': frames / 16000,
                       'speech': name in ('speech', 'overlap', 'hour')}
    return cases


def offline_prefix(work, env, report):
    system = platform.system()
    if system == 'Linux':
        launcher = work / 'offline'
        run(['cc', '-Wall', '-Wextra', '-Werror', CONSUMERS / 'offline-linux.c', '-o', launcher], work, env)
        prefix = [launcher]
        method = 'Linux seccomp; socket creation and outbound calls denied'
    elif system == 'Darwin':
        prefix = ['/usr/bin/sandbox-exec', '-p', '(version 1)(allow default)(deny network*)']
        method = 'Darwin sandbox; network operations denied'
    elif system == 'Windows':
        launcher = work / 'offline.exe'
        run(['cl', '/nologo', '/W4', '/WX', '/D_CRT_SECURE_NO_WARNINGS',
             CONSUMERS / 'offline-windows.c', 'fwpuclnt.lib', '/Fe:' + str(launcher)], work, env)
        prefix = [launcher, sys.executable]
        method = 'Windows WFP dynamic application filters; IPv4/IPv6 bind and connect denied'
    else:
        raise ValueError('OS network isolation is not implemented on this platform')
    # A closed port/network outage is not an isolation success: require an OS
    # permission error for IPv4/IPv6 TCP and UDP in an actual child process.
    probe = """import errno,socket
for family, address in ((socket.AF_INET,('127.0.0.1',9)), (socket.AF_INET6,('::1',9))):
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        try:
            with socket.socket(family,kind) as connection:
                connection.settimeout(2)
                if kind == socket.SOCK_STREAM:
                    connection.connect(address)
                else:
                    connection.sendto(b'probe',address)
        except OSError as error:
            assert error.errno in (errno.EPERM,errno.EACCES), error
        else:
            raise AssertionError('offline filter allowed networking')
print('IPv4/IPv6 TCP/UDP denied')
"""
    control = """import socket
for family, address in ((socket.AF_INET,('127.0.0.1',0)), (socket.AF_INET6,('::1',0))):
    with socket.socket(family) as server:
        server.bind(address)
        server.listen()
        with socket.socket(family) as client:
            client.settimeout(2)
            client.connect(server.getsockname())
print('loopback available outside isolation')
"""
    run([sys.executable, '-c', control], work, env)
    report['network_isolation'] = {'method': method, 'probe': run([*prefix, sys.executable, '-c', probe], work, env).strip()}
    report['network_isolation']['cleanup_probe'] = run([sys.executable, '-c', control], work, env).strip()
    return prefix


def exercise(label, command, cases, work, env, prefix, report, cli=False):
    results = {}
    for name, case in cases.items():
        argv = command(name, case)
        report.setdefault('commands', {}).setdefault(label, {})[name] = [str(a) for a in [*prefix, *argv]]
        if cli and name in ('empty', 'short', 'too-long'):
            run([*prefix, *argv], work, env, reject='audio too short' if name != 'too-long' else 'too long')
            results[name] = {'error': name}
            continue
        result = json.loads(run([*prefix, *argv], work, env))
        if name in ('empty', 'short', 'too-long'):
            if result != {'error': name}:
                raise ValueError(f'{label}: over-limit input accepted')
            results[name] = result
        else:
            if result.get('provenance', {}).get('version') != report['version']:
                raise ValueError(f'{label}: result version mismatch')
            results[name] = validate_result(result, case['duration'], case['speech'], name == 'hour')
    if not cli:
        argv = [*prefix, *command('invalid-rate', cases['speech'])]
        report['commands'][label]['invalid-rate'] = [str(a) for a in argv]
        invalid = json.loads(run(argv, work, env))
        if invalid != {'error': 'invalid-rate'}:
            raise ValueError(f'{label}: invalid rate accepted')
        results['invalid-rate'] = invalid
    report.setdefault('scenarios', {})[label] = results
    return results['speech']


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


def rust_consumer(archive, kernel, work, models, cases, env, report, target_dir, prefix):
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
        if label in ('native', 'local'):
            result = exercise('rust-' + label, lambda name, case: [binary, models, case['wav'], name],
                              cases, work, env, prefix, report)
        else:
            report.setdefault('byo_scenarios', {})[label] = json.loads(run([*prefix, binary], work, env))
            result = 'passed'
        if label == 'local':
            forbidden = {'reqwest', 'ring', 'rustls', 'hyper', 'ureq'}
            resolved = {n['id'] for n in metadata['resolve']['nodes']}
            if any(p['name'] in forbidden and p['id'] in resolved for p in metadata['packages']):
                raise ValueError('local consumer resolves downloader/TLS dependencies')
            for name, expected in [('missing-model', 'OfflineMissing'), ('corrupt-model', 'ChecksumMismatch')]:
                bad = work / name
                shutil.copytree(models, bad)
                model = bad / 'powerset_int8.onnx'
                if name == 'missing-model':
                    model.unlink()
                else:
                    model.write_bytes(b'corrupt')
                run([*prefix, binary, bad, cases['speech']['wav'], 'speech'], work, env, reject=expected)
                report['scenarios']['rust-local'][name] = {'error': name}
                report['commands']['rust-local'][name] = [str(a) for a in [*prefix, binary, bad, cases['speech']['wav'], 'speech']]

        report["imports"]["rust-" + label] = native_imports(binary, env)
        report["surfaces"]["rust-" + label] = result
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
        fixture = ROOT / "tests/data/e2e-smoke/audio/fuzfh.wav"
        with wave.open(str(fixture)) as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError('fixture must be mono PCM16 at 16 kHz')
            speech = audio.readframes(audio.getnframes())
        if not 1 <= len(speech) / 32000 <= 60:
            raise ValueError('fixture must be bounded to 60 seconds')
        cases = scenario_fixtures(work, speech)
        report['fixture'] = {'sha256': sha(fixture), 'duration_secs': len(speech) / 32000}
        report['inputs'] = {name: {'samples': int(case['duration'] * 16000),
                                   'wav_sha256': sha(case['wav']), 'pcm_sha256': sha(case['pcm'])}
                            for name, case in cases.items()}
        env.update(POLYVOICE_VBX_PLDA_DIR=str(models), XDG_CACHE_HOME=str(work / "empty-cache"),
                   LOCALAPPDATA=str(work / "empty-cache"), HF_HOME=str(work / "empty-cache"),
                   HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="")
        prefix = offline_prefix(work, env, report)
        if args.cli:
            cli = work / args.cli.name
            shutil.copyfile(args.cli, cli)
            cli.chmod(0o755)
            if run([cli, "--version"], work, env).strip() != "polyvoice " + expected_version:
                raise ValueError("CLI version does not match release")
            report["imports"]["cli"] = native_imports(cli, env)
            report['surfaces']['cli'] = exercise('cli',
                lambda name, case: [cli, 'diarize', case['wav'], '--models-cache', models, '--json'],
                cases, work, env, prefix, report, cli=True)
            malformed = work / 'malformed.wav'
            malformed.write_bytes(b'not a WAVE file')
            for name, argv, expected in [
                ('malformed', [cli, 'diarize', malformed, '--models-cache', models, '--json'], 'not a WAVE container'),
                ('invalid-config', [cli, 'diarize', cases['speech']['wav'], '--profile', 'invalid'], 'unknown profile')]:
                run([*prefix, *argv], work, env, reject=expected)
                report['scenarios']['cli'][name] = {'error': name}
                report['commands']['cli'][name] = [str(a) for a in [*prefix, *argv]]
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
                     "-lpolyvoice", "-Xlinker", "-rpath", "-Xlinker", ffi, "-o", executable], ffi, env)
                env["DYLD_LIBRARY_PATH" if system == "Darwin" else "LD_LIBRARY_PATH"] = str(ffi)
            report['surfaces']['ffi'] = exercise('ffi',
                lambda name, case: [executable, models, case['pcm'], name], cases, ffi, env, prefix, report)
        if args.wheel:
            venv.EnvBuilder(with_pip=True).create(work / "venv")
            python = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            run([python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", args.wheel], work, env)
            if platform.system() == 'Windows':
                probe = 'import socket,errno; s=socket.socket();\ntry: s.connect(("127.0.0.1",9))\nexcept OSError as e: assert e.errno==errno.EACCES, e\nelse: raise AssertionError("wheel interpreter escaped offline filter")'
                report['network_isolation']['wheel_probe'] = run([*prefix, python, '-I', '-c', probe], work, env).strip() or 'permission denied'
            consumer = work / 'wheel-consumer.py'
            shutil.copyfile(CONSUMERS / 'wheel.py', consumer)
            report['surfaces']['wheel'] = exercise('wheel',
                lambda name, case: [python, '-I', consumer, models, case['pcm'], name],
                cases, work, env, prefix, report)
            extensions = list((work / "venv").rglob("*.pyd" if os.name == "nt" else "*.so"))
            extensions = [p for p in extensions if "_polyvoice" in p.name]
            if len(extensions) != 1:
                raise ValueError("expected one installed polyvoice extension")
            report["imports"]["wheel"] = native_imports(extensions[0], env, wheel=True)
        if args.crate:
            rust_consumer(args.crate, args.staged_kernel, work, models, cases, env, report, args.cargo_target_dir, prefix)
        for name in cases:
            results = [r[name] for r in report.get('scenarios', {}).values()]
            if any(r != results[0] for r in results[1:]):
                raise ValueError(f'packaged front doors disagree on {name}')
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
