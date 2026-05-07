"""M5 — validate INT8 artifacts against acceptance gates from spec §9.4.

Runs three checks per model:

  segmenter (powerset_int8):
    DER hit on hold-out  ≤ +0.5
    softmax KL divergence (output) ≤ 0.05

  embedder (cam_pp_int8 / resnet34_int8):
    EER on VoxCeleb1 hit ≤ +0.30
    cosine vs FP32       mean ≥ 0.998 / p1 ≥ 0.991

Exit code 0 on PASS, non-zero on any failure (with per-budget detail in report).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    sys.exit(f"onnxruntime missing: {exc}")

BUDGETS = {
    "powerset": {"der_delta_max": 0.5, "kl_max": 0.05},
    "embedder": {"eer_delta_max": 0.30, "cosine_mean_min": 0.998, "cosine_p1_min": 0.991},
}


def _load_onnx(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _powerset_compare(
    fp32_path: Path, int8_path: Path, hold_out_audio: Path, hold_out_rttm: Path
) -> dict[str, Any]:
    """Compute DER hit (FP32 → INT8) on VoxConverse-dev hold-out + max softmax KL."""
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.core import Annotation, Segment
    from pyannote.database.util import load_rttm
    import librosa
    import tempfile

    sess_fp32 = _load_onnx(fp32_path)
    sess_int8 = _load_onnx(int8_path)
    in_name = sess_fp32.get_inputs()[0].name

    wavs = sorted(hold_out_audio.glob("*.wav"))[:100]
    if not wavs:
        raise SystemExit(f"No hold-out WAVs in {hold_out_audio}")

    def _frames_to_rttm(probs_TxC: np.ndarray, hop_s: float, file_id: str) -> str:
        argmax = np.argmax(probs_TxC, axis=1)
        class_to_speakers = {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [0, 1],
            5: [0, 2],
            6: [1, 2],
        }
        active = [class_to_speakers[int(c)] for c in argmax]
        lines: list[str] = []
        for spk in range(3):
            in_run = False
            run_start = 0
            for t, frame_speakers in enumerate(active):
                if spk in frame_speakers:
                    if not in_run:
                        in_run = True
                        run_start = t
                else:
                    if in_run:
                        in_run = False
                        s = run_start * hop_s
                        e = t * hop_s
                        lines.append(
                            f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> SPK_{spk} <NA> <NA>"
                        )
            if in_run:
                s = run_start * hop_s
                e = len(active) * hop_s
                lines.append(
                    f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> SPK_{spk} <NA> <NA>"
                )
        return "\n".join(lines) + "\n"

    der_metric_fp32 = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    der_metric_int8 = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    kl_max_seen = 0.0

    def _read_ref(file_id: str) -> Annotation | None:
        path = hold_out_rttm / f"{file_id}.rttm"
        if not path.exists():
            return None
        ann_dict = load_rttm(str(path))
        return next(iter(ann_dict.values()))

    def _str_to_annotation(rttm_str: str) -> Annotation:
        with tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False) as f:
            f.write(rttm_str)
            tmp_path = f.name
        ann_dict = load_rttm(tmp_path)
        return next(iter(ann_dict.values()))

    n_compared = 0
    for wav in wavs:
        file_id = wav.stem
        ref = _read_ref(file_id)
        if ref is None:
            continue

        audio, _ = librosa.load(str(wav), sr=16000, mono=True, duration=10.0)
        if audio.shape[0] < 16000:
            continue
        target_T = 160000
        if audio.shape[0] < target_T:
            audio = np.concatenate(
                [audio, np.zeros(target_T - audio.shape[0], dtype=np.float32)]
            )
        x = audio[:target_T].astype(np.float32).reshape(1, 1, -1)

        out_fp32 = sess_fp32.run(None, {in_name: x})[0]
        out_int8 = sess_int8.run(None, {in_name: x})[0]
        probs_fp32 = _softmax(out_fp32[0], axis=1)
        probs_int8 = _softmax(out_int8[0], axis=1)
        kl = _kl_divergence(probs_fp32, probs_int8)
        kl_max_seen = max(kl_max_seen, float(kl.max()))

        hop_s = 10.0 / probs_fp32.shape[0]
        rttm_fp32 = _frames_to_rttm(probs_fp32, hop_s, file_id)
        rttm_int8 = _frames_to_rttm(probs_int8, hop_s, file_id)

        ref_window = ref.crop(Segment(0.0, 10.0))
        der_metric_fp32(ref_window, _str_to_annotation(rttm_fp32))
        der_metric_int8(ref_window, _str_to_annotation(rttm_int8))
        n_compared += 1

    fp32_der = abs(der_metric_fp32) * 100
    int8_der = abs(der_metric_int8) * 100
    return {
        "fp32_der": fp32_der,
        "int8_der": int8_der,
        "der_delta": int8_der - fp32_der,
        "kl_max": kl_max_seen,
        "n_compared": n_compared,
    }


def _embedder_compare(
    fp32_path: Path,
    int8_path: Path,
    voxceleb_audio: Path,
    voxceleb_trials: Path,
    hold_out_audio: Path,
    embed_input_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Compute EER hit + cosine FP32 vs INT8 over VoxCeleb1 trials + hold-out audio."""
    import librosa

    sess_fp32 = _load_onnx(fp32_path)
    sess_int8 = _load_onnx(int8_path)
    in_name = sess_fp32.get_inputs()[0].name

    rng = np.random.default_rng(42)
    wavs = sorted(hold_out_audio.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No hold-out audio in {hold_out_audio}")
    chunks_to_test = min(200, len(wavs) * 3)
    cosines: list[float] = []
    for _ in range(chunks_to_test):
        wav = wavs[int(rng.integers(0, len(wavs)))]
        audio, _ = librosa.load(str(wav), sr=16000, mono=True, duration=3.0)
        if audio.shape[0] < 16000:
            continue
        feat = _audio_to_input(audio, embed_input_shape)
        emb_fp32 = sess_fp32.run(None, {in_name: feat})[0].flatten()
        emb_int8 = sess_int8.run(None, {in_name: feat})[0].flatten()
        cos = _cosine(emb_fp32, emb_int8)
        cosines.append(cos)
    cos_arr = np.array(cosines) if cosines else np.array([0.0])
    cos_mean = float(cos_arr.mean())
    cos_p1 = float(np.percentile(cos_arr, 1))

    pairs = _read_trials(voxceleb_trials)
    scores_fp32: list[float] = []
    scores_int8: list[float] = []
    labels: list[int] = []
    for label, a, b in pairs[:1000]:
        a_path = voxceleb_audio / a
        b_path = voxceleb_audio / b
        if not (a_path.exists() and b_path.exists()):
            continue
        ea_fp32, eb_fp32 = _embed_pair(sess_fp32, in_name, a_path, b_path, embed_input_shape)
        ea_int8, eb_int8 = _embed_pair(sess_int8, in_name, a_path, b_path, embed_input_shape)
        scores_fp32.append(_cosine(ea_fp32, eb_fp32))
        scores_int8.append(_cosine(ea_int8, eb_int8))
        labels.append(int(label))
    if labels:
        eer_fp32 = _eer(np.array(labels), np.array(scores_fp32))
        eer_int8 = _eer(np.array(labels), np.array(scores_int8))
    else:
        # No usable trial pairs (audio not on disk). Report None to signal skip.
        eer_fp32 = float("nan")
        eer_int8 = float("nan")

    return {
        "cos_mean": cos_mean,
        "cos_p1": cos_p1,
        "fp32_eer": eer_fp32 * 100,
        "int8_eer": eer_int8 * 100,
        "eer_delta": (eer_int8 - eer_fp32) * 100,
        "n_pairs": len(labels),
    }


def _audio_to_input(audio: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert (T,) audio to embedder input. fbank for 80-mel embedders, raw for 1-channel."""
    import librosa

    if shape[1] == 1:
        target_t = shape[-1]
        if audio.shape[0] < target_t:
            audio = np.concatenate(
                [audio, np.zeros(target_t - audio.shape[0], dtype=np.float32)]
            )
        return audio[:target_t].astype(np.float32).reshape(*shape)

    target_t = shape[-1]
    n_mels = shape[-2]
    if audio.shape[0] < 16000:
        audio = np.concatenate([audio, np.zeros(16000 - audio.shape[0])])
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        fmin=20.0,
        fmax=7600.0,
    )
    log_mel = np.log(mel + 1e-6)
    if log_mel.shape[1] < target_t:
        pad = np.zeros((n_mels, target_t - log_mel.shape[1]), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    log_mel = log_mel[:, :target_t]
    return log_mel.reshape(*shape).astype(np.float32)


def _embed_pair(
    sess: ort.InferenceSession,
    in_name: str,
    a: Path,
    b: Path,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    a_audio, _ = librosa.load(str(a), sr=16000, mono=True, duration=3.0)
    b_audio, _ = librosa.load(str(b), sr=16000, mono=True, duration=3.0)
    a_in = _audio_to_input(a_audio, shape)
    b_in = _audio_to_input(b_audio, shape)
    a_emb = sess.run(None, {in_name: a_in})[0].flatten()
    b_emb = sess.run(None, {in_name: b_in})[0].flatten()
    return a_emb, b_emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=1)


def _read_trials(path: Path) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        out.append((int(parts[0]), parts[1], parts[2]))
    return out


def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def _render_report(kind: str, results: dict[str, Any], budgets: dict[str, Any], ok: bool) -> str:
    status = "PASS" if ok else "FAIL"
    lines = [
        f"# INT8 validation report — {kind}",
        "",
        f"**Status:** {status}",
        f"**Calibration:** voxconverse_dev_500_samples (seed 42)",
        "",
        "## Numbers",
        "",
    ]
    for k, v in results.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Budgets")
    for k, v in budgets.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True, type=Path)
    p.add_argument("--int8", required=True, type=Path)
    p.add_argument("--kind", required=True, choices=["powerset", "embedder"])
    p.add_argument("--hold-out", type=Path, help="VoxConverse-dev audio dir (powerset)")
    p.add_argument("--hold-out-rttm", type=Path, help="VoxConverse-dev rttm dir (powerset)")
    p.add_argument("--voxceleb-audio", type=Path, help="VoxCeleb1 wav dir (embedder)")
    p.add_argument("--voxceleb-trials", type=Path, help="VoxCeleb1 veri_test.txt path (embedder)")
    p.add_argument(
        "--embed-input-shape",
        default="1,80,300",
        help="comma-separated shape for embedder input",
    )
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args(argv)

    args.report.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "powerset":
        if not (args.hold_out and args.hold_out_rttm):
            return _die("--hold-out and --hold-out-rttm required for kind=powerset")
        results = _powerset_compare(args.fp32, args.int8, args.hold_out, args.hold_out_rttm)
        budgets = BUDGETS["powerset"]
        ok = (
            results["der_delta"] <= budgets["der_delta_max"]
            and results["kl_max"] <= budgets["kl_max"]
        )
    else:
        if not (args.voxceleb_audio and args.voxceleb_trials and args.hold_out):
            return _die(
                "--voxceleb-audio, --voxceleb-trials, and --hold-out required for kind=embedder"
            )
        shape = tuple(int(x) for x in args.embed_input_shape.split(","))
        results = _embedder_compare(
            args.fp32,
            args.int8,
            args.voxceleb_audio,
            args.voxceleb_trials,
            args.hold_out,
            shape,
        )
        budgets = BUDGETS["embedder"]
        ok = (
            results["eer_delta"] <= budgets["eer_delta_max"]
            and results["cos_mean"] >= budgets["cosine_mean_min"]
            and results["cos_p1"] >= budgets["cosine_p1_min"]
        )

    report = _render_report(args.kind, results, budgets, ok)
    args.report.write_text(report)
    print(report)
    return 0 if ok else 1


def _die(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
