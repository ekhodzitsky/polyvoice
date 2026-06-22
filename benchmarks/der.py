"""Single cross-engine Diarization Error Rate (DER) scorer.

Every engine in this benchmark emits an RTTM hypothesis; this module scores all
of them through *one* implementation, at one disclosed collar, so the numbers are
like-for-like. DER is meaningless without a stated collar — every result this
module produces carries the collar it was scored at.

The computation is the canonical NIST `md-eval` frame model (10 ms frames):

    per scored frame, with Nref active reference speakers, Nsys active system
    speakers (after the optimal 1:1 speaker map), and Ncorrect = reference
    speakers whose mapped system speaker is also active:

        miss       += max(0, Nref - Nsys)
        false_alarm+= max(0, Nsys - Nref)
        confusion  += min(Nref, Nsys) - Ncorrect
        scored_ref += Nref

    DER = (miss + false_alarm + confusion) / scored_ref

Overlap is scored by default (a frame may have >1 active speaker), matching the
strict protocol pyannote 3.1 reports against; pass ``skip_overlap=True`` for the
overlap-excluded variant. The optimal speaker map maximizes matched frames over
the *scored* region (Hungarian assignment), matching `md-eval` and polyvoice's
own Rust DER. These numbers have been cross-checked against the polyvoice
`polyvoice-bench` Rust scorer on the shipped splits.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field

FRAME = 0.010  # 10 ms scoring frame


# --------------------------------------------------------------------------- #
# RTTM I/O
# --------------------------------------------------------------------------- #
def parse_rttm(path: str) -> list[tuple[float, float, str]]:
    """Return SPEAKER turns as (start, end, speaker) from an RTTM file."""
    turns: list[tuple[float, float, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p or p[0] != "SPEAKER":
                continue
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            if dur > 0:
                turns.append((start, start + dur, spk))
    return turns


def _basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- #
# Hungarian assignment (max-weight; self-contained, no scipy required)
# --------------------------------------------------------------------------- #
def _max_weight_assignment(weight: list[list[int]]) -> list[tuple[int, int]]:
    """Return (row, col) pairs maximizing total weight. Tries scipy, else O(n^3)."""
    if not weight or not weight[0]:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        w = np.array(weight, dtype=float)
        r, c = linear_sum_assignment(w, maximize=True)
        return [(int(i), int(j)) for i, j in zip(r, c)]
    except Exception:
        pass
    # Pad to square and run a compact Hungarian on a cost matrix.
    n_rows, n_cols = len(weight), len(weight[0])
    n = max(n_rows, n_cols)
    big = max((max(row) for row in weight), default=0) + 1
    cost = [[big - (weight[i][j] if i < n_rows and j < n_cols else 0)
             for j in range(n)] for i in range(n)]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    INF = float("inf")
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = []
    for j in range(1, n + 1):
        i = p[j]
        if 1 <= i <= n_rows and 1 <= j <= n_cols:
            out.append((i - 1, j - 1))
    return out


# --------------------------------------------------------------------------- #
# Per-file scoring
# --------------------------------------------------------------------------- #
@dataclass
class FileScore:
    name: str
    scored_ref: float = 0.0   # seconds of reference speech in the scored region
    miss: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    ref_speakers: int = 0
    hyp_speakers: int = 0

    @property
    def der(self) -> float:
        return 100.0 * (self.miss + self.false_alarm + self.confusion) / self.scored_ref if self.scored_ref > 0 else 0.0


def _frame_sets(turns, n_frames, spk_index):
    """active[f] = set of speaker indices active in frame f."""
    active = [set() for _ in range(n_frames)]
    for start, end, spk in turns:
        s = max(0, int(round(start / FRAME)))
        e = min(n_frames, int(round(end / FRAME)))
        idx = spk_index[spk]
        for f in range(s, e):
            active[f].add(idx)
    return active


def score_file(ref_turns, hyp_turns, collar: float = 0.25, skip_overlap: bool = False) -> FileScore:
    """Score one file. ``collar`` is the half-collar in seconds (NIST forgiveness)."""
    end_time = max([e for _, e, _ in ref_turns] + [e for _, e, _ in hyp_turns] + [0.0])
    n = int(round(end_time / FRAME)) + 1

    ref_spk = {s: i for i, s in enumerate({t[2] for t in ref_turns})}
    hyp_spk = {s: i for i, s in enumerate({t[2] for t in hyp_turns})}
    ref_active = _frame_sets(ref_turns, n, ref_spk)
    hyp_active = _frame_sets(hyp_turns, n, hyp_spk)

    # Collar: exclude frames within +/- collar of every reference boundary.
    scored = [True] * n
    if collar > 0:
        half = int(round(collar / FRAME))
        for start, end, _ in ref_turns:
            for t in (start, end):
                c = int(round(t / FRAME))
                for f in range(max(0, c - half), min(n, c + half + 1)):
                    scored[f] = False
    if skip_overlap:
        for f in range(n):
            if len(ref_active[f]) > 1:
                scored[f] = False

    # Optimal ref->hyp map maximizing co-occurring scored frames.
    cooc = [[0] * len(hyp_spk) for _ in range(len(ref_spk))]
    for f in range(n):
        if not scored[f]:
            continue
        for ri in ref_active[f]:
            for hj in hyp_active[f]:
                cooc[ri][hj] += 1
    mapping = dict(_max_weight_assignment(cooc)) if ref_spk and hyp_spk else {}

    fs = FileScore(name="", ref_speakers=len(ref_spk), hyp_speakers=len(hyp_spk))
    for f in range(n):
        if not scored[f]:
            continue
        nref = len(ref_active[f])
        nsys = len(hyp_active[f])
        if nref == 0 and nsys == 0:
            continue
        ncorrect = sum(1 for ri in ref_active[f]
                       if ri in mapping and mapping[ri] in hyp_active[f])
        fs.scored_ref += nref * FRAME
        fs.miss += max(0, nref - nsys) * FRAME
        fs.false_alarm += max(0, nsys - nref) * FRAME
        fs.confusion += (min(nref, nsys) - ncorrect) * FRAME
    return fs


# --------------------------------------------------------------------------- #
# Dataset aggregation + bootstrap CI
# --------------------------------------------------------------------------- #
@dataclass
class DatasetScore:
    collar: float
    skip_overlap: bool
    files: list[FileScore] = field(default_factory=list)

    def _agg(self):
        sr = sum(f.scored_ref for f in self.files)
        ms = sum(f.miss for f in self.files)
        fa = sum(f.false_alarm for f in self.files)
        cf = sum(f.confusion for f in self.files)
        return sr, ms, fa, cf

    @property
    def der_micro(self) -> float:
        sr, ms, fa, cf = self._agg()
        return 100.0 * (ms + fa + cf) / sr if sr > 0 else 0.0

    @property
    def der_macro(self) -> float:
        return sum(f.der for f in self.files) / len(self.files) if self.files else 0.0

    def decomposition_micro(self) -> dict:
        sr, ms, fa, cf = self._agg()
        return {
            "miss": 100.0 * ms / sr if sr else 0.0,
            "false_alarm": 100.0 * fa / sr if sr else 0.0,
            "confusion": 100.0 * cf / sr if sr else 0.0,
        }

    def speaker_count_accuracy(self) -> dict:
        exact = pm1 = off2 = 0
        for f in self.files:
            d = abs(f.hyp_speakers - f.ref_speakers)
            if d == 0:
                exact += 1
            elif d == 1:
                pm1 += 1
            else:
                off2 += 1
        return {"exact": exact, "plus_minus_1": pm1, "off_by_2_or_more": off2,
                "files": len(self.files)}

    def bootstrap_ci(self, iterations: int = 1000) -> tuple[float, float]:
        """95% bootstrap CI of the micro DER, resampling files with replacement.

        Deterministic LCG (mirrors the gigastt harness) so CIs are reproducible.
        """
        items = [(f.scored_ref, f.miss + f.false_alarm + f.confusion) for f in self.files]
        n = len(items)
        if n == 0:
            return (0.0, 0.0)
        rng = 123456789
        mask = (1 << 64) - 1
        ders = []
        for _ in range(iterations):
            tot_ref = tot_err = 0.0
            for _ in range(n):
                rng = (rng * 6364136223846793005 + 1) & mask
                idx = (rng >> 32) % n
                tot_ref += items[idx][0]
                tot_err += items[idx][1]
            ders.append(100.0 * tot_err / tot_ref if tot_ref > 0 else 0.0)
        ders.sort()
        return (ders[(iterations * 25) // 1000], ders[(iterations * 975) // 1000])


def score_dataset(ref_dir: str, hyp: dict[str, str] | str,
                  collar: float = 0.25, skip_overlap: bool = False) -> DatasetScore:
    """Score a directory of reference RTTMs against per-file hypothesis RTTMs.

    ``ref_dir`` holds ``<name>.rttm`` references (or a single multi-file RTTM).
    ``hyp`` is either a directory of ``<name>.rttm`` hypotheses or a dict
    ``{name: rttm_path}``. Only files present in both are scored.
    """
    refs = _load_turns_index(ref_dir)
    if isinstance(hyp, dict):
        hyp_index = {name: parse_rttm(p) for name, p in hyp.items()}
    else:
        hyp_index = _load_turns_index(hyp)
    ds = DatasetScore(collar=collar, skip_overlap=skip_overlap)
    for name in sorted(refs):
        if name not in hyp_index:
            continue
        fs = score_file(refs[name], hyp_index[name], collar, skip_overlap)
        fs.name = name
        ds.files.append(fs)
    return ds


def _load_turns_index(path: str) -> dict[str, list[tuple[float, float, str]]]:
    """Index .rttm under ``path`` by basename, returning {name: turns}.

    A single combined RTTM (turns tagged by file-id in column 2) is split per id;
    a directory of per-file RTTMs is keyed by file basename.
    """
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.rttm"), recursive=True)
        if len(files) == 1:
            return _split_combined_rttm(files[0])
        return {_basename(f): parse_rttm(f) for f in files}
    return _split_combined_rttm(path)


def _split_combined_rttm(path: str) -> dict[str, list[tuple[float, float, str]]]:
    by_file: dict[str, list[tuple[float, float, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p or p[0] != "SPEAKER":
                continue
            fid, start, dur, spk = p[1], float(p[3]), float(p[4]), p[7]
            if dur > 0:
                by_file.setdefault(fid, []).append((start, start + dur, spk))
    return by_file


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Score a hypothesis RTTM against a reference RTTM (single DER scorer).")
    ap.add_argument("reference", help="reference .rttm (or directory)")
    ap.add_argument("hypothesis", help="hypothesis .rttm (or directory)")
    ap.add_argument("--collar", type=float, default=0.25, help="half-collar seconds (default 0.25; use 0 for no-collar)")
    ap.add_argument("--skip-overlap", action="store_true", help="exclude overlapped reference frames")
    args = ap.parse_args()

    if os.path.isdir(args.reference):
        ds = score_dataset(args.reference, args.hypothesis, args.collar, args.skip_overlap)
        lo, hi = ds.bootstrap_ci()
        dec = ds.decomposition_micro()
        print(f"files={len(ds.files)} collar={args.collar} skip_overlap={args.skip_overlap}")
        print(f"DER micro={ds.der_micro:.2f}% macro={ds.der_macro:.2f}%  CI95=[{lo:.2f},{hi:.2f}]")
        print(f"  miss={dec['miss']:.2f}% fa={dec['false_alarm']:.2f}% confusion={dec['confusion']:.2f}%")
        print(f"  speaker-count={ds.speaker_count_accuracy()}")
    else:
        fs = score_file(parse_rttm(args.reference), parse_rttm(args.hypothesis), args.collar, args.skip_overlap)
        print(f"DER={fs.der:.2f}% miss={100*fs.miss/fs.scored_ref:.2f}% "
              f"fa={100*fs.false_alarm/fs.scored_ref:.2f}% "
              f"confusion={100*fs.confusion/fs.scored_ref:.2f}% "
              f"(ref_spk={fs.ref_speakers} hyp_spk={fs.hyp_speakers})")


if __name__ == "__main__":
    main()
