#!/usr/bin/env python3
"""Build the weights-only INT8 Parakeet TDT encoder (recommended ASR export).

The FP32 Parakeet encoder is 2.55 GB. This script emits a weights-only INT8
encoder (~670 MB): MatMul weights via blockwise MatMulNBits (int8, 128-wide
blocks, symmetric) and Conv weights via per-output-channel int8
DequantizeLinear. Activations stay FP32 end-to-end. Full INT8 activation
quantization was measured to drop words on far-field audio, while weight-only
quantization is word-loss-neutral on the parity fixtures (2/13312 boundary
words); numbers and protocol: docs/BENCHMARKS.md.

Usage:
  pip install "onnxruntime>=1.30" onnx onnx-ir numpy
  python3 scripts/quantize-parakeet-encoder.py <fp32-model-dir> <out-dir>

<fp32-model-dir> must contain encoder-model.onnx + encoder-model.onnx.data
(from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).
<out-dir> receives encoder-model.int8.onnx + encoder-model.int8.onnx.data;
decoder_joint-model.onnx and vocab.txt are copied when present, so the output
is a drop-in --asr-model directory (the loader picks *.int8.onnx when the FP32
encoder file is absent).

Peak RSS ~3.2 GiB (the FP32 graph is held in memory twice, briefly). On a
shared machine wrap with a memory limit, e.g.:
  systemd-run --user --scope -p MemoryMax=8G python3 scripts/quantize-parakeet-encoder.py ...

Runtime requirement for the produced model: ONNX Runtime >= 1.22 CPU
(MatMulNBits int8 kernels); the ort version pinned by polyvoice-asr already
includes them.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def quantize_matmuls_int8(src: Path, dst: Path, block_size: int) -> None:
    """MatMul weights -> MatMulNBits int8 (blockwise, symmetric).

    accuracy_level stays unset so compute runs in FP32 and activations are
    never quantized. Weightless MatMuls (attention scores) are skipped by the
    quantizer itself.
    """
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    log(f"stage=matmul_nbits src={src} bits=8 block={block_size} sym=True")
    quant = MatMulNBitsQuantizer(
        str(src),
        bits=8,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=None,
    )
    quant.process()
    dst.parent.mkdir(parents=True, exist_ok=True)
    quant.model.save_model_to_file(str(dst), True)
    log(f"stage=matmul_nbits_done size={dst.stat().st_size}")


def quantize_convs_int8(src: Path, dst: Path) -> None:
    """Conv weight initializers -> int8 per-output-channel symmetric + DQ node."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    log(f"stage=conv_int8 src={src}")
    model = onnx.load(str(src))
    graph = model.graph
    init = {i.name: i for i in graph.initializer}

    new_nodes = []
    drop_inits = set()
    n_conv = 0
    for node in graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            t = init.get(node.input[1])
            if t is not None and t.data_type == TensorProto.FLOAT:
                w = numpy_helper.to_array(t)  # [M, C, kH, kW]
                m = w.shape[0]
                flat = w.reshape(m, -1)
                scale = np.abs(flat).max(axis=1) / 127.0
                scale = np.maximum(scale, 1e-12).astype(np.float32)
                q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
                w_name = node.input[1]
                graph.initializer.extend(
                    [
                        numpy_helper.from_array(q, name=w_name + "_q"),
                        numpy_helper.from_array(scale, name=w_name + "_scale"),
                        numpy_helper.from_array(np.zeros(m, dtype=np.int8), name=w_name + "_zp"),
                    ]
                )
                new_nodes.append(
                    helper.make_node(
                        "DequantizeLinear",
                        [w_name + "_q", w_name + "_scale", w_name + "_zp"],
                        [w_name + "_dq"],
                        name=node.name + "/w_dq",
                        axis=0,
                    )
                )
                node.input[1] = w_name + "_dq"
                drop_inits.add(w_name)
                n_conv += 1
                del w, flat, q
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    used = set()
    for node in graph.node:
        used.update(node.input)
    used.update(go.name for go in graph.output)
    kept_init = [i for i in graph.initializer if i.name in used and i.name not in drop_inits]
    del graph.initializer[:]
    graph.initializer.extend(kept_init)
    log(f"stage=conv_int8_done convs={n_conv}")

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=dst.name + ".data",
    )
    log(f"stage=conv_int8_saved size={dst.stat().st_size}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src_dir", type=Path, help="directory with the FP32 encoder-model.onnx[.data]")
    p.add_argument("dst_dir", type=Path, help="output directory for the weights-only INT8 model")
    p.add_argument("--block-size", type=int, default=128, help="MatMulNBits block size (pow2 >= 16)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = p.parse_args()

    src = args.src_dir / "encoder-model.onnx"
    dst = args.dst_dir / "encoder-model.int8.onnx"
    if not src.is_file():
        print(f"error: {src} not found", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"error: {dst} exists (use --force)", file=sys.stderr)
        return 1

    tmp = args.dst_dir / ".matmul-nbits.tmp.onnx"
    try:
        quantize_matmuls_int8(src, tmp, args.block_size)
        quantize_convs_int8(tmp, dst)
    finally:
        for f in (tmp, tmp.with_suffix(tmp.suffix + ".data")):
            f.unlink(missing_ok=True)

    copied = []
    for extra in ("decoder_joint-model.onnx", "vocab.txt"):
        f = args.src_dir / extra
        if f.is_file():
            shutil.copy2(f, args.dst_dir / extra)
            copied.append(extra)
    total = sum(f.stat().st_size for f in args.dst_dir.iterdir() if f.is_file())
    log(f"DONE {args.dst_dir} total={total} B ({total / 1e9:.3f} GB); copied: {', '.join(copied) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
