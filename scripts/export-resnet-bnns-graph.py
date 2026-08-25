#!/usr/bin/env python3
"""Build a Core ML mlprogram of the shipping WeSpeaker ResNet34 (INT8 folded to FP32)."""

from __future__ import annotations

import os
import sys

import coremltools as ct
import numpy as np
import onnx
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX = os.path.join(ROOT, "models/int8/resnet34_int8.onnx")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/int8/resnet34_bnns.mlpackage")

CONVS = [
    ("onnx::Conv_367", "onnx::Conv_368", 32, 1, 3, 1),  # stem
    ("onnx::Conv_370", "onnx::Conv_371", 32, 32, 3, 1),
    ("onnx::Conv_373", "onnx::Conv_374", 32, 32, 3, 1),
    ("onnx::Conv_376", "onnx::Conv_377", 32, 32, 3, 1),
    ("onnx::Conv_379", "onnx::Conv_380", 32, 32, 3, 1),
    ("onnx::Conv_382", "onnx::Conv_383", 32, 32, 3, 1),
    ("onnx::Conv_385", "onnx::Conv_386", 32, 32, 3, 1),
    ("onnx::Conv_388", "onnx::Conv_389", 64, 32, 3, 2),
    ("onnx::Conv_391", "onnx::Conv_392", 64, 64, 3, 1),
    ("onnx::Conv_394", "onnx::Conv_395", 64, 32, 1, 2),
    ("onnx::Conv_397", "onnx::Conv_398", 64, 64, 3, 1),
    ("onnx::Conv_400", "onnx::Conv_401", 64, 64, 3, 1),
    ("onnx::Conv_403", "onnx::Conv_404", 64, 64, 3, 1),
    ("onnx::Conv_406", "onnx::Conv_407", 64, 64, 3, 1),
    ("onnx::Conv_409", "onnx::Conv_410", 64, 64, 3, 1),
    ("onnx::Conv_412", "onnx::Conv_413", 64, 64, 3, 1),
    ("onnx::Conv_415", "onnx::Conv_416", 128, 64, 3, 2),
    ("onnx::Conv_418", "onnx::Conv_419", 128, 128, 3, 1),
    ("onnx::Conv_421", "onnx::Conv_422", 128, 64, 1, 2),
    ("onnx::Conv_424", "onnx::Conv_425", 128, 128, 3, 1),
    ("onnx::Conv_427", "onnx::Conv_428", 128, 128, 3, 1),
    ("onnx::Conv_430", "onnx::Conv_431", 128, 128, 3, 1),
    ("onnx::Conv_433", "onnx::Conv_434", 128, 128, 3, 1),
    ("onnx::Conv_436", "onnx::Conv_437", 128, 128, 3, 1),
    ("onnx::Conv_439", "onnx::Conv_440", 128, 128, 3, 1),
    ("onnx::Conv_442", "onnx::Conv_443", 128, 128, 3, 1),
    ("onnx::Conv_445", "onnx::Conv_446", 128, 128, 3, 1),
    ("onnx::Conv_448", "onnx::Conv_449", 128, 128, 3, 1),
    ("onnx::Conv_451", "onnx::Conv_452", 128, 128, 3, 1),
    ("onnx::Conv_454", "onnx::Conv_455", 256, 128, 3, 2),
    ("onnx::Conv_457", "onnx::Conv_458", 256, 256, 3, 1),
    ("onnx::Conv_460", "onnx::Conv_461", 256, 128, 1, 2),
    ("onnx::Conv_463", "onnx::Conv_464", 256, 256, 3, 1),
    ("onnx::Conv_466", "onnx::Conv_467", 256, 256, 3, 1),
    ("onnx::Conv_469", "onnx::Conv_470", 256, 256, 3, 1),
    ("onnx::Conv_472", "onnx::Conv_473", 256, 256, 3, 1),
]


def _np(t: onnx.TensorProto) -> np.ndarray:
    return onnx.numpy_helper.to_array(t)


def _dequant(init: dict, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name in init:
        return _np(init[name]).astype(np.float32).reshape(shape)
    qn = name + "_quantized"
    # weight: name_scale; bias: name_quantized_scale
    for sn in (name + "_scale", name + "_quantized_scale"):
        if qn in init and sn in init:
            q = _np(init[qn]).astype(np.float32)
            scale = _np(init[sn]).astype(np.float32)
            if scale.size == shape[0]:
                q = q.reshape(shape[0], -1) * scale.reshape(shape[0], 1)
                return q.reshape(shape)
            return (q * float(np.reshape(scale, -1)[0])).reshape(shape)
    raise KeyError(name)


def load_folded(path: str) -> dict[str, np.ndarray]:
    m = onnx.load(path)
    init = {t.name: t for t in m.graph.initializer}
    out: dict[str, np.ndarray] = {}
    for wname, bname, oc, ic, k, _s in CONVS:
        out[wname] = np.ascontiguousarray(_dequant(init, wname, (oc, ic, k, k)))
        out[bname] = np.ascontiguousarray(_dequant(init, bname, (oc,)))
    return out


def conv(x, w, b, stride: int, k: int):
    if k == 3:
        return mb.conv(
            x=x,
            weight=w,
            bias=b,
            strides=[stride, stride],
            pad_type="same",
            dilations=[1, 1],
        )
    return mb.conv(
        x=x,
        weight=w,
        bias=b,
        strides=[stride, stride],
        pad_type="valid",
        dilations=[1, 1],
    )


def identity(x, it, w):
    w1, b1, _o, _i, k1, s1 = next(it)
    w2, b2, _o, _i, k2, s2 = next(it)
    y = mb.relu(x=conv(x, w[w1], w[b1], s1, k1))
    y = conv(y, w[w2], w[b2], s2, k2)
    return mb.relu(x=mb.add(x=x, y=y))


def down_block(x, it, w):
    w1, b1, _o, _i, k1, s1 = next(it)
    w2, b2, _o, _i, k2, s2 = next(it)
    wd, bd, _o, _i, kd, sd = next(it)
    y = mb.relu(x=conv(x, w[w1], w[b1], s1, k1))
    y = conv(y, w[w2], w[b2], s2, k2)
    skip = conv(x, w[wd], w[bd], sd, kd)
    return mb.relu(x=mb.add(x=skip, y=y))


def main() -> None:
    weights = load_folded(ONNX)
    print("loaded", len(weights) // 2, "convs", flush=True)
    sym_t = get_new_symbol()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 80, sym_t), dtype=types.fp32)])
    def resnet(x):
        it = iter(CONVS)
        wn, bn, _o, _i, k, s = next(it)
        x = mb.relu(x=conv(x, weights[wn], weights[bn], s, k))
        for _ in range(3):
            x = identity(x, it, weights)
        x = down_block(x, it, weights)
        for _ in range(3):
            x = identity(x, it, weights)
        x = down_block(x, it, weights)
        for _ in range(5):
            x = identity(x, it, weights)
        x = down_block(x, it, weights)
        for _ in range(2):
            x = identity(x, it, weights)
        return x

    print("listed convs", len(CONVS))

    ml = ct.convert(
        resnet,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name="x",
                shape=(1, 1, 80, ct.RangeDim(8, 8192, default=80)),
            )
        ],
    )
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    ml.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
