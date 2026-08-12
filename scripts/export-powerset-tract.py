#!/usr/bin/env python3
"""Rewrite powerset ONNX so pure-Rust tract can load and run it.

Transforms (preserves numerical behaviour under onnxruntime):
  1. Inline identical-branch ``If`` (export artifact: then/else are the same Conv).
  2. Expand ``InstanceNormalization`` into ReduceMean/Sub/Mul/Sqrt/Div + affine.

The rewritten graph keeps dynamic ``[N,1,T]`` for ort. Tract still needs a
concrete T at session build (see ``try_optimize_with_concrete_nct`` in
``src/onnx/tract_session.rs``): product window **10 s @ 16 kHz = 160000**.

Usage:
  python3 scripts/export-powerset-tract.py \\
      --input models/powerset_fp32.onnx \\
      --output models/powerset_fp32_tract.onnx

  # optional ort self-check
  python3 scripts/export-powerset-tract.py --input models/powerset_fp32.onnx \\
      --output models/powerset_fp32_tract.onnx --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _inline_identical_if(graph: onnx.GraphProto) -> int:
    new_nodes: list[onnx.NodeProto] = []
    n_inlined = 0
    for n in graph.node:
        if n.op_type != "If":
            new_nodes.append(n)
            continue
        then_g = next(a.g for a in n.attribute if a.name == "then_branch")
        else_g = next(a.g for a in n.attribute if a.name == "else_branch")

        def sig(g: onnx.GraphProto) -> list:
            return [
                (
                    x.op_type,
                    tuple(x.input),
                    tuple(
                        (a.name, list(a.ints) if a.ints else a.i if a.type == 2 else a.f)
                        for a in x.attribute
                        if a.name
                        in ("kernel_shape", "pads", "dilations", "strides", "group")
                    ),
                )
                for x in g.node
            ]

        if sig(then_g) != sig(else_g) or len(then_g.node) != 1:
            new_nodes.append(n)
            continue
        sn = then_g.node[0]
        new_n = helper.make_node(
            sn.op_type, list(sn.input), list(n.output), name=f"{sn.name}_inlined"
        )
        del new_n.attribute[:]
        new_n.attribute.extend(sn.attribute)
        new_nodes.append(new_n)
        n_inlined += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    return n_inlined


def _expand_instance_norm(graph: onnx.GraphProto) -> int:
    """Replace InstanceNormalization with rank-3 N,C,T ReduceMean formula."""
    new_nodes: list[onnx.NodeProto] = []
    extras: list[onnx.TensorProto] = []
    n_exp = 0

    def const(name: str, arr: np.ndarray) -> str:
        extras.append(numpy_helper.from_array(arr, name=name))
        return name

    inits = {i.name: i for i in graph.initializer}
    for n in graph.node:
        if n.op_type != "InstanceNormalization":
            new_nodes.append(n)
            continue
        x, scale, bias = n.input[0], n.input[1], n.input[2]
        y = n.output[0]
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        pref = "rw" + str(abs(hash(y)))[:8]
        # mean/var over time axis only (N,C,T)
        mean = helper.make_node(
            "ReduceMean", [x], [f"{pref}_mean"], name=f"{pref}_rm", axes=[2], keepdims=1
        )
        xc = helper.make_node("Sub", [x, f"{pref}_mean"], [f"{pref}_xc"], name=f"{pref}_sub")
        xc2 = helper.make_node(
            "Mul", [f"{pref}_xc", f"{pref}_xc"], [f"{pref}_xc2"], name=f"{pref}_sq"
        )
        var = helper.make_node(
            "ReduceMean",
            [f"{pref}_xc2"],
            [f"{pref}_var"],
            name=f"{pref}_rv",
            axes=[2],
            keepdims=1,
        )
        eps_n = const(f"{pref}_eps", np.array(eps, dtype=np.float32))
        vare = helper.make_node(
            "Add", [f"{pref}_var", eps_n], [f"{pref}_ve"], name=f"{pref}_adde"
        )
        std = helper.make_node("Sqrt", [f"{pref}_ve"], [f"{pref}_std"], name=f"{pref}_sqrt")
        norm = helper.make_node(
            "Div", [f"{pref}_xc", f"{pref}_std"], [f"{pref}_norm"], name=f"{pref}_div"
        )
        if scale not in inits:
            raise RuntimeError(f"scale initializer {scale!r} missing for {n.name}")
        c = int(np.prod(numpy_helper.to_array(inits[scale]).shape))
        sh = const(f"{pref}_sh", np.array([1, c, 1], dtype=np.int64))
        sc_r = helper.make_node("Reshape", [scale, sh], [f"{pref}_sc"], name=f"{pref}_rsc")
        bi_r = helper.make_node("Reshape", [bias, sh], [f"{pref}_bi"], name=f"{pref}_rbi")
        sca = helper.make_node(
            "Mul", [f"{pref}_norm", f"{pref}_sc"], [f"{pref}_sca"], name=f"{pref}_mul"
        )
        out = helper.make_node("Add", [f"{pref}_sca", f"{pref}_bi"], [y], name=f"{pref}_add")
        new_nodes.extend([mean, xc, xc2, var, vare, std, norm, sc_r, bi_r, sca, out])
        n_exp += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(extras)
    return n_exp


def rewrite(path_in: Path, path_out: Path) -> dict:
    m = onnx.load(str(path_in))
    n_if = _inline_identical_if(m.graph)
    n_in = _expand_instance_norm(m.graph)
    del m.graph.value_info[:]
    # Keep dynamic [N,1,T] for ort; tract binds concrete T at load.
    for i in m.graph.input:
        if i.name == "x":
            t = i.type.tensor_type
            del t.shape.dim[:]
            for val, param in [(0, "N"), (1, None), (0, "T")]:
                d = t.shape.dim.add()
                if param is not None:
                    d.dim_param = param
                else:
                    d.dim_value = val
    onnx.checker.check_model(m)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path_out))
    return {"if_inlined": n_if, "instance_norm_expanded": n_in, "nodes": len(m.graph.node)}


def verify(path_src: Path, path_rw: Path, t: int = 160_000, seed: int = 0) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((1, 1, t)) * 0.05).astype(np.float32)
    s1 = ort.InferenceSession(str(path_src), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(path_rw), providers=["CPUExecutionProvider"])
    name = s1.get_inputs()[0].name
    y1 = s1.run(None, {name: x})[0]
    y2 = s2.run(None, {name: x})[0]
    if y1.shape != y2.shape:
        raise RuntimeError(f"shape mismatch {y1.shape} vs {y2.shape}")
    return float(np.max(np.abs(y1 - y2)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("models/powerset_fp32.onnx"))
    p.add_argument("--output", type=Path, default=Path("models/powerset_fp32_tract.onnx"))
    p.add_argument("--verify", action="store_true", help="ort max-abs check @ T=160000")
    args = p.parse_args()
    if not args.input.is_file():
        print(f"FATAL: missing {args.input}", file=sys.stderr)
        return 1
    stats = rewrite(args.input, args.output)
    print(f"wrote {args.output} nodes={stats['nodes']} "
          f"if_inlined={stats['if_inlined']} instance_norm={stats['instance_norm_expanded']}")
    if args.verify:
        diff = verify(args.input, args.output)
        print(f"ort max-abs diff @ T=160000: {diff:.6e}")
        if diff > 1e-3:
            print("WARN: diff exceeds 1e-3", file=sys.stderr)
            return 2
    print("next: cargo test --lib --features onnx,backend-tract "
          "powerset_fp32_tract_friendly -- --nocapture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
