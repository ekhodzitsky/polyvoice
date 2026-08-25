//! Minimal ONNX initializer reader.
//!
//! Walks `ModelProto.graph.initializer` only. Does **not** parse or execute
//! the computation graph.

use crate::error::KernelError;
use std::collections::HashMap;
use std::path::Path;

const WIRE_VARINT: u64 = 0;
const WIRE_64: u64 = 1;
const WIRE_LEN: u64 = 2;
const WIRE_32: u64 = 5;

/// ONNX `TensorProto.data_type` we care about.
const DT_FLOAT: i32 = 1;
const DT_INT8: i32 = 3;
const DT_INT32: i32 = 6;

#[derive(Clone, Debug)]
pub enum OnnxPayload {
    F32(Vec<f32>),
    I8(Vec<i8>),
    I32(Vec<i32>),
}

#[derive(Clone, Debug)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub payload: OnnxPayload,
}

pub fn load_initializers(path: &Path) -> Result<HashMap<String, OnnxTensor>, KernelError> {
    let bytes = std::fs::read(path).map_err(|e| KernelError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    parse_initializers(&bytes)
}

fn parse_initializers(bytes: &[u8]) -> Result<HashMap<String, OnnxTensor>, KernelError> {
    let mut pos = 0;
    let mut graph = None;
    while pos < bytes.len() {
        let (key, p) = read_key(bytes, pos)?;
        pos = p;
        let field = key >> 3;
        let wire = key & 7;
        if field == 7 && wire == WIRE_LEN {
            let (blob, p) = read_len_blob(bytes, pos)?;
            pos = p;
            graph = Some(blob);
        } else {
            pos = skip_value(bytes, pos, wire)?;
        }
    }
    let graph = graph.ok_or_else(|| KernelError::Model {
        detail: "ModelProto has no graph".into(),
    })?;
    let mut out = HashMap::new();
    let mut pos = 0;
    while pos < graph.len() {
        let (key, p) = read_key(graph, pos)?;
        pos = p;
        let field = key >> 3;
        let wire = key & 7;
        if field == 5 && wire == WIRE_LEN {
            let (blob, p) = read_len_blob(graph, pos)?;
            pos = p;
            if let Some(t) = parse_tensor_proto(blob)? {
                out.insert(t.name.clone(), t);
            }
        } else {
            pos = skip_value(graph, pos, wire)?;
        }
    }
    if out.is_empty() {
        return Err(KernelError::Model {
            detail: "graph has no initializers".into(),
        });
    }
    Ok(out)
}

fn parse_tensor_proto(bytes: &[u8]) -> Result<Option<OnnxTensor>, KernelError> {
    let mut pos = 0;
    let mut dims: Vec<i64> = Vec::new();
    let mut data_type: i32 = 0;
    let mut name = String::new();
    let mut raw: Option<&[u8]> = None;
    let mut float_data: Vec<f32> = Vec::new();
    let mut int32_data: Vec<i32> = Vec::new();
    while pos < bytes.len() {
        let (key, p) = read_key(bytes, pos)?;
        pos = p;
        let field = key >> 3;
        let wire = key & 7;
        match (field, wire) {
            (1, WIRE_VARINT) => {
                let (v, p) = read_varint(bytes, pos)?;
                pos = p;
                dims.push(v as i64);
            }
            (1, WIRE_LEN) => {
                let (blob, p) = read_len_blob(bytes, pos)?;
                pos = p;
                let mut i = 0;
                while i < blob.len() {
                    let (v, n) = read_varint(blob, i)?;
                    i = n;
                    dims.push(v as i64);
                }
            }
            (2, WIRE_VARINT) => {
                let (v, p) = read_varint(bytes, pos)?;
                pos = p;
                data_type = v as i32;
            }
            (4, WIRE_LEN) => {
                let (blob, p) = read_len_blob(bytes, pos)?;
                pos = p;
                if blob.len() % 4 != 0 {
                    return Err(KernelError::Model {
                        detail: "float_data length not multiple of 4".into(),
                    });
                }
                float_data.extend(
                    blob.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
            }
            (4, WIRE_32) => {
                if pos + 4 > bytes.len() {
                    return Err(KernelError::Model {
                        detail: "truncated float_data".into(),
                    });
                }
                float_data.push(f32::from_le_bytes(
                    bytes[pos..pos + 4].try_into().unwrap_or([0; 4]),
                ));
                pos += 4;
            }
            (5, WIRE_LEN) => {
                let (blob, p) = read_len_blob(bytes, pos)?;
                pos = p;
                int32_data.extend(read_packed_int32(blob)?);
            }
            (5, WIRE_VARINT) => {
                let (v, p) = read_varint(bytes, pos)?;
                pos = p;
                int32_data.push(v as i32);
            }
            (8, WIRE_LEN) => {
                let (blob, p) = read_len_blob(bytes, pos)?;
                pos = p;
                name = String::from_utf8_lossy(blob).into_owned();
            }
            (9, WIRE_LEN) => {
                let (blob, p) = read_len_blob(bytes, pos)?;
                pos = p;
                raw = Some(blob);
            }
            _ => pos = skip_value(bytes, pos, wire)?,
        }
    }
    let dims: Vec<usize> = dims
        .into_iter()
        .map(|d| usize::try_from(d).unwrap_or(0))
        .collect();
    let expected: usize = dims.iter().product();
    let payload = match data_type {
        DT_FLOAT => {
            let data = if let Some(raw) = raw {
                if raw.len() % 4 != 0 {
                    return Err(KernelError::Model {
                        detail: format!("raw_data for {name} not multiple of 4"),
                    });
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                float_data
            };
            OnnxPayload::F32(data)
        }
        DT_INT8 => {
            let data = if let Some(raw) = raw {
                raw.iter().map(|&b| b as i8).collect()
            } else {
                int32_data.into_iter().map(|v| v as i8).collect()
            };
            OnnxPayload::I8(data)
        }
        DT_INT32 => {
            let data = if let Some(raw) = raw {
                if raw.len() % 4 != 0 {
                    return Err(KernelError::Model {
                        detail: format!("raw_data for {name} not multiple of 4"),
                    });
                }
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                int32_data
            };
            OnnxPayload::I32(data)
        }
        _ => return Ok(None),
    };
    let got = match &payload {
        OnnxPayload::F32(v) => v.len(),
        OnnxPayload::I8(v) => v.len(),
        OnnxPayload::I32(v) => v.len(),
    };
    if expected != got {
        return Err(KernelError::Model {
            detail: format!("initializer {name} has {got} values, shape {dims:?} wants {expected}"),
        });
    }
    Ok(Some(OnnxTensor {
        name,
        dims,
        payload,
    }))
}

fn read_packed_int32(blob: &[u8]) -> Result<Vec<i32>, KernelError> {
    let mut i = 0;
    let mut out = Vec::new();
    while i < blob.len() {
        let (v, n) = read_varint(blob, i)?;
        out.push(v as i32);
        i = n;
    }
    Ok(out)
}

/// Float initializer, or QDQ `{name}_quantized` + scale + zero-point.
///
/// INT8 LSTM `W` is stored `[dir, input, 4H]`; the FP32 file is
/// `[dir, 4H, input]`. Last two axes are swapped when that is the only mismatch.
pub fn take_f32(
    init: &HashMap<String, OnnxTensor>,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, KernelError> {
    if let Some(t) = init.get(name)
        && let OnnxPayload::F32(v) = &t.payload
    {
        return coerce_shape(name, &t.dims, v, expected);
    }
    let qname = format!("{name}_quantized");
    let Some(qt) = init.get(&qname) else {
        return Err(KernelError::Weight {
            name: name.to_owned(),
            expected: expected.to_vec(),
            got: Vec::new(),
        });
    };
    let (scale, s_dims) = take_scale(init, name)?;
    let data = match &qt.payload {
        OnnxPayload::I8(q) => {
            let zp = take_zp_i8(init, name, scale.len())?;
            dequant_i8(q, &qt.dims, &scale, &s_dims, &zp)
        }
        OnnxPayload::I32(q) => {
            let zp = take_zp_i32(init, name, scale.len())?;
            dequant_i32(q, &qt.dims, &scale, &s_dims, &zp)
        }
        OnnxPayload::F32(_) => {
            return Err(KernelError::Model {
                detail: format!("{qname} is float, not quantized"),
            });
        }
    };
    coerce_shape(name, &qt.dims, &data, expected)
}

/// Raw INT8 QDQ payload `{name}_quantized` + scale + zp (no dequant).
#[allow(clippy::type_complexity)]
pub fn take_i8_quant(
    init: &HashMap<String, OnnxTensor>,
    name: &str,
    expected: &[usize],
) -> Result<(Vec<i8>, Vec<f32>, Vec<i8>), KernelError> {
    let qname = format!("{name}_quantized");
    let qt = init.get(&qname).ok_or_else(|| KernelError::Weight {
        name: qname.clone(),
        expected: expected.to_vec(),
        got: Vec::new(),
    })?;
    if qt.dims != expected {
        return Err(KernelError::Weight {
            name: qname,
            expected: expected.to_vec(),
            got: qt.dims.clone(),
        });
    }
    let OnnxPayload::I8(q) = &qt.payload else {
        return Err(KernelError::Model {
            detail: format!("{qname} is not INT8"),
        });
    };
    let (scale, _) = take_scale(init, name)?;
    let zp = take_zp_i8(init, name, scale.len())?;
    Ok((q.clone(), scale, zp))
}

fn coerce_shape(
    name: &str,
    dims: &[usize],
    data: &[f32],
    expected: &[usize],
) -> Result<Vec<f32>, KernelError> {
    if dims == expected {
        return Ok(data.to_vec());
    }
    if dims.len() == 3
        && expected.len() == 3
        && dims[0] == expected[0]
        && dims[1] == expected[2]
        && dims[2] == expected[1]
    {
        return Ok(transpose_last2(data, dims[0], dims[1], dims[2]));
    }
    Err(KernelError::Weight {
        name: name.to_owned(),
        expected: expected.to_vec(),
        got: dims.to_vec(),
    })
}

fn take_scale(
    init: &HashMap<String, OnnxTensor>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), KernelError> {
    for key in [format!("{name}_scale"), format!("{name}_quantized_scale")] {
        if let Some(t) = init.get(&key)
            && let OnnxPayload::F32(v) = &t.payload
        {
            return Ok((v.clone(), t.dims.clone()));
        }
    }
    Err(KernelError::Weight {
        name: format!("{name}_scale"),
        expected: Vec::new(),
        got: Vec::new(),
    })
}

fn take_zp_i8(
    init: &HashMap<String, OnnxTensor>,
    name: &str,
    n: usize,
) -> Result<Vec<i8>, KernelError> {
    for key in [
        format!("{name}_zero_point"),
        format!("{name}_quantized_zero_point"),
    ] {
        if let Some(t) = init.get(&key) {
            match &t.payload {
                OnnxPayload::I8(v) => return Ok(v.clone()),
                OnnxPayload::I32(v) => return Ok(v.iter().map(|&x| x as i8).collect()),
                OnnxPayload::F32(_) => {}
            }
        }
    }
    Ok(vec![0; n.max(1)])
}

fn take_zp_i32(
    init: &HashMap<String, OnnxTensor>,
    name: &str,
    n: usize,
) -> Result<Vec<i32>, KernelError> {
    for key in [
        format!("{name}_zero_point"),
        format!("{name}_quantized_zero_point"),
    ] {
        if let Some(t) = init.get(&key) {
            match &t.payload {
                OnnxPayload::I32(v) => return Ok(v.clone()),
                OnnxPayload::I8(v) => return Ok(v.iter().map(|&x| i32::from(x)).collect()),
                OnnxPayload::F32(_) => {}
            }
        }
    }
    Ok(vec![0; n.max(1)])
}

fn dequant_i8(q: &[i8], q_dims: &[usize], scale: &[f32], s_dims: &[usize], zp: &[i8]) -> Vec<f32> {
    let idx = scale_index_fn(q_dims, s_dims);
    q.iter()
        .enumerate()
        .map(|(i, &qv)| {
            let si = idx(i).min(scale.len().saturating_sub(1));
            let z = f32::from(*zp.get(si).unwrap_or(&0));
            let s = scale.get(si).copied().unwrap_or(1.0);
            (f32::from(qv) - z) * s
        })
        .collect()
}

fn dequant_i32(
    q: &[i32],
    q_dims: &[usize],
    scale: &[f32],
    s_dims: &[usize],
    zp: &[i32],
) -> Vec<f32> {
    let idx = scale_index_fn(q_dims, s_dims);
    q.iter()
        .enumerate()
        .map(|(i, &qv)| {
            let si = idx(i).min(scale.len().saturating_sub(1));
            let z = zp.get(si).copied().unwrap_or(0) as f32;
            let s = scale.get(si).copied().unwrap_or(1.0);
            (qv as f32 - z) * s
        })
        .collect()
}

/// Map a linear index in `q_dims` to a linear index in `s_dims`.
///
/// 1-D scales attach to the first matching axis (Conv `axis=0`, else last).
/// Multi-D scales pick a left-greedy subsequence of equal-sized axes
/// (LSTM `[dir, in, 4H]` × `[dir, 4H]`).
fn scale_index_fn(q_dims: &[usize], s_dims: &[usize]) -> impl Fn(usize) -> usize {
    let q_dims = q_dims.to_vec();
    let s_dims = if s_dims.is_empty() {
        vec![1]
    } else {
        s_dims.to_vec()
    };
    let axes: Vec<usize> = if s_dims.len() == 1 {
        let slen = s_dims[0];
        match q_dims.iter().position(|&d| d == slen) {
            Some(ax) => vec![ax],
            None => vec![q_dims.len().saturating_sub(1)],
        }
    } else {
        let mut used = vec![false; q_dims.len()];
        let mut axes = Vec::with_capacity(s_dims.len());
        for &sd in &s_dims {
            if let Some(ax) = q_dims
                .iter()
                .enumerate()
                .position(|(i, &d)| !used[i] && d == sd)
            {
                used[ax] = true;
                axes.push(ax);
            }
        }
        if axes.len() != s_dims.len() {
            axes = (0..s_dims.len().min(q_dims.len())).collect();
        }
        axes
    };
    move |linear: usize| {
        if q_dims.is_empty() {
            return 0;
        }
        let coords = unravel(linear, &q_dims);
        let mut acc = 0;
        for (k, &ax) in axes.iter().enumerate() {
            let c = *coords.get(ax).unwrap_or(&0);
            let stride: usize = s_dims[k + 1..].iter().product();
            acc += c * stride;
        }
        acc
    }
}

fn unravel(mut i: usize, dims: &[usize]) -> Vec<usize> {
    let mut out = vec![0; dims.len()];
    for ax in (0..dims.len()).rev() {
        let d = dims[ax].max(1);
        out[ax] = i % d;
        i /= d;
    }
    out
}

fn transpose_last2(data: &[f32], a: usize, b: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; a.saturating_mul(b).saturating_mul(c)];
    for ia in 0..a {
        for ib in 0..b {
            for ic in 0..c {
                let src = (ia * b + ib) * c + ic;
                let dst = (ia * c + ic) * b + ib;
                out[dst] = data[src];
            }
        }
    }
    out
}

fn read_key(bytes: &[u8], pos: usize) -> Result<(u64, usize), KernelError> {
    read_varint(bytes, pos)
}

fn read_varint(bytes: &[u8], mut pos: usize) -> Result<(u64, usize), KernelError> {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        if pos >= bytes.len() {
            return Err(KernelError::Model {
                detail: "truncated varint".into(),
            });
        }
        let b = bytes[pos];
        pos += 1;
        result |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift > 63 {
            return Err(KernelError::Model {
                detail: "varint overflow".into(),
            });
        }
    }
}

fn read_len_blob(bytes: &[u8], pos: usize) -> Result<(&[u8], usize), KernelError> {
    let (len, pos) = read_varint(bytes, pos)?;
    let end = pos
        .checked_add(len as usize)
        .ok_or_else(|| KernelError::Model {
            detail: "length overflow".into(),
        })?;
    if end > bytes.len() {
        return Err(KernelError::Model {
            detail: "truncated length-delimited field".into(),
        });
    }
    Ok((&bytes[pos..end], end))
}

fn skip_value(bytes: &[u8], pos: usize, wire: u64) -> Result<usize, KernelError> {
    match wire {
        WIRE_VARINT => {
            let (_, p) = read_varint(bytes, pos)?;
            Ok(p)
        }
        WIRE_64 => {
            if pos + 8 > bytes.len() {
                return Err(KernelError::Model {
                    detail: "truncated 64-bit field".into(),
                });
            }
            Ok(pos + 8)
        }
        WIRE_LEN => {
            let (_, p) = read_len_blob(bytes, pos)?;
            Ok(p)
        }
        WIRE_32 => {
            if pos + 4 > bytes.len() {
                return Err(KernelError::Model {
                    detail: "truncated 32-bit field".into(),
                });
            }
            Ok(pos + 4)
        }
        other => Err(KernelError::Model {
            detail: format!("unsupported protobuf wire type {other}"),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn int8_resnet() -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/int8/resnet34_int8.onnx");
        p.is_file().then_some(p)
    }

    #[test]
    fn dequant_conv370_matches_reference() {
        let Some(path) = int8_resnet() else {
            return;
        };
        let init = load_initializers(&path).unwrap();
        let w = take_f32(&init, "onnx::Conv_370", &[32, 32, 3, 3]).unwrap();
        let b = take_f32(&init, "onnx::Conv_371", &[32]).unwrap();
        let sum: f32 = w.iter().sum();
        let bsum: f32 = b.iter().sum();
        assert!(
            (sum + 13.909225).abs() < 0.01,
            "conv370 sum={sum} first={:?}",
            &w[..8]
        );
        assert!((bsum - 5.667903).abs() < 0.01, "bias371 sum={bsum}");
        assert!((w[0] + 0.02577596).abs() < 1e-5, "w0={}", w[0]);
    }

    #[test]
    fn dequant_lstm784_matches_reference() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/int8/powerset_int8.onnx");
        if !p.is_file() {
            return;
        }
        let init = load_initializers(&p).unwrap();
        let w = take_f32(&init, "onnx::LSTM_784", &[2, 512, 60]).unwrap();
        let sum: f32 = w.iter().sum();
        assert!(
            (sum + 3267.298).abs() < 0.5,
            "lstm784 sum={sum} first={:?}",
            &w[..8]
        );
        assert!((w[0] + 0.48777825).abs() < 1e-4, "w0={}", w[0]);
    }
}
