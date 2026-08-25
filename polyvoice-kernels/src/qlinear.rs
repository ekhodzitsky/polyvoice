//! ONNX `DynamicQuantizeLinear` (UINT8) + `MatMulInteger`.

/// Quantize `x` to UINT8 as ONNX `DynamicQuantizeLinear`.
pub fn dyn_quant_u8(x: &[f32]) -> (Vec<u8>, f32, u8) {
    let mut xmin = 0.0f32;
    let mut xmax = 0.0f32;
    for &v in x {
        xmin = xmin.min(v);
        xmax = xmax.max(v);
    }
    xmin = xmin.min(0.0);
    xmax = xmax.max(0.0);
    let scale = if xmax > xmin {
        (xmax - xmin) / 255.0
    } else {
        1.0
    };
    let zp = ((0.0 - xmin) / scale).round_ties_even().clamp(0.0, 255.0) as u8;
    let zpf = f32::from(zp);
    let mut y = vec![0u8; x.len()];
    for (o, &v) in y.iter_mut().zip(x.iter()) {
        *o = (v / scale + zpf).round_ties_even().clamp(0.0, 255.0) as u8;
    }
    (y, scale, zp)
}

/// `Y = ((A - a_zp) @ (B - b_zp)) * a_scale * b_scale + bias`.
///
/// `A` is `[m, k]` UINT8, `B` is **`[n, k]`** INT8 (transposed from ONNX
/// `[k, n]` so each output channel is contiguous). `b_scale` / `b_zp` are
/// scalar or length-`n`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_integer(
    a: &[u8],
    a_zp: u8,
    b_nk: &[i8],
    b_zp: &[i8],
    a_scale: f32,
    b_scale: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b_nk.len(), n.saturating_mul(k));
    debug_assert_eq!(c.len(), m.saturating_mul(n));
    debug_assert_eq!(bias.len(), n);
    let az = i32::from(a_zp);
    let zp0 = i32::from(b_zp.first().copied().unwrap_or(0));
    let s0 = b_scale.first().copied().unwrap_or(1.0);
    for mi in 0..m {
        let arow = &a[mi * k..mi * k + k];
        let crow = &mut c[mi * n..mi * n + n];
        for ni in 0..n {
            let bz = if b_zp.len() == n {
                i32::from(b_zp[ni])
            } else {
                zp0
            };
            let brow = &b_nk[ni * k..ni * k + k];
            let mut acc = 0i32;
            let mut ki = 0;
            while ki + 8 <= k {
                acc += (i32::from(arow[ki]) - az) * (i32::from(brow[ki]) - bz);
                acc += (i32::from(arow[ki + 1]) - az) * (i32::from(brow[ki + 1]) - bz);
                acc += (i32::from(arow[ki + 2]) - az) * (i32::from(brow[ki + 2]) - bz);
                acc += (i32::from(arow[ki + 3]) - az) * (i32::from(brow[ki + 3]) - bz);
                acc += (i32::from(arow[ki + 4]) - az) * (i32::from(brow[ki + 4]) - bz);
                acc += (i32::from(arow[ki + 5]) - az) * (i32::from(brow[ki + 5]) - bz);
                acc += (i32::from(arow[ki + 6]) - az) * (i32::from(brow[ki + 6]) - bz);
                acc += (i32::from(arow[ki + 7]) - az) * (i32::from(brow[ki + 7]) - bz);
                ki += 8;
            }
            while ki < k {
                acc += (i32::from(arow[ki]) - az) * (i32::from(brow[ki]) - bz);
                ki += 1;
            }
            let sw = if b_scale.len() == n { b_scale[ni] } else { s0 };
            crow[ni] = acc as f32 * a_scale * sw + bias[ni];
        }
    }
}

/// ONNX `[k, n]` INT8 → `[n, k]` for the integer GEMM.
pub fn transpose_kn(w: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut t = vec![0i8; n.saturating_mul(k)];
    for ki in 0..k {
        for ni in 0..n {
            t[ni * k + ki] = w[ki * n + ni];
        }
    }
    t
}

/// Dynamic-quant `X[m,k]` then integer GEMM against INT8 `W[k,n]`.
#[allow(clippy::too_many_arguments)]
pub fn dyn_matmul(
    x: &[f32],
    w: &[i8],
    w_scale: &[f32],
    w_zp: &[i8],
    bias: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let (xq, xs, xz) = dyn_quant_u8(x);
    let mut y = vec![0.0f32; m.saturating_mul(n)];
    #[cfg(not(target_vendor = "apple"))]
    if crate::rten_matmul::gemm_u8i8_nk(&xq, xz, w, w_scale, w_zp, bias, &mut y, m, n, k, xs) {
        return y;
    }
    matmul_integer(&xq, xz, w, w_zp, xs, w_scale, bias, &mut y, m, n, k);
    y
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn dyn_matmul_rten_matches_scalar() {
        let m = 6;
        let n = 8;
        let k = 12;
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.07 - 1.3).collect();
        let w: Vec<i8> = (0..n * k)
            .map(|i| (i as i8).wrapping_mul(3).wrapping_sub(17))
            .collect();
        let w_scale: Vec<f32> = (0..n).map(|i| 0.01 + i as f32 * 0.001).collect();
        let w_zp: Vec<i8> = (0..n).map(|i| (i as i8) - 2).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.05 - 0.2).collect();
        let got = dyn_matmul(&x, &w, &w_scale, &w_zp, &bias, m, n, k);
        let (xq, xs, xz) = dyn_quant_u8(&x);
        let mut want = vec![0.0f32; m * n];
        matmul_integer(&xq, xz, &w, &w_zp, xs, &w_scale, &bias, &mut want, m, n, k);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "i={i} got={g} want={w}");
        }
    }

    #[test]
    fn dyn_quant_includes_zero() {
        let x = vec![-1.0, 0.0, 1.0];
        let (q, s, zp) = dyn_quant_u8(&x);
        assert!(s > 0.0);
        let deq: Vec<f32> = q
            .iter()
            .map(|&v| (f32::from(v) - f32::from(zp)) * s)
            .collect();
        assert!((deq[1]).abs() < s);
        assert!(deq[0] < 0.0 && deq[2] > 0.0);
    }
}
