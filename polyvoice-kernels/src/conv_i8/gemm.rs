#[allow(clippy::too_many_arguments)]
fn gemm_panel(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    nk: &[i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let mut mo = oc0;
    while mo < oc1 {
        let mr = (conv.oc - mo).min(MR);
        let mut no = 0usize;
        while no < pn {
            let nr = (pn - no).min(NR);
            if mr == MR && nr == NR {
                let mut tmp = [[0i32; NR]; MR];
                kernel_mxn(conv, mo, &nk[no * conv.k_pad..], &mut tmp);
                for (mi, row) in tmp.iter().enumerate() {
                    for (t, &acc) in row.iter().enumerate() {
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no + t,
                            mo + mi,
                            acc,
                            relu,
                            i8d,
                        );
                    }
                }
            } else {
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * conv.k_pad..(mo + mi + 1) * conv.k_pad];
                    for t in 0..nr {
                        let xr = &nk[(no + t) * conv.k_pad..(no + t + 1) * conv.k_pad];
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no + t,
                            mo + mi,
                            dot_i8(wr, xr),
                            relu,
                            i8d,
                        );
                    }
                }
            }
            no += nr;
        }
        mo += mr;
    }
}

#[allow(clippy::too_many_arguments)]
fn store_col(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    tile: &[i8],
    _k_raw: usize,
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    for oc in oc0..oc1 {
        let wr = &conv.q_w_pad[oc * conv.k_pad..oc * conv.k_pad + conv.k_pad];
        let a = dot_i8(wr, &tile[..conv.k_pad]);
        write_out_sl(conv, yd, ybase, yh, yw, oy, ox, oc, a, relu, i8d);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_out_sl(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    oc: usize,
    acc: i32,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    let mut v = acc as f32 * conv.out_scale[oc] + conv.eff_bias[oc];
    if relu && v < 0.0 {
        v = 0.0;
    }
    let idx = ybase + (oc * yh + oy) * yw + ox;
    if let Some(d) = i8d {
        d.store(idx, v);
    } else {
        if let Some((s, z)) = conv.out_q {
            v = requant_f32(v, s, z);
        }
        yd[idx] = v;
    }
}

#[cfg(target_arch = "aarch64")]
fn has_dotprod() -> bool {
    #[cfg(target_vendor = "apple")]
    {
        true
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
}

/// AVX-512 VNNI (VPDPBUSD): the x86_64 exact-integer i8 dot. VPMADDUBSW
/// kernels are not exact for full-range i8, so VNNI is the only gate.
#[cfg(target_arch = "x86_64")]
fn has_avx512_vnni() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    })
}

/// True when the zip 4×16 tile kernels can run on this CPU.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn zip_ok(k_pad: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        has_dotprod() && k_pad.is_multiple_of(4)
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_avx512_vnni() && k_pad.is_multiple_of(4)
    }
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        return unsafe { dot_i8_sdot(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx512_vnni() {
        return unsafe { dot_i8_vnni(a, b) };
    }
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += i32::from(x) * i32::from(y);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe)]
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::{int8x16_t, int32x4_t, vaddvq_s32, vdupq_n_s32, vld1q_s8};
    debug_assert!(a.len().is_multiple_of(16));
    let mut acc = unsafe { vdupq_n_s32(0) };
    let mut k = 0;
    while k < a.len() {
        unsafe {
            let av: int8x16_t = vld1q_s8(a.as_ptr().add(k));
            let bv: int8x16_t = vld1q_s8(b.as_ptr().add(k));
            acc = sdot_dotprod(acc, av, bv);
        }
        k += 16;
    }
    let _: int32x4_t = acc;
    unsafe { vaddvq_s32(acc) }
}

/// Full-range i8×i8 dot via VPDPBUSD (u8×i8→i32). `a` is offset by +128
/// (sign-bit flip) to fit u8; the 128·Σb bias is subtracted in the epilogue.
/// Unlike VPMADDUBSW there is no intermediate saturation, so this is exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[allow(unused_unsafe)]
unsafe fn dot_i8_vnni(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::{
        _mm_cvtsi128_si32, _mm_dpbusd_epi32, _mm_extract_epi32, _mm_loadu_si128, _mm_set1_epi8,
        _mm_setzero_si128, _mm_xor_si128, _mm512_dpbusd_epi32, _mm512_loadu_si512,
        _mm512_reduce_add_epi32, _mm512_set1_epi8, _mm512_setzero_si512, _mm512_xor_si512,
    };
    let n = a.len().min(b.len());
    let mut acc = unsafe { _mm512_setzero_si512() };
    let mut bsum = unsafe { _mm512_setzero_si512() };
    let flip = unsafe { _mm512_set1_epi8(-128) };
    let ones = unsafe { _mm512_set1_epi8(1) };
    let mut k = 0usize;
    while k + 64 <= n {
        unsafe {
            let av = _mm512_xor_si512(_mm512_loadu_si512(a.as_ptr().add(k).cast()), flip);
            let bv = _mm512_loadu_si512(b.as_ptr().add(k).cast());
            acc = _mm512_dpbusd_epi32(acc, av, bv);
            bsum = _mm512_dpbusd_epi32(bsum, ones, bv);
        }
        k += 64;
    }
    let mut total = unsafe { _mm512_reduce_add_epi32(acc) - 128 * _mm512_reduce_add_epi32(bsum) };
    // 16-byte tail chunks (k_pad is a multiple of 16, so the scalar tail is
    // only a safety net for non-padded callers).
    while k + 16 <= n {
        unsafe {
            let av = _mm_xor_si128(
                _mm_loadu_si128(a.as_ptr().add(k).cast()),
                _mm_set1_epi8(-128),
            );
            let bv = _mm_loadu_si128(b.as_ptr().add(k).cast());
            let acc4 = _mm_dpbusd_epi32(_mm_setzero_si128(), av, bv);
            let bsum4 = _mm_dpbusd_epi32(_mm_setzero_si128(), _mm_set1_epi8(1), bv);
            total += _mm_cvtsi128_si32(acc4)
                + _mm_extract_epi32::<1>(acc4)
                + _mm_extract_epi32::<2>(acc4)
                + _mm_extract_epi32::<3>(acc4)
                - 128
                    * (_mm_cvtsi128_si32(bsum4)
                        + _mm_extract_epi32::<1>(bsum4)
                        + _mm_extract_epi32::<2>(bsum4)
                        + _mm_extract_epi32::<3>(bsum4));
        }
        k += 16;
    }
    let mut rest = 0i32;
    for i in k..n {
        rest += i32::from(a[i]) * i32::from(b[i]);
    }
    total + rest
}

fn kernel_mxn(conv: &Conv2d, mo: usize, tile: &[i8], out: &mut [[i32; NR]; MR]) {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        unsafe {
            kernel_mxn_sdot(conv, mo, tile, out);
        }
        return;
    }
    for mi in 0..MR {
        let wr = &conv.q_w_pad[(mo + mi) * conv.k_pad..(mo + mi + 1) * conv.k_pad];
        for t in 0..NR {
            out[mi][t] = dot_i8(wr, &tile[t * conv.k_pad..(t + 1) * conv.k_pad]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe)]
unsafe fn kernel_mxn_sdot(conv: &Conv2d, mo: usize, tile: &[i8], out: &mut [[i32; NR]; MR]) {
    use std::arch::aarch64::{int8x16_t, vaddvq_s32, vdupq_n_s32, vld1q_s8};
    let k_pad = conv.k_pad;
    let mut acc = [[unsafe { vdupq_n_s32(0) }; NR]; MR];
    let wp = conv.q_w_pad.as_ptr();
    let tp = tile.as_ptr();
    let mut k = 0;
    while k < k_pad {
        unsafe {
            let mut xs = [vld1q_s8(tp.add(k)); NR];
            for (t, xv) in xs.iter_mut().enumerate() {
                *xv = vld1q_s8(tp.add(t * k_pad + k));
            }
            for (row, acc_row) in acc.iter_mut().enumerate() {
                let w: int8x16_t = vld1q_s8(wp.add((mo + row) * k_pad + k));
                for (t, xv) in xs.iter().enumerate() {
                    acc_row[t] = sdot_dotprod(acc_row[t], w, *xv);
                }
            }
        }
        k += 16;
    }
    for (mi, row) in out.iter_mut().enumerate() {
        for (t, slot) in row.iter_mut().enumerate() {
            *slot = unsafe { vaddvq_s32(acc[mi][t]) };
        }
    }
}

/// KN-panel GEMM: each vector holds adjacent output pixels (not a K-reduction).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn gemm_panel_kn16(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &[i8],
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let zip_len = conv.k_pad.saturating_mul(pn);
    let use_zip = zip.len() >= zip_len && pn.is_multiple_of(NR16) && conv.k_pad.is_multiple_of(4);
    if use_zip {
        pack_kn_zip16(kn, &mut zip[..zip_len], conv.k_pad, pn);
    }
    let mut mo = oc0;
    while mo < oc1 {
        let mr = (conv.oc - mo).min(MR);
        let mut no = 0usize;
        while no < pn {
            if mr == MR && no + NR16 <= pn {
                unsafe {
                    if use_zip {
                        kernel_4x16_zip_store(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no,
                            mo,
                            zip,
                            pn,
                            no,
                            relu,
                            i8d,
                        );
                    } else {
                        kernel_4x16_kn_store(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no,
                            mo,
                            kn,
                            pn,
                            no,
                            relu,
                            i8d,
                        );
                    }
                }
                no += NR16;
            } else {
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * conv.k_pad..(mo + mi + 1) * conv.k_pad];
                    for t in no..pn {
                        let mut acc = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            acc += i32::from(wv) * i32::from(kn[kk * pn + t]);
                        }
                        write_out_sl(conv, yd, ybase, yh, yw, oy, ox + t, mo + mi, acc, relu, i8d);
                    }
                }
                no = pn;
            }
        }
        mo += mr;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store4_i8(dst: *mut i8, v: std::arch::aarch64::float32x4_t, scale: f32, zp: f32) {
    use std::arch::aarch64::{
        vaddq_f32, vcombine_s16, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vget_lane_s32, vmaxq_f32,
        vminq_f32, vqmovn_s16, vqmovn_s32, vreinterpret_s32_s8, vrndaq_f32,
    };
    unsafe {
        let q = vaddq_f32(
            vrndaq_f32(vdivq_f32(v, vdupq_n_f32(scale))),
            vdupq_n_f32(zp),
        );
        let qi = vcvtq_s32_f32(vmaxq_f32(
            vminq_f32(q, vdupq_n_f32(127.0)),
            vdupq_n_f32(-128.0),
        ));
        let i16 = vqmovn_s32(qi);
        let i8x8 = vqmovn_s16(vcombine_s16(i16, i16));
        let bits = vget_lane_s32::<0>(vreinterpret_s32_s8(i8x8)) as u32;
        core::ptr::write_unaligned(dst as *mut u32, bits);
    }
}

/// Requant 4 lanes onto the pre-add QDQ lattice (round half away, clamp).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn requant4_f32(
    v: std::arch::aarch64::float32x4_t,
    scale: f32,
    zp: f32,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vaddq_f32, vdivq_f32, vdupq_n_f32, vmaxq_f32, vminq_f32, vmulq_f32, vrndaq_f32, vsubq_f32,
    };
    unsafe {
        let vs = vdupq_n_f32(scale);
        let vz = vdupq_n_f32(zp);
        let q = vrndaq_f32(vdivq_f32(v, vs));
        let q = vaddq_f32(q, vz);
        let q = vmaxq_f32(vminq_f32(q, vdupq_n_f32(127.0)), vdupq_n_f32(-128.0));
        vmulq_f32(vsubq_f32(q, vz), vs)
    }
}

#[cfg(target_arch = "x86_64")]
fn pack_kn_zip16(kn: &[i8], zip: &mut [i8], k_pad: usize, pn: usize) {
    let mut o = 0usize;
    let mut k = 0usize;
    while k + 4 <= k_pad {
        let mut n0 = 0usize;
        while n0 + 16 <= pn {
            for j in 0..16 {
                zip[o + 4 * j] = kn[k * pn + n0 + j];
                zip[o + 4 * j + 1] = kn[(k + 1) * pn + n0 + j];
                zip[o + 4 * j + 2] = kn[(k + 2) * pn + n0 + j];
                zip[o + 4 * j + 3] = kn[(k + 3) * pn + n0 + j];
            }
            o += 64;
            n0 += 16;
        }
        k += 4;
    }
}

/// Sum of one padded weight row — the `128·Σw` correction the VNNI kernels
/// subtract after offsetting activations by +128.
#[cfg(target_arch = "x86_64")]
fn w_row_sum(conv: &Conv2d, oc: usize) -> i32 {
    if let Some(&s) = conv.w_sum.get(oc) {
        return s;
    }
    let row = &conv.q_w_pad[oc * conv.k_pad..(oc + 1) * conv.k_pad];
    row.iter().map(|&v| i32::from(v)).sum()
}

/// copysign(0.5, x) — the half-away-from-zero rounding addend.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unused_unsafe)]
#[inline]
pub(crate) unsafe fn round_half_away_adj(
    x: std::arch::x86_64::__m512,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::{_mm512_and_ps, _mm512_or_ps, _mm512_set1_ps};
    unsafe { _mm512_or_ps(_mm512_and_ps(x, _mm512_set1_ps(-0.0)), _mm512_set1_ps(0.5)) }
}

/// f32::round (half away from zero) to integers: trunc(x + copysign(0.5, x)).
/// Exact for |x| < 2^31, so valid for i8 lattice indices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unused_unsafe)]
#[inline]
pub(crate) unsafe fn round_half_away_epi32(
    x: std::arch::x86_64::__m512,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::{_mm512_add_ps, _mm512_cvttps_epi32};
    unsafe { _mm512_cvttps_epi32(_mm512_add_ps(x, round_half_away_adj(x))) }
}

/// Dequant + relu + store of 16 adjacent output pixels held in one zmm
/// (dword lane j = pixel ox+j). Mirrors the f32/i8 store of `write_out_sl`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[allow(unused_unsafe)]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn store16_out(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    oc: usize,
    acc: std::arch::x86_64::__m512i,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::x86_64::{
        _mm_storeu_si128, _mm512_add_epi32, _mm512_cvtepi32_ps, _mm512_cvtsepi32_epi8,
        _mm512_div_ps, _mm512_fmadd_ps, _mm512_max_epi32, _mm512_max_ps, _mm512_min_epi32,
        _mm512_mul_ps, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
        _mm512_sub_ps,
    };
    unsafe {
        let mut v = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(acc),
            _mm512_set1_ps(conv.out_scale[oc]),
            _mm512_set1_ps(conv.eff_bias[oc]),
        );
        if relu {
            v = _mm512_max_ps(v, _mm512_setzero_ps());
        }
        let idx = ybase + (oc * yh + oy) * yw + ox;
        if let Some(d) = i8d {
            // Graph QuantizeLinear: divide, round half away, add zp, clamp.
            let qi = _mm512_add_epi32(
                round_half_away_epi32(_mm512_div_ps(v, _mm512_set1_ps(d.scale))),
                _mm512_set1_epi32(i32::from(d.zp as i8)),
            );
            let qi = _mm512_max_epi32(
                _mm512_min_epi32(qi, _mm512_set1_epi32(127)),
                _mm512_set1_epi32(-128),
            );
            _mm_storeu_si128((d.p as *mut i8).add(idx).cast(), _mm512_cvtsepi32_epi8(qi));
        } else {
            if let Some((s, z)) = conv.out_q {
                // Requant onto the pre-add QDQ lattice.
                let qi = _mm512_add_epi32(
                    round_half_away_epi32(_mm512_div_ps(v, _mm512_set1_ps(s))),
                    _mm512_set1_epi32(i32::from(z)),
                );
                let qi = _mm512_max_epi32(
                    _mm512_min_epi32(qi, _mm512_set1_epi32(127)),
                    _mm512_set1_epi32(-128),
                );
                v = _mm512_mul_ps(
                    _mm512_sub_ps(_mm512_cvtepi32_ps(qi), _mm512_set1_ps(f32::from(z))),
                    _mm512_set1_ps(s),
                );
            }
            _mm512_storeu_ps(yd.as_mut_ptr().add(idx), v);
        }
    }
}

/// 4 output rows × 16 pixels from the zipped panel: dword lane j of the
/// accumulator is the dot for pixel `n0+j`. VPDPBUSD is u8×i8, so the
/// activation vector is offset by +128 (sign-bit flip) and each row's
/// `128·Σw` is subtracted in the epilogue — exact, no saturation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[allow(unused_unsafe)]
#[allow(clippy::too_many_arguments)]
unsafe fn kernel_4x16_zip_store(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    mo: usize,
    zip: &[i8],
    pn: usize,
    n0: usize,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::x86_64::{
        _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_set1_epi8, _mm512_set1_epi32,
        _mm512_setzero_si512, _mm512_slli_epi32, _mm512_sub_epi32, _mm512_xor_si512,
    };
    let k_pad = conv.k_pad;
    let tiles_n = pn / NR16;
    let tile = n0 / NR16;
    let zp = zip.as_ptr();
    let wp = conv.q_w_pad.as_ptr();
    unsafe {
        let flip = _mm512_set1_epi8(-128);
        let mut acc = [_mm512_setzero_si512(); MR];
        let mut k = 0usize;
        while k + 4 <= k_pad {
            let off = (k / 4) * tiles_n * 64 + tile * 64;
            let xv = _mm512_xor_si512(_mm512_loadu_si512(zp.add(off).cast()), flip);
            for r in 0..MR {
                let w4 = core::ptr::read_unaligned(wp.add((mo + r) * k_pad + k).cast::<i32>());
                acc[r] = _mm512_dpbusd_epi32(acc[r], xv, _mm512_set1_epi32(w4));
            }
            k += 4;
        }
        for r in 0..MR {
            let corr = _mm512_slli_epi32::<7>(_mm512_set1_epi32(w_row_sum(conv, mo + r)));
            let a = _mm512_sub_epi32(acc[r], corr);
            store16_out(conv, yd, ybase, yh, yw, oy, ox, mo + r, a, relu, i8d);
        }
    }
}

/// Same 4×16 tile straight from the KN panel: the 4-K × 16-pixel interleave
/// the zip packer writes is done in registers instead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[allow(unused_unsafe)]
#[allow(clippy::too_many_arguments)]
unsafe fn kernel_4x16_kn_store(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    mo: usize,
    kn: &[i8],
    pn: usize,
    n0: usize,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::x86_64::{
        _mm_loadu_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8,
        _mm_unpacklo_epi16, _mm512_castsi128_si512, _mm512_dpbusd_epi32, _mm512_inserti32x4,
        _mm512_set1_epi8, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_slli_epi32,
        _mm512_sub_epi32, _mm512_xor_si512,
    };
    let k_pad = conv.k_pad;
    let wp = conv.q_w_pad.as_ptr();
    let kp = kn.as_ptr();
    unsafe {
        let flip = _mm512_set1_epi8(-128);
        let mut acc = [_mm512_setzero_si512(); MR];
        let mut k = 0usize;
        while k + 4 <= k_pad {
            let b0 = _mm_loadu_si128(kp.add(k * pn + n0).cast());
            let b1 = _mm_loadu_si128(kp.add((k + 1) * pn + n0).cast());
            let b2 = _mm_loadu_si128(kp.add((k + 2) * pn + n0).cast());
            let b3 = _mm_loadu_si128(kp.add((k + 3) * pn + n0).cast());
            let z01l = _mm_unpacklo_epi8(b0, b1);
            let z01h = _mm_unpackhi_epi8(b0, b1);
            let z23l = _mm_unpacklo_epi8(b2, b3);
            let z23h = _mm_unpackhi_epi8(b2, b3);
            let q0 = _mm_unpacklo_epi16(z01l, z23l);
            let q1 = _mm_unpackhi_epi16(z01l, z23l);
            let q2 = _mm_unpacklo_epi16(z01h, z23h);
            let q3 = _mm_unpackhi_epi16(z01h, z23h);
            let xb = _mm512_inserti32x4::<3>(
                _mm512_inserti32x4::<2>(
                    _mm512_inserti32x4::<1>(_mm512_castsi128_si512(q0), q1),
                    q2,
                ),
                q3,
            );
            let xv = _mm512_xor_si512(xb, flip);
            for r in 0..MR {
                let w4 = core::ptr::read_unaligned(wp.add((mo + r) * k_pad + k).cast::<i32>());
                acc[r] = _mm512_dpbusd_epi32(acc[r], xv, _mm512_set1_epi32(w4));
            }
            k += 4;
        }
        for r in 0..MR {
            let corr = _mm512_slli_epi32::<7>(_mm512_set1_epi32(w_row_sum(conv, mo + r)));
            let a = _mm512_sub_epi32(acc[r], corr);
            store16_out(conv, yd, ybase, yh, yw, oy, ox, mo + r, a, relu, i8d);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn pack_kn_zip16(kn: &[i8], zip: &mut [i8], k_pad: usize, pn: usize) {
    use std::arch::aarch64::{
        vld1q_s8, vreinterpretq_s8_s16, vreinterpretq_s16_s8, vst1q_s8, vzip1q_s8, vzip1q_s16,
        vzip2q_s8, vzip2q_s16,
    };
    let mut o = 0usize;
    let mut k = 0usize;
    while k + 4 <= k_pad {
        let mut n0 = 0usize;
        while n0 + 16 <= pn {
            unsafe {
                let b0 = vld1q_s8(kn.as_ptr().add(k * pn + n0));
                let b1 = vld1q_s8(kn.as_ptr().add((k + 1) * pn + n0));
                let b2 = vld1q_s8(kn.as_ptr().add((k + 2) * pn + n0));
                let b3 = vld1q_s8(kn.as_ptr().add((k + 3) * pn + n0));
                let z01l = vzip1q_s8(b0, b1);
                let z01h = vzip2q_s8(b0, b1);
                let z23l = vzip1q_s8(b2, b3);
                let z23h = vzip2q_s8(b2, b3);
                let zp = zip.as_mut_ptr().add(o);
                vst1q_s8(
                    zp,
                    vreinterpretq_s8_s16(vzip1q_s16(
                        vreinterpretq_s16_s8(z01l),
                        vreinterpretq_s16_s8(z23l),
                    )),
                );
                vst1q_s8(
                    zp.add(16),
                    vreinterpretq_s8_s16(vzip2q_s16(
                        vreinterpretq_s16_s8(z01l),
                        vreinterpretq_s16_s8(z23l),
                    )),
                );
                vst1q_s8(
                    zp.add(32),
                    vreinterpretq_s8_s16(vzip1q_s16(
                        vreinterpretq_s16_s8(z01h),
                        vreinterpretq_s16_s8(z23h),
                    )),
                );
                vst1q_s8(
                    zp.add(48),
                    vreinterpretq_s8_s16(vzip2q_s16(
                        vreinterpretq_s16_s8(z01h),
                        vreinterpretq_s16_s8(z23h),
                    )),
                );
            }
            o += 64;
            n0 += 16;
        }
        k += 4;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe, clippy::too_many_arguments)]
unsafe fn kernel_4x16_zip_store(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    mo: usize,
    zip: &[i8],
    pn: usize,
    n0: usize,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::aarch64::{
        int8x16_t, int32x4_t, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_s8,
        vmaxq_f32, vmovq_n_f32, vst1q_f32,
    };
    let k_pad = conv.k_pad;
    let tiles_n = pn / NR16;
    let tile = n0 / NR16;
    let zp = zip.as_ptr();
    let mut acc = [[unsafe { vdupq_n_s32(0) }; 4]; MR];
    let mut k = 0usize;
    while k + 4 <= k_pad {
        unsafe {
            let off = (k / 4) * tiles_n * 64 + tile * 64;
            let bt = [
                vld1q_s8(zp.add(off)),
                vld1q_s8(zp.add(off + 16)),
                vld1q_s8(zp.add(off + 32)),
                vld1q_s8(zp.add(off + 48)),
            ];
            let a_tile: int8x16_t = if !conv.q_w_4x4.is_empty() {
                let w4 = conv.q_w_4x4.as_ptr();
                let wrow = (mo / 4) * (k_pad / 4) * 16;
                vld1q_s8(w4.add(wrow + (k / 4) * 16))
            } else {
                let wp = conv.q_w_pad.as_ptr();
                let mut abuf = [0i8; 16];
                core::ptr::copy_nonoverlapping(wp.add(mo * k_pad + k), abuf.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 1) * k_pad + k),
                    abuf.as_mut_ptr().add(4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 2) * k_pad + k),
                    abuf.as_mut_ptr().add(8),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 3) * k_pad + k),
                    abuf.as_mut_ptr().add(12),
                    4,
                );
                vld1q_s8(abuf.as_ptr())
            };
            for i in 0..4 {
                acc[0][i] = sdot_lane::<0>(acc[0][i], bt[i], a_tile);
                acc[1][i] = sdot_lane::<1>(acc[1][i], bt[i], a_tile);
                acc[2][i] = sdot_lane::<2>(acc[2][i], bt[i], a_tile);
                acc[3][i] = sdot_lane::<3>(acc[3][i], bt[i], a_tile);
            }
        }
        k += 4;
    }
    let zero = unsafe { vmovq_n_f32(0.0) };
    for r in 0..MR {
        let scale = unsafe { vdupq_n_f32(conv.out_scale[mo + r]) };
        let bias = unsafe { vdupq_n_f32(conv.eff_bias[mo + r]) };
        let dst0 = ybase + ((mo + r) * yh + oy) * yw + ox;
        for i in 0..4 {
            unsafe {
                let mut v = vfmaq_f32(bias, vcvtq_f32_s32(acc[r][i]), scale);
                if relu {
                    v = vmaxq_f32(v, zero);
                }
                if let Some(d) = i8d {
                    store4_i8((d.p as *mut i8).add(dst0 + i * 4), v, d.scale, d.zp);
                } else {
                    if let Some((s, z)) = conv.out_q {
                        v = requant4_f32(v, s, f32::from(z));
                    }
                    vst1q_f32(yd[dst0 + i * 4..].as_mut_ptr(), v);
                }
            }
        }
    }
    let _: int32x4_t = acc[0][0];
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe, clippy::too_many_arguments)]
unsafe fn kernel_4x16_kn_store(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    mo: usize,
    kn: &[i8],
    pn: usize,
    n0: usize,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::aarch64::{
        int8x16_t, int32x4_t, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_s8,
        vmaxq_f32, vmovq_n_f32, vreinterpretq_s8_s16, vreinterpretq_s16_s8, vst1q_f32, vzip1q_s8,
        vzip1q_s16, vzip2q_s8, vzip2q_s16,
    };
    let k_pad = conv.k_pad;
    let wp = conv.q_w_pad.as_ptr();
    let kp = kn.as_ptr();
    let mut acc = [[unsafe { vdupq_n_s32(0) }; 4]; MR];
    let mut k = 0usize;
    while k + 4 <= k_pad {
        unsafe {
            if k + 8 <= k_pad {
                // Next K-group of activations + the next 4×4 weight tile.
                core::arch::asm!(
                    "prfm pldl1keep, [{p}]",
                    p = in(reg) kp.add((k + 4) * pn + n0),
                    options(readonly, nostack),
                );
                if !conv.q_w_4x4.is_empty() {
                    let tiles_k = k_pad / 4;
                    let noff = (mo / 4) * tiles_k * 16 + (k / 4 + 1) * 16;
                    core::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) conv.q_w_4x4.as_ptr().add(noff),
                        options(readonly, nostack),
                    );
                }
            }
            let b0 = vld1q_s8(kp.add(k * pn + n0));
            let b1 = vld1q_s8(kp.add((k + 1) * pn + n0));
            let b2 = vld1q_s8(kp.add((k + 2) * pn + n0));
            let b3 = vld1q_s8(kp.add((k + 3) * pn + n0));
            let z01l = vzip1q_s8(b0, b1);
            let z01h = vzip2q_s8(b0, b1);
            let z23l = vzip1q_s8(b2, b3);
            let z23h = vzip2q_s8(b2, b3);
            let bt = [
                vreinterpretq_s8_s16(vzip1q_s16(
                    vreinterpretq_s16_s8(z01l),
                    vreinterpretq_s16_s8(z23l),
                )),
                vreinterpretq_s8_s16(vzip2q_s16(
                    vreinterpretq_s16_s8(z01l),
                    vreinterpretq_s16_s8(z23l),
                )),
                vreinterpretq_s8_s16(vzip1q_s16(
                    vreinterpretq_s16_s8(z01h),
                    vreinterpretq_s16_s8(z23h),
                )),
                vreinterpretq_s8_s16(vzip2q_s16(
                    vreinterpretq_s16_s8(z01h),
                    vreinterpretq_s16_s8(z23h),
                )),
            ];
            let a_tile: int8x16_t = if !conv.q_w_4x4.is_empty() {
                let tiles_k = k_pad / 4;
                let off = (mo / 4) * tiles_k * 16 + (k / 4) * 16;
                vld1q_s8(conv.q_w_4x4.as_ptr().add(off))
            } else {
                let mut abuf = [0i8; 16];
                core::ptr::copy_nonoverlapping(wp.add(mo * k_pad + k), abuf.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 1) * k_pad + k),
                    abuf.as_mut_ptr().add(4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 2) * k_pad + k),
                    abuf.as_mut_ptr().add(8),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 3) * k_pad + k),
                    abuf.as_mut_ptr().add(12),
                    4,
                );
                vld1q_s8(abuf.as_ptr())
            };
            for i in 0..4 {
                acc[0][i] = sdot_lane::<0>(acc[0][i], bt[i], a_tile);
                acc[1][i] = sdot_lane::<1>(acc[1][i], bt[i], a_tile);
                acc[2][i] = sdot_lane::<2>(acc[2][i], bt[i], a_tile);
                acc[3][i] = sdot_lane::<3>(acc[3][i], bt[i], a_tile);
            }
        }
        k += 4;
    }
    let zero = unsafe { vmovq_n_f32(0.0) };
    for r in 0..MR {
        let scale = unsafe { vdupq_n_f32(conv.out_scale[mo + r]) };
        let bias = unsafe { vdupq_n_f32(conv.eff_bias[mo + r]) };
        let dst0 = ybase + ((mo + r) * yh + oy) * yw + ox;
        for i in 0..4 {
            unsafe {
                let mut v = vfmaq_f32(bias, vcvtq_f32_s32(acc[r][i]), scale);
                if relu {
                    v = vmaxq_f32(v, zero);
                }
                if let Some(d) = i8d {
                    store4_i8((d.p as *mut i8).add(dst0 + i * 4), v, d.scale, d.zp);
                } else {
                    if let Some((s, z)) = conv.out_q {
                        v = requant4_f32(v, s, f32::from(z));
                    }
                    vst1q_f32(yd[dst0 + i * 4..].as_mut_ptr(), v);
                }
            }
        }
    }
    let _: int32x4_t = acc[0][0];
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
#[allow(unused_unsafe)]
unsafe fn sdot_lane<const LANE: i32>(
    mut acc: std::arch::aarch64::int32x4_t,
    b: std::arch::aarch64::int8x16_t,
    a: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SDOT Vd.4S, Vn.16B, Vm.4B[lane] — four columns × 4-K against one A row.
    unsafe {
        match LANE {
            0 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[0]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            1 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[1]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            2 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[2]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            3 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[3]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            _ => {}
        }
    }
    acc
}

/// Must live in a `+dotprod` function — Linux rustc will not assemble `sdot`
/// on the generic aarch64 target. Keep it `#[inline]` so the helper does not
/// take SIMD args across a non-dotprod ABI boundary.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
#[allow(unused_unsafe)]
unsafe fn sdot_dotprod(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SDOT Vd.4S, Vn.16B, Vm.16B — four 4-wide i8 dots into the four lanes.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

#[cfg(test)]
mod strided2_tests {
    #[test]
    fn copy_strided2_matches_scalar() {
        for n in [1usize, 8, 16, 17, 32, 33, 48, 64] {
            let src: Vec<i8> = (0..n * 2 + 8).map(|i| (i as i8).wrapping_mul(3)).collect();
            let mut got = vec![0i8; n];
            let mut want = vec![0i8; n];
            super::copy_strided2(&mut got, &src);
            for i in 0..n {
                want[i] = src[2 * i];
            }
            assert_eq!(got, want, "n={n}");
        }
    }
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    target_vendor = "apple",
    not(apple_accelerate)
))]
mod fused_pack_tests {
    use super::*;

    fn test_conv(ic: usize, oc: usize, zp: i8) -> Conv2d {
        let weights = (0..oc * ic * 9)
            .map(|i| (i as i8).wrapping_mul(17))
            .collect();
        Conv2d::quantized(oc, ic, 3, 1, weights, vec![0.003; oc], vec![0.07; oc])
            .with_input_quant(0.04, zp)
    }

    #[test]
    fn direct_packing_convolution_matches_reference_bits() {
        force_i8(true);
        for (ic, oc) in [(1, 4), (3, 7), (32, 32), (128, 128)] {
            for w in [15usize, 16, 17, 31, 32, 33, 50] {
                for zp in [-128i8, -17, 0, 127] {
                    let conv = test_conv(ic, oc, zp);
                    let mut input = Tensor::zeros(2, ic, 3, w);
                    for (i, value) in input.data.iter_mut().enumerate() {
                        *value = ((i % 101) as f32 - 50.0) * 0.031;
                    }
                    REFERENCE_PACK.with(|v| v.set(true));
                    let want = conv.forward(&input);
                    REFERENCE_PACK.with(|v| v.set(false));
                    let got = conv.forward(&input);
                    assert!(
                        got.data
                            .iter()
                            .zip(&want.data)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "ic={ic} oc={oc} w={w} zp={zp}"
                    );
                }
            }
        }
        force_i8(false);
    }

    #[test]
    #[ignore = "manual paired convolution timing"]
    fn bench_direct_packing_shapes() {
        use std::{hint::black_box, time::Instant};
        force_i8(true);
        for (ic, oc, h, w) in [
            (32, 32, 80, 400),
            (64, 64, 40, 200),
            (128, 128, 20, 100),
            (256, 256, 10, 50),
        ] {
            let conv = test_conv(ic, oc, -128);
            let mut input = Tensor::zeros(1, ic, h, w);
            for (i, value) in input.data.iter_mut().enumerate() {
                *value = ((i % 101) as f32 - 50.0) * 0.031;
            }
            for repeat in 0..6 {
                let order = if repeat % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                };
                for reference in order {
                    REFERENCE_PACK.with(|v| v.set(reference));
                    black_box(conv.forward(black_box(&input)));
                    let start = Instant::now();
                    for _ in 0..5 {
                        black_box(conv.forward(black_box(&input)));
                    }
                    println!(
                        "packing_shape ic={ic} oc={oc} h={h} w={w} repeat={repeat} reference={reference} ms={:.6}",
                        start.elapsed().as_secs_f64() * 200.0
                    );
                }
            }
        }
        REFERENCE_PACK.with(|v| v.set(false));
        force_i8(false);
    }

    #[test]
    fn direct_zip_matches_scalar_layout_and_reference() {
        for ic in [1usize, 2, 3, 4, 7, 16, 32, 128] {
            for zp in [-128i8, -17, 0, 127] {
                let conv =
                    Conv2d::quantized(4, ic, 3, 1, vec![1; 4 * ic * 9], vec![0.1; 4], vec![0.0; 4])
                        .with_input_quant(0.04, zp);
                for w in [16usize, 17, 31, 32, 33, 100] {
                    let h = 5;
                    let wp = w + 2;
                    let plane = ic * wp;
                    let input: Vec<i8> = (0..ic * h * w)
                        .map(|i| (i as i8).wrapping_mul(37))
                        .collect();
                    for oy in 0..h {
                        let mut rows = vec![zp; 3 * plane];
                        for kh in 0..3 {
                            let iy = (oy + kh).checked_sub(1).unwrap_or(h);
                            let slot = row_slot(oy, kh);
                            pack_pad_row(
                                &input,
                                ic,
                                h,
                                w,
                                wp,
                                zp,
                                iy,
                                &mut rows[slot * plane..(slot + 1) * plane],
                            );
                        }
                        for pn in [16usize, 32] {
                            if pn > w {
                                continue;
                            }
                            for ox in [0, w - pn] {
                                let len = conv.k_pad * pn;
                                let mut got = vec![73; len + 16];
                                gather_zip_direct(
                                    &conv,
                                    &rows,
                                    plane,
                                    wp,
                                    oy,
                                    ox,
                                    pn,
                                    &mut got[..len],
                                );
                                let mut kn = vec![91; len];
                                let mut reference = vec![0; len];
                                gather_kn_from_rows(&conv, &rows, plane, wp, oy, ox, pn, &mut kn);
                                pack_kn_zip16(&kn, &mut reference, conv.k_pad, pn);
                                assert_eq!(
                                    &got[..len],
                                    reference,
                                    "ic={ic} zp={zp} w={w} oy={oy} ox={ox} pn={pn}"
                                );
                                assert_eq!(&got[len..], &[73; 16]);
                                for k in 0..conv.k_pad {
                                    for n in 0..pn {
                                        let expected = if k >= ic * 9 {
                                            0
                                        } else {
                                            let (c, kh, kw) = (k / 9, k % 9 / 3, k % 3);
                                            match (
                                                (oy + kh).checked_sub(1),
                                                (ox + n + kw).checked_sub(1),
                                            ) {
                                                (Some(y), Some(x)) if y < h && x < w => {
                                                    input[(c * h + y) * w + x]
                                                }
                                                _ => zp,
                                            }
                                        };
                                        let index = ((k / 4) * (pn / 16) + n / 16) * 64
                                            + (n % 16) * 4
                                            + k % 4;
                                        assert_eq!(got[index], expected, "k={k} n={n}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
