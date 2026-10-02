fn quantize(src: &[f32], scale: f32, zp: i8, dst: &mut [i8]) {
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    #[cfg(target_arch = "aarch64")]
    quantize_neon(src, s, z, dst);
    #[cfg(not(target_arch = "aarch64"))]
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        let q = (v / s).round() + z;
        *d = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
fn quantize_neon(src: &[f32], s: f32, z: f32, dst: &mut [i8]) {
    use std::arch::aarch64::{vgetq_lane_s32, vmovq_n_f32};
    let vs = unsafe { vmovq_n_f32(s) };
    let vz = unsafe { vmovq_n_f32(z) };
    let lo = unsafe { vmovq_n_f32(-128.0) };
    let hi = unsafe { vmovq_n_f32(127.0) };
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        unsafe {
            let q0 = quant4(src.as_ptr().add(i), vs, vz, lo, hi);
            let q1 = quant4(src.as_ptr().add(i + 4), vs, vz, lo, hi);
            let q2 = quant4(src.as_ptr().add(i + 8), vs, vz, lo, hi);
            let q3 = quant4(src.as_ptr().add(i + 12), vs, vz, lo, hi);
            store16_i8(dst.as_mut_ptr().add(i), q0, q1, q2, q3);
        }
        i += 16;
    }
    while i + 4 <= n {
        unsafe {
            let qi = quant4(src.as_ptr().add(i), vs, vz, lo, hi);
            dst[i] = vgetq_lane_s32::<0>(qi) as i8;
            dst[i + 1] = vgetq_lane_s32::<1>(qi) as i8;
            dst[i + 2] = vgetq_lane_s32::<2>(qi) as i8;
            dst[i + 3] = vgetq_lane_s32::<3>(qi) as i8;
        }
        i += 4;
    }
    for j in i..n {
        let q = (src[j] / s).round() + z;
        dst[j] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn quant4(
    src: *const f32,
    vs: std::arch::aarch64::float32x4_t,
    vz: std::arch::aarch64::float32x4_t,
    lo: std::arch::aarch64::float32x4_t,
    hi: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::{
        vaddq_f32, vcvtq_s32_f32, vdivq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vrndaq_f32,
    };
    unsafe {
        let x = vld1q_f32(src);
        vcvtq_s32_f32(vmaxq_f32(
            vminq_f32(vaddq_f32(vrndaq_f32(vdivq_f32(x, vs)), vz), hi),
            lo,
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store16_i8(
    dst: *mut i8,
    q0: std::arch::aarch64::int32x4_t,
    q1: std::arch::aarch64::int32x4_t,
    q2: std::arch::aarch64::int32x4_t,
    q3: std::arch::aarch64::int32x4_t,
) {
    use std::arch::aarch64::{vcombine_s8, vcombine_s16, vqmovn_s16, vqmovn_s32, vst1q_s8};
    unsafe {
        let a = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        let b = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(dst, vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)));
    }
}

