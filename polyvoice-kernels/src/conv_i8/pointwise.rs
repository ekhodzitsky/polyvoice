/// Last input-x of a `pn`-wide 3×3 tile starting at output `ox` (pad=1).
fn last_ix(ox: usize, pn: usize, sx: usize) -> usize {
    (ox + pn - 1) * sx + 1
}

fn gather_one_3x3(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    oy: usize,
    ox: usize,
    tile: &mut [i8],
) {
    let (ic, k_pad, zp, sx, pad) = (
        conv.ic,
        conv.k_pad,
        conv.act_zp,
        conv.stride.max(1),
        conv.pad,
    );
    let k_raw = ic * 9;
    tile[..k_raw].fill(zp);
    tile[k_raw..k_pad].fill(0);
    let hh = h as isize;
    let ww = w as isize;
    let iy0 = oy as isize * sx as isize - pad as isize;
    let ix0 = ox as isize * sx as isize - pad as isize;
    for c in 0..ic {
        for kh in 0..3 {
            let iy = iy0 + kh as isize;
            for kw in 0..3 {
                let ix = ix0 + kw as isize;
                if iy >= 0 && iy < hh && ix >= 0 && ix < ww {
                    tile[c * 9 + kh * 3 + kw] = ximg[(c * h + iy as usize) * w + ix as usize];
                }
            }
        }
    }
}

/// Interior 3×3 tile: gather into KN (contiguous pixels per tap) then
/// transpose to NK for the SDOT kernel. `pn` is 8 or 32.
#[allow(clippy::too_many_arguments)]
fn implicit_tile_3x3(
    conv: &Conv2d,
    ximg: &[i8],
    y: &mut Tensor,
    ybase: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
) {
    let (ic, k_pad, sx) = (conv.ic, conv.k_pad, conv.stride.max(1));
    let nkk = k_pad * pn;
    debug_assert!(nkk <= kn.len() && nkk <= nk.len());
    kn[..nkk].fill(0);
    for c in 0..ic {
        for kh in 0..3 {
            let iy = oy * sx + kh - 1;
            let src = &ximg[(c * h + iy) * w..];
            for kw in 0..3 {
                let kidx = c * 9 + kh * 3 + kw;
                let ix0 = ox * sx + kw - 1;
                let dst = &mut kn[kidx * pn..kidx * pn + pn];
                if sx == 1 {
                    dst.copy_from_slice(&src[ix0..ix0 + pn]);
                } else {
                    for (t, d) in dst.iter_mut().enumerate() {
                        *d = src[ix0 + t * sx];
                    }
                }
            }
        }
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if zip_ok(k_pad) && pn.is_multiple_of(NR16) {
        gemm_panel_kn16(
            conv,
            &mut y.data,
            ybase,
            oh,
            ow,
            oy,
            ox,
            pn,
            kn,
            nk,
            relu,
            None,
            0,
            conv.oc,
        );
        return;
    }
    transpose_kn_nk(&kn[..nkk], &mut nk[..nkk], k_pad, pn);
    gemm_panel(
        conv,
        &mut y.data,
        ybase,
        oh,
        ow,
        oy,
        ox,
        pn,
        nk,
        relu,
        None,
        0,
        conv.oc,
    );
}

fn transpose_kn_nk(kn: &[i8], nk: &mut [i8], k: usize, n: usize) {
    for ki in 0..k {
        let src = &kn[ki * n..ki * n + n];
        for ni in 0..n {
            nk[ni * k + ki] = src[ni];
        }
    }
}

fn conv1x1(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
) {
    let (ic, k_pad, sx) = (conv.ic, conv.k_pad, conv.stride.max(1));
    let (oh, ow, oc) = (y.h, y.w, conv.oc);
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let row_zip = zip_ok(k_pad);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let row_zip = false;
    for ni in 0..n {
        let xbase = ni * ic * h * w;
        let ybase = ni * oc * oh * ow;
        let ximg = &xq[xbase..xbase + ic * h * w];
        for oy in 0..oh {
            let iy = oy * sx;
            if row_zip {
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                conv1x1_row_zip(
                    conv,
                    ximg,
                    h,
                    w,
                    iy,
                    sx,
                    &mut y.data,
                    ybase,
                    oh,
                    ow,
                    oy,
                    kn,
                    nk,
                    relu,
                    k_pad,
                    oc,
                );
            } else {
                let mut ox = 0usize;
                while ox < ow {
                    let pn = conv1x1_pn(sx, ow, w, ox);
                    if pn == 0 {
                        break;
                    }
                    gather_1x1(conv, ximg, h, w, iy, ox, pn, sx, kn);
                    transpose_kn_nk(kn, nk, k_pad, pn);
                    gemm_panel(
                        conv,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        oy,
                        ox,
                        pn,
                        nk,
                        relu,
                        None,
                        0,
                        oc,
                    );
                    ox += pn;
                }
            }
        }
    }
}

fn conv1x1_pn(sx: usize, ow: usize, w: usize, ox: usize) -> usize {
    let max_pn = if sx == 1 {
        ow - ox
    } else {
        let last = w.saturating_sub(1) / sx;
        last.saturating_add(1).saturating_sub(ox)
    };
    if max_pn >= PN {
        PN
    } else if max_pn >= NR16 {
        NR16
    } else {
        max_pn.min(PN)
    }
}

fn gather_1x1(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    iy: usize,
    ox: usize,
    pn: usize,
    sx: usize,
    kn: &mut [i8],
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    if k_pad > ic {
        kn[ic * pn..k_pad * pn].fill(0);
    }
    for c in 0..ic {
        let dst = &mut kn[c * pn..c * pn + pn];
        if sx == 1 {
            let src0 = (c * h + iy) * w + ox;
            dst.copy_from_slice(&ximg[src0..src0 + pn]);
        } else if sx == 2 {
            let src0 = (c * h + iy) * w + ox * 2;
            copy_strided2(dst, &ximg[src0..]);
        } else {
            for t in 0..pn {
                dst[t] = ximg[(c * h + iy) * w + (ox + t) * sx];
            }
        }
    }
}

/// Gather every in-row 1×1 tile, zip once, then stream OC so the 4-row
/// weight panel stays hot (same schedule as 3×3 s1).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn conv1x1_row_zip(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    iy: usize,
    sx: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    kn: &mut [i8],
    zip: &mut [i8],
    relu: bool,
    k_pad: usize,
    oc: usize,
) {
    const MAX_TILES: usize = 256;
    let mut oxs = [0usize; MAX_TILES];
    let mut pns = [0usize; MAX_TILES];
    let mut zoff = [0usize; MAX_TILES];
    let mut ntiles = 0usize;
    let mut acc = 0usize;
    let mut ox = 0usize;
    while ox < ow && ntiles < MAX_TILES {
        let pn = conv1x1_pn(sx, ow, w, ox);
        if pn == 0 || !pn.is_multiple_of(NR16) {
            break;
        }
        gather_1x1(conv, ximg, h, w, iy, ox, pn, sx, kn);
        pack_kn_zip16(kn, &mut zip[acc..acc + k_pad * pn], k_pad, pn);
        oxs[ntiles] = ox;
        pns[ntiles] = pn;
        zoff[ntiles] = acc;
        acc += k_pad * pn;
        ntiles += 1;
        ox += pn;
    }
    let mut mo = 0usize;
    while mo < oc {
        let mr = (oc - mo).min(MR);
        if mr == MR {
            for t in 0..ntiles {
                let pn = pns[t];
                let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                let mut no = 0usize;
                while no + NR16 <= pn {
                    unsafe {
                        kernel_4x16_zip_store(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            oxs[t] + no,
                            mo,
                            z,
                            pn,
                            no,
                            relu,
                            None,
                        );
                    }
                    no += NR16;
                }
            }
        } else {
            for t in 0..ntiles {
                gather_1x1(conv, ximg, h, w, iy, oxs[t], pns[t], sx, kn);
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                    for ti in 0..pns[t] {
                        let mut a = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            a += i32::from(wv) * i32::from(kn[kk * pns[t] + ti]);
                        }
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            oxs[t] + ti,
                            mo + mi,
                            a,
                            relu,
                            None,
                        );
                    }
                }
            }
        }
        mo += mr;
    }
    // Leftover 1..15: one overlapping 4x16 so the 1x1 tail stays on SDOT.
    // gemm_panel 4x8 on this tail is the old spilled mxn kernel.
    if ow >= NR16 && ox < ow && zip.len() >= k_pad * NR16 {
        let ox0 = ow - NR16;
        let pn = NR16;
        gather_1x1(conv, ximg, h, w, iy, ox0, pn, sx, kn);
        pack_kn_zip16(kn, &mut zip[..k_pad * pn], k_pad, pn);
        let z = &zip[..k_pad * pn];
        let mut mo = 0usize;
        while mo < oc {
            let mr = (oc - mo).min(MR);
            if mr == MR {
                unsafe {
                    kernel_4x16_zip_store(
                        conv, yd, ybase, oh, ow, oy, ox0, mo, z, pn, 0, relu, None,
                    );
                }
            } else {
                gather_1x1(conv, ximg, h, w, iy, ox0, pn, sx, kn);
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                    for ti in 0..pn {
                        let mut a = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                        }
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            ox0 + ti,
                            mo + mi,
                            a,
                            relu,
                            None,
                        );
                    }
                }
            }
            mo += mr;
        }
        ox = ow;
    }
    while ox < ow {
        let pn = conv1x1_pn(sx, ow, w, ox);
        if pn == 0 {
            break;
        }
        gather_1x1(conv, ximg, h, w, iy, ox, pn, sx, kn);
        let nkk = k_pad * pn;
        transpose_kn_nk(&kn[..nkk], &mut zip[..nkk], k_pad, pn);
        gemm_panel(conv, yd, ybase, oh, ow, oy, ox, pn, zip, relu, None, 0, oc);
        ox += pn;
    }
}

fn conv_gather(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    tile: &mut [i8],
    relu: bool,
) {
    let (ic, k, stride, pad, k_pad) = (conv.ic, conv.k, conv.stride, conv.pad, conv.k_pad);
    let k_raw = ic * k * k;
    for ni in 0..n {
        let xbase = ni * ic * h * w;
        let ybase = ni * conv.oc * y.h * y.w;
        let ximg = &xq[xbase..xbase + ic * h * w];
        for oy in 0..y.h {
            for ox in 0..y.w {
                tile[..k_raw].fill(conv.act_zp);
                tile[k_raw..k_pad].fill(0);
                let iy0 = oy as isize * stride as isize - pad as isize;
                let ix0 = ox as isize * stride as isize - pad as isize;
                let mut kk = 0usize;
                for c in 0..ic {
                    for kh in 0..k {
                        let iy = iy0 + kh as isize;
                        for kw in 0..k {
                            let ix = ix0 + kw as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                tile[kk] = ximg[(c * h + iy as usize) * w + ix as usize];
                            }
                            kk += 1;
                        }
                    }
                }
                debug_assert_eq!(kk, k_raw);
                store_col(
                    conv,
                    &mut y.data,
                    ybase,
                    y.h,
                    y.w,
                    oy,
                    ox,
                    tile,
                    k_raw,
                    relu,
                    None,
                    0,
                    conv.oc,
                );
            }
        }
    }
}

