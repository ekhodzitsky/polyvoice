/// 3×3 s2 pad=1: three padded input rows, advance by two rows per output
/// row (one row is reused). Gather is a stride-2 pull from `[ic][w+2]`.
fn conv3x3_s2_rows(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    relu: bool,
    i8d: Option<I8Dest>,
) {
    let (ic, oc, zp) = (conv.ic, conv.oc, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    let oh = y.h;
    let ow = y.w;
    let kn_len = conv.k_pad.saturating_mul(PN);
    let zip_len = conv.k_pad.saturating_mul(ow.max(PN).min(ZIP_WAVE_PX));
    let threads = intra_threads();
    let macs = (oc as u64) * (ic as u64) * 9 * (oh as u64) * (ow as u64);
    // l3/l4 downsample (oc>=128) was serial even with set_intra(2). OC-split
    // re-gathers per worker; two extra intra::run per ResNet, not per-row.
    if threads > 1 && macs >= 20_000_000 && oc >= 128 && oc / MR >= 2 {
        let tcount = threads.min(oc / MR).max(1);
        if tcount > 1 {
            let y_ptr = y.data.as_mut_ptr() as usize;
            let y_len = y.data.len();
            for ni in 0..n {
                let xbase = ni * ic * h * w;
                let ybase = ni * oc * oh * ow;
                let ximg = &xq[xbase..xbase + ic * h * w];
                crate::intra::run(tcount, |t| {
                    let (a, b) = oc_bounds(oc, t, tcount);
                    if a >= b {
                        return;
                    }
                    let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
                    ROW3.with(|rc| {
                        PANEL_KN.with(|knc| {
                            PANEL_NK.with(|nkc| {
                                let mut rows = rc.borrow_mut();
                                let mut kn = knc.borrow_mut();
                                let mut nk = nkc.borrow_mut();
                                if kn.len() < kn_len {
                                    kn.resize(kn_len, 0);
                                }
                                if nk.len() < zip_len {
                                    nk.resize(zip_len, 0);
                                }
                                if rows.len() < 3 * plane {
                                    rows.resize(3 * plane, zp);
                                }
                                s2_scan(
                                    conv, ximg, h, w, yd, ybase, oh, ow, &mut rows, &mut kn,
                                    &mut nk, relu, i8d, a, b,
                                );
                            });
                        });
                    });
                });
            }
            return;
        }
    }
    ROW3.with(|rc| {
        PANEL_KN.with(|knc| {
            PANEL_NK.with(|nkc| {
                let mut rows = rc.borrow_mut();
                let mut kn = knc.borrow_mut();
                let mut nk = nkc.borrow_mut();
                if kn.len() < kn_len {
                    kn.resize(kn_len, 0);
                }
                if nk.len() < zip_len {
                    nk.resize(zip_len, 0);
                }
                if rows.len() < 3 * plane {
                    rows.resize(3 * plane, zp);
                }
                for ni in 0..n {
                    let xbase = ni * ic * h * w;
                    let ybase = ni * oc * oh * ow;
                    let ximg = &xq[xbase..xbase + ic * h * w];
                    s2_scan(
                        conv,
                        ximg,
                        h,
                        w,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        &mut rows,
                        &mut kn,
                        &mut nk,
                        relu,
                        i8d,
                        0,
                        oc,
                    );
                }
            });
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn s2_scan(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    rows: &mut [i8],
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let (ic, zp) = (conv.ic, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    // iy = -1 is out of range → pack_pad_row fills zp.
    pack_pad_row(ximg, ic, h, w, wp, zp, usize::MAX, &mut rows[0..plane]);
    pack_pad_row(ximg, ic, h, w, wp, zp, 0, &mut rows[plane..2 * plane]);
    pack_pad_row(ximg, ic, h, w, wp, zp, 1, &mut rows[2 * plane..3 * plane]);
    let mut top = 0usize;
    let mut mid = 1usize;
    let mut bot = 2usize;
    for oy in 0..oh {
        s2_scan_row_zip(
            conv,
            rows,
            plane,
            wp,
            [top, mid, bot],
            yd,
            ybase,
            oh,
            ow,
            oy,
            kn,
            nk,
            relu,
            i8d,
            oc0,
            oc1,
        );
        let next_mid = 2 * (oy + 1);
        let next_bot = next_mid + 1;
        top = bot;
        mid = (bot + 1) % 3;
        bot = (bot + 2) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            next_mid,
            &mut rows[mid * plane..(mid + 1) * plane],
        );
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            next_bot,
            &mut rows[bot * plane..(bot + 1) * plane],
        );
    }
}

fn gather_kn_s2(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    slots: [usize; 3],
    ox: usize,
    pn: usize,
    kn: &mut [i8],
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    let k_raw = ic * 9;
    let nkk = k_pad * pn;
    if k_pad > k_raw {
        kn[k_raw * pn..nkk].fill(0);
    }
    for c in 0..ic {
        for kh in 0..3 {
            let row_base = slots[kh] * plane + c * wp;
            for kw in 0..3 {
                let kidx = c * 9 + kh * 3 + kw;
                let dst = &mut kn[kidx * pn..kidx * pn + pn];
                copy_strided2(dst, &rows[row_base + ox * 2 + kw..]);
            }
        }
    }
}

/// Dest[i] = src[2*i]. 16-wide unzip on aarch64; scalar tail.
pub(crate) fn copy_strided2(dst: &mut [i8], src: &[i8]) {
    let n = dst.len();
    debug_assert!(src.len() >= n.saturating_mul(2).saturating_sub(1).max(n));
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vld1q_s8, vst1q_s8, vuzp1q_s8};
        // Two 16-wide vuzp1 (not vld2q — that lost on product). Need 64 src
        // bytes; skip when the padded row is a tight 2n-1 tail.
        while i + 32 <= n && 2 * i + 64 <= src.len() {
            unsafe {
                let p = src.as_ptr().add(2 * i);
                let d = dst.as_mut_ptr().add(i);
                vst1q_s8(d, vuzp1q_s8(vld1q_s8(p), vld1q_s8(p.add(16))));
                vst1q_s8(
                    d.add(16),
                    vuzp1q_s8(vld1q_s8(p.add(32)), vld1q_s8(p.add(48))),
                );
            }
            i += 32;
        }
        while i + 16 <= n && 2 * i + 32 <= src.len() {
            unsafe {
                let a = vld1q_s8(src.as_ptr().add(2 * i));
                let b = vld1q_s8(src.as_ptr().add(2 * i + 16));
                vst1q_s8(dst.as_mut_ptr().add(i), vuzp1q_s8(a, b));
            }
            i += 16;
        }
    }
    while i < n {
        dst[i] = src[2 * i];
        i += 1;
    }
}

/// Same zip-then-OC schedule as s1, with stride-2 gather from padded rows.
#[allow(clippy::too_many_arguments)]
fn s2_scan_row_zip(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    slots: [usize; 3],
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    kn: &mut [i8],
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let k_pad = conv.k_pad;
    let k_raw = conv.ic * 9;
    const WAVE: usize = 32;
    let mut oxs = [0usize; WAVE];
    let mut pns = [0usize; WAVE];
    let mut zoff = [0usize; WAVE];
    let mut ox = 0usize;
    let use_zip = {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            zip_ok(k_pad)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            false
        }
    };
    if !use_zip {
        while ox < ow {
            let pn = if ox + PN <= ow {
                PN
            } else if ox + NR16 <= ow {
                NR16
            } else {
                break;
            };
            gather_kn_s2(conv, rows, plane, wp, slots, ox, pn, kn);
            let nkk = k_pad * pn;
            transpose_kn_nk(&kn[..nkk], &mut zip[..nkk], k_pad, pn);
            gemm_panel(
                conv, yd, ybase, oh, ow, oy, ox, pn, zip, relu, i8d, oc0, oc1,
            );
            ox += pn;
        }
        while ox < ow {
            // Scalar tail: one output pixel, NK tile in kn[0..k_pad].
            gather_kn_s2(conv, rows, plane, wp, slots, ox, 1, kn);
            store_col(
                conv, yd, ybase, oh, ow, oy, ox, kn, k_raw, relu, i8d, oc0, oc1,
            );
            ox += 1;
        }
        return;
    }
    while ox < ow {
        let mut ntiles = 0usize;
        let mut acc = 0usize;
        while ox < ow && ntiles < WAVE {
            let pn = if ox + PN <= ow {
                PN
            } else if ox + NR16 <= ow {
                NR16
            } else {
                break;
            };
            oxs[ntiles] = ox;
            pns[ntiles] = pn;
            zoff[ntiles] = acc;
            acc += k_pad * pn;
            ntiles += 1;
            ox += pn;
        }
        if ntiles == 0 {
            while ox < ow {
                gather_kn_s2(conv, rows, plane, wp, slots, ox, 1, kn);
                store_col(
                    conv, yd, ybase, oh, ow, oy, ox, kn, k_raw, relu, i8d, oc0, oc1,
                );
                ox += 1;
            }
            break;
        }
        debug_assert!(zip.len() >= acc);
        for t in 0..ntiles {
            let pn = pns[t];
            gather_kn_s2(conv, rows, plane, wp, slots, oxs[t], pn, kn);
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            pack_kn_zip16(kn, &mut zip[zoff[t]..zoff[t] + k_pad * pn], k_pad, pn);
        }
        // Fat-K s2 only (layer-4 3x3 downsample, k_pad=1152). Narrower s2
        // stays W-hot OC-outer; k_pad>=512 on s2 already lost on product.
        let tile_outer = k_pad >= 1024;
        if tile_outer {
            for t in 0..ntiles {
                let pn = pns[t];
                let ox0 = oxs[t];
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                let mut mo = oc0;
                while mo < oc1 {
                    let mr = (conv.oc - mo).min(MR);
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_s2(conv, rows, plane, wp, slots, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
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
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                    mo += mr;
                }
            }
        } else {
            let mut mo = oc0;
            while mo < oc1 {
                let mr = (conv.oc - mo).min(MR);
                for t in 0..ntiles {
                    let pn = pns[t];
                    let ox0 = oxs[t];
                    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                    let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_s2(conv, rows, plane, wp, slots, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
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
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                }
                mo += mr;
            }
        }
    }
}

fn conv3x3(
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
    let (ic, oc) = (conv.ic, conv.oc);
    let (oh, ow, sx) = (y.h, y.w, conv.stride.max(1));
    let k_raw = ic * 9;
    for ni in 0..n {
        let xbase = ni * ic * h * w;
        let ybase = ni * oc * oh * ow;
        let ximg = &xq[xbase..xbase + ic * h * w];
        for oy in 0..oh {
            let iy_mid = oy * sx;
            let interior_y = iy_mid >= 1 && iy_mid + 1 < h;
            let mut ox = 0usize;
            if interior_y {
                while ox < 1 && ox < ow {
                    gather_one_3x3(conv, ximg, h, w, oy, ox, nk);
                    store_col(
                        conv,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        oy,
                        ox,
                        nk,
                        k_raw,
                        relu,
                        None,
                        0,
                        conv.oc,
                    );
                    ox += 1;
                }
                while ox + PN <= ow && last_ix(ox, PN, sx) < w {
                    implicit_tile_3x3(conv, ximg, y, ybase, h, w, oh, ow, oy, ox, PN, kn, nk, relu);
                    ox += PN;
                }
                while ox + NR16 <= ow && last_ix(ox, NR16, sx) < w {
                    implicit_tile_3x3(
                        conv, ximg, y, ybase, h, w, oh, ow, oy, ox, NR16, kn, nk, relu,
                    );
                    ox += NR16;
                }
                while ox + NR <= ow && last_ix(ox, NR, sx) < w {
                    implicit_tile_3x3(conv, ximg, y, ybase, h, w, oh, ow, oy, ox, NR, kn, nk, relu);
                    ox += NR;
                }
            }
            while ox < ow {
                gather_one_3x3(conv, ximg, h, w, oy, ox, nk);
                store_col(
                    conv,
                    &mut y.data,
                    ybase,
                    oh,
                    ow,
                    oy,
                    ox,
                    nk,
                    k_raw,
                    relu,
                    None,
                    0,
                    conv.oc,
                );
                ox += 1;
            }
        }
    }
}

