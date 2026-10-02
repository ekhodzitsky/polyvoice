/// 3×3 s1: keep three padded input rows packed as `[3][ic][w+2]` so the
/// channel stride is `w+2` instead of `h*w`.
///
/// Intra-op: late layers (oc ≥ 64) split *output channels* so each worker
/// keeps a small weight working set. Early 32-ch maps keep the existing
/// output-row split (tiny weights, huge spatial). Never spawn inside a tile.
#[allow(clippy::too_many_arguments)]
fn conv3x3_s1_rows(
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
    // Serial/oy-split: one WAVE slab (overwritten). Shared-OC still zips the
    // whole row so workers can stream W without re-gather.
    // +NR16: shared-OC leftover 1..15 is one overlapping 4x16 (same as serial).
    let zip_row = conv.k_pad.saturating_mul(ow.max(PN).saturating_add(NR16));
    let zip_len = conv.k_pad.saturating_mul(ow.max(PN).min(ZIP_WAVE_PX));
    let threads = intra_threads();
    // MAC estimate: skip the pool when spawn-equivalent work is tiny (stem).
    let macs = (oc as u64) * (ic as u64) * 9 * (oh as u64) * (ow as u64);
    if threads > 1 && macs >= 20_000_000 {
        let y_ptr = y.data.as_mut_ptr() as usize;
        let y_len = y.data.len();
        // Wide OC: zip the row once, then split only the SDOT stream so
        // workers do not re-gather. Narrow OC / fat maps: split output rows.
        let split_oc = oc >= 128 && oc / MR >= 2;
        let tcount = if split_oc {
            threads.min(oc / MR).max(1)
        } else {
            threads.min(oh.max(1))
        };
        if tcount > 1 && split_oc {
            ROW3.with(|rc| {
                PANEL_KN.with(|knc| {
                    PANEL_NK.with(|nkc| {
                        let mut rows = rc.borrow_mut();
                        let mut kn = knc.borrow_mut();
                        let mut nk = nkc.borrow_mut();
                        if kn.len() < kn_len {
                            kn.resize(kn_len, 0);
                        }
                        let map_len = zip_row.saturating_mul(oh.max(1));
                        if nk.len() < map_len {
                            nk.resize(map_len, 0);
                        }
                        if rows.len() < 3 * plane {
                            rows.resize(3 * plane, zp);
                        }
                        for ni in 0..n {
                            let xbase = ni * ic * h * w;
                            let ybase = ni * oc * oh * ow;
                            let ximg = &xq[xbase..xbase + ic * h * w];
                            s1_scan(
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
                                0,
                                oh,
                                tcount,
                            );
                        }
                    });
                });
            });
            return;
        }
        if tcount > 1 {
            for ni in 0..n {
                let xbase = ni * ic * h * w;
                let ybase = ni * oc * oh * ow;
                let ximg = &xq[xbase..xbase + ic * h * w];
                crate::intra::run(tcount, |t| {
                    let chunk = oh.div_ceil(tcount);
                    let oy0 = t * chunk;
                    let oy1 = (oy0 + chunk).min(oh);
                    if oy0 >= oy1 {
                        return;
                    }
                    // SAFETY: oy-split writes disjoint row ranges. The pool
                    // joins before return.
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
                                s1_scan(
                                    conv, ximg, h, w, yd, ybase, oh, ow, &mut rows, &mut kn,
                                    &mut nk, relu, i8d, 0, oc, oy0, oy1, 0,
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
                    s1_scan(
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
                        0,
                        oh,
                        0,
                    );
                }
            });
        });
    });
}

fn oc_bounds(oc: usize, t: usize, n: usize) -> (usize, usize) {
    let groups = oc / MR;
    let base = groups / n;
    let extra = groups % n;
    let g0 = t * base + t.min(extra);
    let g1 = g0 + base + if t < extra { 1 } else { 0 };
    let lo = g0 * MR;
    let hi = if t + 1 == n { oc } else { g1 * MR };
    (lo, hi)
}

#[allow(clippy::too_many_arguments)]
fn s1_scan(
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
    oy0: usize,
    oy1: usize,
    oc_par: usize,
) {
    let (ic, zp) = (conv.ic, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    let k_raw = ic * 9;
    for kh in 0..2 {
        let iy = oy0.wrapping_add(kh).wrapping_sub(1);
        let slot = (oy0 + kh) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            iy,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let row_zip = zip_ok(conv.k_pad);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let row_zip = false;
    if oc_par > 1 && row_zip {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            s1_scan_shared_oc(
                conv, ximg, h, w, yd, ybase, oh, ow, rows, kn, nk, relu, i8d, oy0, oy1, oc_par,
                plane, wp, zp,
            );
            return;
        }
    }
    for oy in oy0..oy1 {
        let slot = (oy + 2) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            oy + 1,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
        // A pn-wide tile at `ox` reads padded x in [ox, ox+pn+1]. The row
        // is `w+2` wide, so ox+pn <= w. For 3x3 s1 pad=1, ow == w.
        if row_zip {
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            s1_scan_row_zip(
                conv, rows, plane, wp, yd, ybase, oh, ow, oy, kn, nk, relu, i8d, oc0, oc1,
            );
        } else {
            let mut ox = 0usize;
            while ox + PN <= ow {
                implicit_tile_from_rows(
                    conv, rows, plane, wp, yd, ybase, oh, ow, oy, ox, PN, kn, nk, relu, i8d, oc0,
                    oc1,
                );
                ox += PN;
            }
            while ox + NR16 <= ow {
                implicit_tile_from_rows(
                    conv, rows, plane, wp, yd, ybase, oh, ow, oy, ox, NR16, kn, nk, relu, i8d, oc0,
                    oc1,
                );
                ox += NR16;
            }
            while ox < ow {
                gather_one_from_rows(conv, rows, plane, wp, oy, ox, nk);
                store_col(
                    conv, yd, ybase, oh, ow, oy, ox, nk, k_raw, relu, i8d, oc0, oc1,
                );
                ox += 1;
            }
        }
    }
}

/// Gather every in-row tile, zip once, then stream OC across tiles so the
/// 4-row weight panel stays hot. Bound is `ox+pn <= ow` (right halo fits).
/// Zip every output row once, then one intra-op OC stream. Per-row pool
/// wakeups cost more than a T=400 layer-3 row; one dispatch amortizes.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn s1_scan_shared_oc(
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
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oy0: usize,
    oy1: usize,
    oc_par: usize,
    plane: usize,
    wp: usize,
    zp: i8,
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    let k_raw = ic * 9;
    let row_bytes = k_pad.saturating_mul(ow.max(PN).saturating_add(NR16));
    let nrows = oy1.saturating_sub(oy0);
    let need = row_bytes.saturating_mul(nrows.max(1));
    if zip.len() < need {
        // Caller should have reserved the map; fall back to serial waves.
        for oy in oy0..oy1 {
            let slot = (oy + 2) % 3;
            pack_pad_row(
                ximg,
                ic,
                h,
                w,
                wp,
                zp,
                oy + 1,
                &mut rows[slot * plane..(slot + 1) * plane],
            );
            s1_scan_row_zip(
                conv, rows, plane, wp, yd, ybase, oh, ow, oy, kn, zip, relu, i8d, 0, conv.oc,
            );
        }
        return;
    }
    const MAX_TILES: usize = 256;
    let mut oxs = [0usize; MAX_TILES];
    let mut pns = [0usize; MAX_TILES];
    let mut zoff = [0usize; MAX_TILES];
    let mut ntiles = 0usize;
    let mut acc = 0usize;
    let mut ox = 0usize;
    while ox < ow && ntiles < MAX_TILES {
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
    // Leftover 1..15: one overlapping 4x16 so intra matches serial waves.
    if ow >= NR16 && ox < ow && ntiles < MAX_TILES {
        let ox0 = ow - NR16;
        oxs[ntiles] = ox0;
        pns[ntiles] = NR16;
        zoff[ntiles] = acc;
        acc += k_pad * NR16;
        ntiles += 1;
        ox = ow;
    }
    let tail_ox = ox;
    if ntiles == 0 {
        for oy in oy0..oy1 {
            let slot = (oy + 2) % 3;
            pack_pad_row(
                ximg,
                ic,
                h,
                w,
                wp,
                zp,
                oy + 1,
                &mut rows[slot * plane..(slot + 1) * plane],
            );
            let mut x = tail_ox;
            while x < ow {
                gather_one_from_rows(conv, rows, plane, wp, oy, x, kn);
                store_col(
                    conv, yd, ybase, oh, ow, oy, x, kn, k_raw, relu, i8d, 0, conv.oc,
                );
                x += 1;
            }
        }
        return;
    }
    for kh in 0..2 {
        let iy = oy0.wrapping_add(kh).wrapping_sub(1);
        let slot = (oy0 + kh) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            iy,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
    }
    for oy in oy0..oy1 {
        let slot = (oy + 2) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            oy + 1,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
        let dest = &mut zip[(oy - oy0) * row_bytes..(oy - oy0) * row_bytes + acc];
        for t in 0..ntiles {
            let pn = pns[t];
            gather_zip_from_rows(
                conv,
                rows,
                plane,
                wp,
                oy,
                oxs[t],
                pn,
                kn,
                &mut dest[zoff[t]..zoff[t] + k_pad * pn],
            );
        }
        let mut x = tail_ox;
        while x < ow {
            gather_one_from_rows(conv, rows, plane, wp, oy, x, kn);
            store_col(
                conv, yd, ybase, oh, ow, oy, x, kn, k_raw, relu, i8d, 0, conv.oc,
            );
            x += 1;
        }
    }
    let y_ptr = yd.as_mut_ptr() as usize;
    let y_len = yd.len();
    let zip_ptr = zip.as_ptr() as usize;
    crate::intra::run(oc_par, |t| {
        let (a, b) = oc_bounds(conv.oc, t, oc_par);
        if a >= b {
            return;
        }
        let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
        let zip = unsafe { std::slice::from_raw_parts(zip_ptr as *const i8, need) };
        for oy in oy0..oy1 {
            let rowz = &zip[(oy - oy0) * row_bytes..(oy - oy0) * row_bytes + acc];
            let mut mo = a;
            while mo < b {
                let mr = (conv.oc - mo).min(MR);
                if mr != MR {
                    mo += mr;
                    continue;
                }
                for ti in 0..ntiles {
                    let pn = pns[ti];
                    let z = &rowz[zoff[ti]..zoff[ti] + k_pad * pn];
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
                                oxs[ti] + no,
                                mo,
                                z,
                                pn,
                                no,
                                relu,
                                i8d,
                            );
                        }
                        no += NR16;
                    }
                }
                mo += mr;
            }
        }
    });
    // Leftover OC groups (oc not multiple of 4) stay serial.
    let rem = conv.oc % MR;
    if rem != 0 {
        let mo0 = conv.oc - rem;
        for oy in oy0..oy1 {
            let rowz = &zip[(oy - oy0) * row_bytes..(oy - oy0) * row_bytes + acc];
            for ti in 0..ntiles {
                let pn = pns[ti];
                let ox0 = oxs[ti];
                gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
                for mi in 0..rem {
                    let wr = &conv.q_w_pad[(mo0 + mi) * k_pad..(mo0 + mi + 1) * k_pad];
                    for tcol in 0..pn {
                        let mut accu = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            accu += i32::from(wv) * i32::from(kn[kk * pn + tcol]);
                        }
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            ox0 + tcol,
                            mo0 + mi,
                            accu,
                            relu,
                            i8d,
                        );
                    }
                }
                let _ = rowz;
            }
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn s1_scan_row_zip(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
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
    // Waves of tiles so a long row (T=2000/4100) stays on SDOT; 32 tiles
    // is 1024 px and keeps the zip panel around one L2-sized slab.
    const WAVE: usize = 32;
    let mut oxs = [0usize; WAVE];
    let mut pns = [0usize; WAVE];
    let mut zoff = [0usize; WAVE];
    let mut ox = 0usize;
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
            // Leftover 1..15: one overlapping 4x16 tile (reuses zip[0..16*k_pad]).
            // 8..15 already kept; this extends to the 4-wide l3 tail (ow=100).
            if ow >= NR16 && ox < ow && zip.len() >= k_pad * NR16 {
                let ox0 = ow - NR16;
                let pn = NR16;
                gather_zip_from_rows(
                    conv,
                    rows,
                    plane,
                    wp,
                    oy,
                    ox0,
                    pn,
                    kn,
                    &mut zip[..k_pad * pn],
                );
                let z = &zip[..k_pad * pn];
                let mut mo = oc0;
                while mo < oc1 {
                    let mr = (conv.oc - mo).min(MR);
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
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
                            gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
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
                ox = ow;
            }
            while ox < ow {
                gather_one_from_rows(conv, rows, plane, wp, oy, ox, kn);
                store_col(
                    conv, yd, ybase, oh, ow, oy, ox, kn, k_raw, relu, i8d, oc0, oc1,
                );
                ox += 1;
            }
            break;
        }
        debug_assert!(zip.len() >= acc);
        // Fat K (layer 3/4): gather+pack+OC one tile so the 36–70 KiB A slab
        // stays in L1. Packing the whole WAVE first evicts it (WAVE=32 of
        // k_pad=1152 is ~1 MiB). WAVE stays 32; this is not fat-K WAVE=1.
        // Narrow K still zips the wave, then streams W-hot OC-outer.
        let tile_outer = k_pad >= 1024;
        if tile_outer {
            for t in 0..ntiles {
                let pn = pns[t];
                gather_zip_from_rows(
                    conv,
                    rows,
                    plane,
                    wp,
                    oy,
                    oxs[t],
                    pn,
                    kn,
                    &mut zip[zoff[t]..zoff[t] + k_pad * pn],
                );
                let ox0 = oxs[t];
                let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                let mut mo = oc0;
                while mo < oc1 {
                    let mr = (conv.oc - mo).min(MR);
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
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
                            gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
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
            for t in 0..ntiles {
                let pn = pns[t];
                gather_zip_from_rows(
                    conv,
                    rows,
                    plane,
                    wp,
                    oy,
                    oxs[t],
                    pn,
                    kn,
                    &mut zip[zoff[t]..zoff[t] + k_pad * pn],
                );
            }
            let mut mo = oc0;
            while mo < oc1 {
                let mr = (conv.oc - mo).min(MR);
                for t in 0..ntiles {
                    let pn = pns[t];
                    let ox0 = oxs[t];
                    let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
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
                            gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
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

fn pack_pad_row(
    ximg: &[i8],
    ic: usize,
    h: usize,
    w: usize,
    wp: usize,
    zp: i8,
    iy: usize,
    dst: &mut [i8],
) {
    if iy >= h {
        dst.fill(zp);
        return;
    }
    for c in 0..ic {
        let d = &mut dst[c * wp..c * wp + wp];
        d[0] = zp;
        d[1..1 + w].copy_from_slice(&ximg[(c * h + iy) * w..(c * h + iy) * w + w]);
        d[1 + w] = zp;
    }
}

fn row_slot(oy: usize, kh: usize) -> usize {
    // rows packed: before loop, slot 0 = iy=-1, slot 1 = iy=0;
    // each oy packs iy=oy+1 into slot (oy+2)%3.
    // iy = oy + kh - 1 lives in slot (oy + kh) % 3?
    // oy=0, kh=0, iy=-1 → slot 0. (0+0)%3=0.
    // oy=0, kh=1, iy=0 → slot 1.
    // oy=0, kh=2, iy=1 → packed this iter into slot 2.
    // oy=1, kh=0, iy=0 → slot 1. (1+0)%3=1.
    // oy=1, kh=1, iy=1 → slot 2.
    // oy=1, kh=2, iy=2 → packed into slot (1+2)%3=0. Yes.
    (oy + kh) % 3
}

fn gather_one_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    oy: usize,
    ox: usize,
    tile: &mut [i8],
) {
    let (ic, k_pad, zp) = (conv.ic, conv.k_pad, conv.act_zp);
    let k_raw = ic * 9;
    tile[..k_raw].fill(zp);
    tile[k_raw..k_pad].fill(0);
    for c in 0..ic {
        for kh in 0..3 {
            let rs = row_slot(oy, kh) * plane + c * wp + ox;
            for kw in 0..3 {
                tile[c * 9 + kh * 3 + kw] = rows[rs + kw];
            }
        }
    }
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    target_vendor = "apple",
    not(apple_accelerate)
))]
thread_local! {
    static REFERENCE_PACK: Cell<bool> = const { Cell::new(false) };
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn gather_zip_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
    zip: &mut [i8],
) {
    #[cfg(all(
        test,
        target_arch = "aarch64",
        target_vendor = "apple",
        not(apple_accelerate)
    ))]
    if REFERENCE_PACK.with(Cell::get) {
        gather_kn_from_rows(conv, rows, plane, wp, oy, ox, pn, kn);
        pack_kn_zip16(kn, zip, conv.k_pad, pn);
        return;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(apple_accelerate)
    ))]
    {
        let _ = kn;
        gather_zip_direct(conv, rows, plane, wp, oy, ox, pn, zip);
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(apple_accelerate)
    )))]
    {
        gather_kn_from_rows(conv, rows, plane, wp, oy, ox, pn, kn);
        pack_kn_zip16(kn, zip, conv.k_pad, pn);
    }
}

// Load four input taps directly into the SDOT panel layout, avoiding the
// intermediate K-by-N copy. Keep the established packer on other backends.
#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(apple_accelerate)
))]
fn gather_zip_direct(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    zip: &mut [i8],
) {
    use std::arch::aarch64::{int8x16x4_t, vdupq_n_s8, vld1q_s8, vst4q_s8};
    assert!(pn > 0 && pn.is_multiple_of(16));
    let zip = &mut zip[..conv.k_pad * pn];
    for (group, block) in zip.chunks_exact_mut(4 * pn).enumerate() {
        let taps: [Option<&[i8]>; 4] = std::array::from_fn(|lane| {
            let k = group * 4 + lane;
            if k >= conv.ic * 9 {
                return None;
            }
            let (c, kh, kw) = (k / 9, k % 9 / 3, k % 3);
            let base = row_slot(oy, kh) * plane + c * wp + ox + kw;
            Some(&rows[base..base + pn])
        });
        for (tile, dst) in block.chunks_exact_mut(64).enumerate() {
            // Each present tap is pn bytes and tile*16+16 <= pn. Each
            // destination chunk is exactly 64 bytes, the size of four vectors.
            unsafe {
                let v = taps.map(|tap| match tap {
                    Some(src) => vld1q_s8(src.as_ptr().add(tile * 16)),
                    None => vdupq_n_s8(0),
                });
                vst4q_s8(dst.as_mut_ptr(), int8x16x4_t(v[0], v[1], v[2], v[3]));
            }
        }
    }
}

fn gather_kn_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    let k_raw = ic * 9;
    let nkk = k_pad * pn;
    // k_raw taps are overwritten; only the 16-wide K tail needs zeros.
    if k_pad > k_raw {
        kn[k_raw * pn..nkk].fill(0);
    }
    for c in 0..ic {
        for kh in 0..3 {
            let base = row_slot(oy, kh) * plane + c * wp + ox;
            for kw in 0..3 {
                let kidx = c * 9 + kh * 3 + kw;
                kn[kidx * pn..kidx * pn + pn].copy_from_slice(&rows[base + kw..base + kw + pn]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn implicit_tile_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    gather_kn_from_rows(conv, rows, plane, wp, oy, ox, pn, kn);
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if zip_ok(conv.k_pad) && pn.is_multiple_of(NR16) {
        gemm_panel_kn16(
            conv, yd, ybase, oh, ow, oy, ox, pn, kn, nk, relu, i8d, oc0, oc1,
        );
        return;
    }
    let k_pad = conv.k_pad;
    let nkk = k_pad * pn;
    transpose_kn_nk(&kn[..nkk], &mut nk[..nkk], k_pad, pn);
    gemm_panel(conv, yd, ybase, oh, ow, oy, ox, pn, nk, relu, i8d, oc0, oc1);
}

