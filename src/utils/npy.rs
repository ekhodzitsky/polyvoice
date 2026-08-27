//! Minimal NPY v1.0/v2.0 reader (C-order, '<f4', '<f8' or '<i8').
//!
//! Shared by the VBx PLDA parameter files (`clusterer::plda`) and the AS-norm
//! imposter cohort (`clusterer::asnorm`). Avoids an ndarray-npy dependency
//! (which pins an older ndarray major) by parsing the small, fixed NPY format
//! we control. Supports 1-D and 2-D little-endian arrays in C (row-major)
//! order — exactly what the offline parameter dumps emit. Pure `std`, no
//! ONNX, wasm32-clean.

use std::path::Path;

/// Errors reading an `.npy` file.
#[derive(Debug, thiserror::Error)]
pub(crate) enum NpyError {
    #[error("npy io error on {path}: {detail}")]
    Io { path: String, detail: String },
}

/// PLDA dumps are ~265 KiB; AS-norm cohorts are a few MiB. Reject huge
/// files before `read` so a local `--vbx-plda-dir` cannot OOM the process.
const MAX_NPY_BYTES: u64 = 8 * 1024 * 1024;

fn read_file(path: &Path) -> Result<Vec<u8>, NpyError> {
    let meta = std::fs::metadata(path).map_err(|e| NpyError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    if meta.len() > MAX_NPY_BYTES {
        return Err(NpyError::Io {
            path: path.display().to_string(),
            detail: format!("npy file is {} bytes; max is {MAX_NPY_BYTES}", meta.len()),
        });
    }
    std::fs::read(path).map_err(|e| NpyError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

/// Parse an NPY header, returning `(dtype, shape, data_offset)`.
fn parse_npy_header(bytes: &[u8], path: &Path) -> Result<(String, Vec<usize>, usize), NpyError> {
    let bad = |detail: &str| NpyError::Io {
        path: path.display().to_string(),
        detail: detail.to_string(),
    };
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(bad("not an NPY file"));
    }
    let major = bytes[6];
    // v1.0: 2-byte header len; v2.0+: 4-byte. We emit v1.0 but accept both.
    let (header_len, header_start) = if major >= 2 {
        let len_bytes: [u8; 4] = bytes
            .get(8..12)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| bad("truncated NPY header"))?;
        (u32::from_le_bytes(len_bytes) as usize, 12usize)
    } else {
        let l = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (l, 10usize)
    };
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_start + header_len)
            .ok_or_else(|| bad("truncated NPY header"))?,
    )
    .map_err(|_| bad("non-utf8 NPY header"))?;

    let descr = extract_between(header, "'descr':", ',')
        .or_else(|| extract_between(header, "\"descr\":", ','))
        .ok_or_else(|| bad("no descr in NPY header"))?
        .trim()
        .trim_matches(['\'', '"', ' '])
        .to_string();

    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err(bad("fortran-order NPY unsupported"));
    }

    let shape_str = extract_between(header, "'shape':", ')')
        .or_else(|| extract_between(header, "\"shape\":", ')'))
        .ok_or_else(|| bad("no shape in NPY header"))?;
    let shape: Vec<usize> = shape_str
        .trim_start_matches([' ', '('])
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() { None } else { t.parse().ok() }
        })
        .collect();

    Ok((descr, shape, header_start + header_len))
}

/// Slice of `s` after `key` up to (not including) the next `end` char.
fn extract_between(s: &str, key: &str, end: char) -> Option<String> {
    let start = s.find(key)? + key.len();
    let rest = &s[start..];
    let stop = rest.find(end)?;
    Some(rest[..stop].to_string())
}

/// Read an NPY file as flat f64 values plus its shape. '<f4' and '<i8' widen
/// to f64 exactly; the element count is whatever the payload holds (shape is
/// NOT cross-checked here — callers that reshape validate it themselves).
pub(crate) fn read_npy_flat(path: &Path) -> Result<(Vec<f64>, Vec<usize>), NpyError> {
    let bytes = read_file(path)?;
    let (descr, shape, offset) = parse_npy_header(&bytes, path)?;
    let data = &bytes[offset..];
    let elem_size = match descr.as_str() {
        "<f8" | "<i8" => 8,
        "<f4" => 4,
        other => {
            return Err(NpyError::Io {
                path: path.display().to_string(),
                detail: format!("unsupported NPY dtype {other} (expected <f4, <f8 or <i8)"),
            });
        }
    };
    // A payload whose length is not a whole number of elements is truncated
    // or carries trailing garbage — reject it rather than silently dropping
    // the tail bytes.
    if data.len() % elem_size != 0 {
        return Err(NpyError::Io {
            path: path.display().to_string(),
            detail: format!(
                "NPY payload of {} bytes is not a multiple of the {elem_size}-byte element size ({descr})",
                data.len()
            ),
        });
    }
    let values: Vec<f64> = match descr.as_str() {
        "<f8" => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        "<f4" => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect(),
        "<i8" => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f64)
            .collect(),
        _ => unreachable!("elem_size match above rejects other dtypes"),
    };
    Ok((values, shape))
}

/// Read a 2-D NPY matrix as row-major f32 values, returning
/// `(values, rows, cols)`. '<f8' payloads narrow to f32 (the cohort files we
/// ship are '<f4', so no precision is lost on the intended path).
pub(crate) fn read_npy_f32_2d(path: &Path) -> Result<(Vec<f32>, usize, usize), NpyError> {
    let (values, shape) = read_npy_flat(path)?;
    if shape.len() != 2 {
        return Err(NpyError::Io {
            path: path.display().to_string(),
            detail: format!("expected a 2-D NPY matrix, got shape {shape:?}"),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    if values.len() != rows * cols {
        return Err(NpyError::Io {
            path: path.display().to_string(),
            detail: format!(
                "NPY payload holds {} values but shape {shape:?} needs {}",
                values.len(),
                rows * cols
            ),
        });
    }
    Ok((values.iter().map(|&v| v as f32).collect(), rows, cols))
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal NPY v1.0 file (C-order) with the given dtype/shape/payload.
    fn write_npy_v1(
        dir: &std::path::Path,
        name: &str,
        descr: &str,
        shape: &[usize],
        data: &[u8],
    ) -> std::path::PathBuf {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        };
        let header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    fn f8_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f4_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn read_npy_flat_f8_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write_npy_v1(tmp.path(), "a.npy", "<f8", &[2], &f8_bytes(&[1.5, -2.25]));
        let (values, shape) = read_npy_flat(&path).unwrap();
        assert_eq!(values, &[1.5, -2.25]);
        assert_eq!(shape, &[2]);
    }

    #[test]
    fn read_npy_flat_f4_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write_npy_v1(
            tmp.path(),
            "m.npy",
            "<f4",
            &[2, 2],
            &f4_bytes(&[1.0, 2.0, 3.0, 4.0]),
        );
        let (values, shape) = read_npy_flat(&path).unwrap();
        assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn read_npy_i8_widens_to_f64() {
        let tmp = tempfile::TempDir::new().unwrap();
        let data: Vec<u8> = [1i64, -2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = write_npy_v1(tmp.path(), "i.npy", "<i8", &[3], &data);
        let (values, _) = read_npy_flat(&path).unwrap();
        assert_eq!(values, &[1.0, -2.0, 3.0]);
    }

    #[test]
    fn read_npy_accepts_v2_header() {
        let tmp = tempfile::TempDir::new().unwrap();
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }";
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&f8_bytes(&[42.0]));
        let path = tmp.path().join("v2.npy");
        std::fs::write(&path, &bytes).unwrap();
        let (values, _) = read_npy_flat(&path).unwrap();
        assert_eq!(values, &[42.0]);
    }

    #[test]
    fn read_npy_rejects_bad_magic_and_short_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let short = tmp.path().join("short.npy");
        std::fs::write(&short, b"\x93NUM").unwrap();
        let err = read_npy_flat(&short).unwrap_err();
        assert!(format!("{err}").contains("not an NPY file"), "{err}");

        let bad = tmp.path().join("bad.npy");
        std::fs::write(&bad, b"NOTNUMPY!!rest of the file").unwrap();
        let err = read_npy_flat(&bad).unwrap_err();
        assert!(format!("{err}").contains("not an NPY file"), "{err}");
    }

    #[test]
    fn read_npy_rejects_truncated_header() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("trunc.npy");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&500u16.to_le_bytes()); // claims a 500-byte header
        bytes.extend_from_slice(b"{'descr': '<f8'"); // but the file ends early
        std::fs::write(&path, &bytes).unwrap();
        let err = read_npy_flat(&path).unwrap_err();
        assert!(format!("{err}").contains("truncated NPY header"), "{err}");
    }

    #[test]
    fn read_npy_rejects_truncated_v2_header() {
        // A v2.0 file whose 4-byte header length is cut short must error, not
        // panic on the length read.
        let tmp = tempfile::TempDir::new().unwrap();
        for name in ["v2-10.npy", "v2-11.npy"] {
            let path = tmp.path().join(name);
            let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
            bytes.extend_from_slice(&[0x10, 0x00]); // only half the length field
            if name == "v2-11.npy" {
                bytes.push(0x00);
            }
            std::fs::write(&path, &bytes).unwrap();
            let err = read_npy_flat(&path).unwrap_err();
            assert!(
                format!("{err}").contains("truncated NPY header"),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn read_npy_rejects_ragged_payload() {
        // One whole <f4 element plus two trailing bytes: the tail must be
        // rejected, not silently dropped.
        let tmp = tempfile::TempDir::new().unwrap();
        let mut payload = f4_bytes(&[1.0]);
        payload.extend_from_slice(&[0xAA, 0xBB]);
        let path = write_npy_v1(tmp.path(), "ragged.npy", "<f4", &[1], &payload);
        let err = read_npy_flat(&path).unwrap_err();
        assert!(format!("{err}").contains("not a multiple"), "{err}");
    }

    #[test]
    fn read_npy_rejects_fortran_order() {
        let tmp = tempfile::TempDir::new().unwrap();
        let header = "{'descr': '<f8', 'fortran_order': True, 'shape': (1,), }";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&f8_bytes(&[1.0]));
        let path = tmp.path().join("f.npy");
        std::fs::write(&path, &bytes).unwrap();
        let err = read_npy_flat(&path).unwrap_err();
        assert!(format!("{err}").contains("fortran-order"), "{err}");
    }

    #[test]
    fn read_npy_rejects_file_over_size_cap() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("huge.npy");
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(MAX_NPY_BYTES + 1).unwrap();
        drop(f);
        let err = read_npy_flat(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("max is"), "{msg}");
        assert!(msg.contains(&MAX_NPY_BYTES.to_string()), "{msg}");
    }

    #[test]
    fn read_npy_rejects_unsupported_dtype() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write_npy_v1(tmp.path(), "u.npy", "<u1", &[2], &[1, 2]);
        let err = read_npy_flat(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unsupported NPY dtype <u1"), "{msg}");
    }

    #[test]
    fn read_npy_rejects_headers_missing_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let no_descr = "{'fortran_order': False, 'shape': (1,), }";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(no_descr.len() as u16).to_le_bytes());
        bytes.extend_from_slice(no_descr.as_bytes());
        bytes.extend_from_slice(&f8_bytes(&[1.0]));
        let path = tmp.path().join("nod.npy");
        std::fs::write(&path, &bytes).unwrap();
        let err = read_npy_flat(&path).unwrap_err();
        assert!(format!("{err}").contains("no descr"), "{err}");

        let no_shape = "{'descr': '<f8', 'fortran_order': False, }";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(no_shape.len() as u16).to_le_bytes());
        bytes.extend_from_slice(no_shape.as_bytes());
        bytes.extend_from_slice(&f8_bytes(&[1.0]));
        let path = tmp.path().join("nos.npy");
        std::fs::write(&path, &bytes).unwrap();
        let err = read_npy_flat(&path).unwrap_err();
        assert!(format!("{err}").contains("no shape"), "{err}");
    }

    #[test]
    fn read_npy_missing_file_reports_path() {
        let path = std::path::Path::new("/no/such/dir/matrix.npy");
        let err = read_npy_flat(path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("/no/such/dir/matrix.npy"), "{msg}");
    }

    #[test]
    fn read_npy_f32_2d_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write_npy_v1(
            tmp.path(),
            "m.npy",
            "<f4",
            &[2, 3],
            &f4_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let (values, rows, cols) = read_npy_f32_2d(&path).unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn read_npy_f32_2d_rejects_non_2d_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write_npy_v1(
            tmp.path(),
            "v.npy",
            "<f8",
            &[3],
            &f8_bytes(&[1.0, 2.0, 3.0]),
        );
        let err = read_npy_f32_2d(&path).unwrap_err();
        assert!(format!("{err}").contains("2-D"), "{err}");
    }

    #[test]
    fn read_npy_f32_2d_rejects_data_shape_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Header claims (2, 2) but only three values follow.
        let path = write_npy_v1(
            tmp.path(),
            "m.npy",
            "<f8",
            &[2, 2],
            &f8_bytes(&[1.0, 2.0, 3.0]),
        );
        let err = read_npy_f32_2d(&path).unwrap_err();
        assert!(format!("{err}").contains("payload"), "{err}");
    }

    #[test]
    fn extract_between_slices_between_key_and_end() {
        let h = "{'shape': (2, 3), }";
        assert_eq!(extract_between(h, "'shape':", ')').unwrap(), " (2, 3");
        assert!(extract_between(h, "'missing':", ')').is_none());
        // Key present but the end char never comes.
        assert!(extract_between("abc key: value", "key:", '!').is_none());
    }
}
