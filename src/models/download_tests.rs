use super::*;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

const TEST_BYTES: &[u8] = b"polyvoice";

/// Compute the expected SHA-256 of `TEST_BYTES` at test time, so the test is
/// robust against typos in a hardcoded constant.
fn test_bytes_sha256() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(TEST_BYTES);
    format!("{:x}", h.finalize())
}

#[test]
fn verify_existing_file_passes_when_hash_matches() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.bin");
    fs::write(&path, TEST_BYTES).unwrap();
    verify_sha256(&path, &test_bytes_sha256()).expect("hash must match");
}

#[test]
fn verify_existing_file_fails_when_hash_differs() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.bin");
    fs::write(&path, b"different content").unwrap();
    let err = verify_sha256(&path, &test_bytes_sha256()).expect_err("must mismatch");
    assert!(matches!(err, DownloadError::ChecksumMismatch { .. }));
}

#[test]
#[cfg_attr(miri, ignore)]
fn verify_streams_large_file_without_loading_into_ram() {
    // Write a 5 MB file; verify_sha256 must use streaming reader, not Vec::read_to_end.
    // The test passes purely if it doesn't OOM and computes a deterministic hash.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("big.bin");
    let mut f = fs::File::create(&path).unwrap();
    for _ in 0..5 * 1024 {
        // 5 MB of '\0'
        f.write_all(&[0u8; 1024]).unwrap();
    }
    // SHA-256 of 5 MB of zero bytes:
    let expected = sha256_of_zeros_5mb();
    verify_sha256(&path, &expected).expect("streaming hash should match");
}

fn sha256_of_zeros_5mb() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for _ in 0..5 * 1024 {
        h.update([0u8; 1024]);
    }
    format!("{:x}", h.finalize())
}

#[test]
#[cfg_attr(miri, ignore)]
fn download_with_checksum_no_signature_fallback() {
    // When signature is None and the file is already cached with a matching
    // hash, download_with_checksum_and_signature must take the cache-hit
    // path and return Ok(false) without touching the network.
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("cached.bin");
    fs::write(&dest, TEST_BYTES).unwrap();
    let sha = test_bytes_sha256();

    // A completely invalid URL proves we never reach the download path.
    let result = download_with_checksum_and_signature(
        "http://[invalid:definitely:not:a:real:url]",
        &sha,
        None,
        &dest,
    );
    assert!(
        result.is_ok(),
        "fallback should succeed: {:?}",
        result.err()
    );
    assert!(!result.unwrap(), "should be cached (no download)");

    // Calling the old wrapper should behave identically.
    let result2 =
        download_with_checksum("http://[invalid:definitely:not:a:real:url]", &sha, &dest);
    assert!(
        result2.is_ok(),
        "wrapper should succeed: {:?}",
        result2.err()
    );
    assert!(!result2.unwrap(), "wrapper should also be cached");
}

#[test]
fn rejects_non_https_url() {
    // dest does not exist, so the call goes past the cache-hit branch into
    // the scheme check. An http:// URL must be rejected before any network
    // or filesystem side effect.
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("model.bin");
    let err = download_with_checksum_and_signature(
        "http://unreachable.invalid/model.bin",
        &test_bytes_sha256(),
        None,
        &dest,
    )
    .expect_err("non-https URL must be rejected");
    assert!(matches!(err, DownloadError::InsecureScheme { .. }));
    assert!(!dest.exists(), "no file should be created");
    assert!(
        !dir.path().join(".model.bin.partial").exists(),
        "no .partial should be created"
    );
}

#[test]
fn require_https_accepts_https_case_insensitively() {
    require_https("https://example.com/model.onnx").expect("lowercase https");
    require_https("HTTPS://example.com/model.onnx").expect("uppercase https");
    require_https("HtTpS://example.com/model.onnx").expect("mixed case https");
}

#[test]
fn require_https_rejects_other_schemes_and_short_strings() {
    for url in [
        "http://example.com/model.onnx",
        "ftp://example.com/model.onnx",
        "https:/",
        "https",
        "",
    ] {
        let err = require_https(url).expect_err("must reject: {url}");
        assert!(
            matches!(err, DownloadError::InsecureScheme { .. }),
            "expected InsecureScheme for {url}, got {err:?}"
        );
    }
}

#[test]
fn prepare_partial_path_creates_parents_and_names_partial() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("nested").join("deep").join("model.bin");
    let tmp = prepare_partial_path(&dest).expect("must prepare partial path");
    assert!(
        dest.parent().unwrap().is_dir(),
        "parent directories must be created"
    );
    assert_eq!(
        tmp.file_name().unwrap().to_str().unwrap(),
        ".model.bin.partial"
    );
    assert_eq!(tmp.parent(), dest.parent());
}

#[test]
fn prepare_partial_path_falls_back_to_default_name() {
    // A path with no file name uses the "model" fallback for the .partial name.
    let tmp = prepare_partial_path(Path::new("..")).expect("must prepare partial path");
    assert_eq!(tmp.file_name().unwrap().to_str().unwrap(), ".model.partial");
}

#[test]
fn write_capped_streams_all_bytes_under_cap() {
    let dir = TempDir::new().unwrap();
    let tmp = dir.path().join(".data.partial");
    let mut file = fs::File::create(&tmp).unwrap();
    // 200 KiB: spans multiple 64 KiB buffer reads.
    let data: Vec<u8> = (0..200 * 1024u32).map(|i| (i % 251) as u8).collect();
    let mut seen: Vec<u8> = Vec::new();
    write_capped(
        std::io::Cursor::new(data.clone()),
        &mut file,
        &tmp,
        data.len() as u64,
        &mut |chunk| seen.extend_from_slice(chunk),
    )
    .expect("under-cap stream must succeed");
    drop(file);
    assert_eq!(seen, data, "every chunk must be reported in order");
    assert_eq!(fs::read(&tmp).unwrap(), data, "file must hold all bytes");
}

#[test]
fn write_capped_allows_exactly_cap_bytes() {
    let dir = TempDir::new().unwrap();
    let tmp = dir.path().join(".exact.partial");
    let mut file = fs::File::create(&tmp).unwrap();
    let mut noop = |_: &[u8]| {};
    write_capped(
        std::io::Cursor::new(vec![7u8; 100]),
        &mut file,
        &tmp,
        100,
        &mut noop,
    )
    .expect("exactly-at-cap stream must succeed");
    drop(file);
    assert_eq!(fs::read(&tmp).unwrap().len(), 100);
}

#[test]
fn write_capped_read_error_is_io() {
    struct FailingReader;
    impl Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "boom"))
        }
    }
    let dir = TempDir::new().unwrap();
    let tmp = dir.path().join(".readerr.partial");
    let mut file = fs::File::create(&tmp).unwrap();
    let mut noop = |_: &[u8]| {};
    let err = write_capped(FailingReader, &mut file, &tmp, 1024, &mut noop)
        .expect_err("reader failure must surface");
    assert!(matches!(err, DownloadError::Io { .. }));
}

#[test]
fn write_capped_write_error_is_io() {
    let dir = TempDir::new().unwrap();
    let tmp = dir.path().join(".writeerr.partial");
    fs::write(&tmp, b"x").unwrap();
    // Open read-only: write_all must fail.
    let mut file = fs::File::open(&tmp).unwrap();
    let mut noop = |_: &[u8]| {};
    let err = write_capped(
        std::io::Cursor::new(vec![1u8; 10]),
        &mut file,
        &tmp,
        1024,
        &mut noop,
    )
    .expect_err("write failure must surface");
    assert!(matches!(err, DownloadError::Io { .. }));
}

#[test]
fn cache_miss_with_corrupt_existing_file_proceeds_past_cache() {
    // dest exists but its hash does not match: serve_cache_hit must decline,
    // so the call continues to the scheme check and rejects http://.
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("stale.bin");
    fs::write(&dest, b"corrupt contents").unwrap();
    let err = download_with_checksum_and_signature(
        "http://unreachable.invalid/stale.bin",
        &test_bytes_sha256(),
        None,
        &dest,
    )
    .expect_err("corrupt cache must not serve; http must be rejected");
    assert!(matches!(err, DownloadError::InsecureScheme { .. }));
}

#[test]
fn malformed_signature_fails_before_network() {
    // Signature text is parsed before any fetch: garbage must error out as
    // SignatureInvalid even though the URL is valid https.
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("model.bin");
    let err = download_with_checksum_and_signature(
        "https://127.0.0.1:1/model.bin",
        &test_bytes_sha256(),
        Some("this is not a minisign signature"),
        &dest,
    )
    .expect_err("malformed signature must be rejected");
    assert!(matches!(err, DownloadError::SignatureInvalid { .. }));
    assert!(!dest.exists());
}

#[test]
#[cfg_attr(miri, ignore)]
fn https_fetch_to_refused_localhost_is_network_error() {
    // Nothing listens on 127.0.0.1:1, so the fetch fails fast with a
    // connection-refused network error, without leaving the loopback
    // interface. A well-formed signature rides along so the pre-parse and
    // stream-verifier construction paths are exercised too.
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("model.bin");
    let sig = fixture_signature();
    let err = download_with_checksum_and_signature(
        "https://127.0.0.1:1/model.bin",
        &"0".repeat(64),
        Some(&sig),
        &dest,
    )
    .expect_err("refused connection must error");
    assert!(
        matches!(err, DownloadError::Network { .. }),
        "expected Network error, got {err:?}"
    );
    assert!(
        !dir.path().join(".model.bin.partial").exists(),
        "partial must not exist when the request itself fails"
    );
}

#[test]
fn verifier_finish_deletes_tmp_on_checksum_mismatch() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("model.bin");
    let tmp = dir.path().join(".model.bin.partial");
    fs::write(&tmp, TEST_BYTES).unwrap();
    let mut verifier = DownloadVerifier::new(&dest, None).unwrap();
    verifier.update(TEST_BYTES);
    let err = verifier
        .finish(&tmp, &dest, &"f".repeat(64))
        .expect_err("wrong hash must fail");
    assert!(matches!(err, DownloadError::ChecksumMismatch { .. }));
    assert!(!tmp.exists(), "tmp must be deleted on mismatch");
}

#[test]
fn verifier_finish_passes_on_matching_hash() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("model.bin");
    let tmp = dir.path().join(".model.bin.partial");
    fs::write(&tmp, TEST_BYTES).unwrap();
    let mut verifier = DownloadVerifier::new(&dest, None).unwrap();
    verifier.update(TEST_BYTES);
    verifier
        .finish(&tmp, &dest, &test_bytes_sha256())
        .expect("matching hash must pass");
    assert!(tmp.exists(), "tmp is left in place for the rename step");
}

#[test]
#[cfg_attr(miri, ignore)]
fn cache_hit_with_valid_signature_serves_without_network() {
    let Some((dir, dest, sig)) = cached_fixture_model() else {
        return;
    };
    let sha = sha256_of_file(&dest);
    let downloaded = download_with_checksum_and_signature(
        "https://127.0.0.1:1/model.bin",
        &sha,
        Some(&sig),
        &dest,
    )
    .expect("cached + signed model must serve");
    assert!(!downloaded, "cache hit must not download");
    drop(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn cache_hit_with_mismatched_signature_is_rejected() {
    // Cached file hashes correctly, but the signature belongs to a
    // different model: the cache hit must fail signature verification.
    let Some((dir, dest, _sig)) = cached_fixture_model() else {
        return;
    };
    let wrong_sig = fs::read_to_string(fixture_path("powerset_fp32.onnx.minisig"))
        .expect("powerset signature fixture");
    let sha = sha256_of_file(&dest);
    let err = download_with_checksum_and_signature(
        "https://127.0.0.1:1/model.bin",
        &sha,
        Some(&wrong_sig),
        &dest,
    )
    .expect_err("foreign signature must not validate");
    assert!(matches!(err, DownloadError::SignatureInvalid { .. }));
    drop(dir);
}

#[test]
fn download_error_display_mentions_context() {
    let io_err = DownloadError::Io {
        path: PathBuf::from("/tmp/x"),
        source: io::Error::new(io::ErrorKind::NotFound, "gone"),
    };
    assert!(format!("{io_err}").contains("/tmp/x"));

    let mismatch = DownloadError::ChecksumMismatch {
        path: PathBuf::from("m.bin"),
        expected: "ab".repeat(32),
        actual: "cd".repeat(32),
    };
    assert!(format!("{mismatch}").contains("m.bin"));

    let insecure = DownloadError::InsecureScheme {
        url: "http://x".to_owned(),
    };
    assert!(format!("{insecure}").contains("http://x"));

    let too_large = DownloadError::TooLarge {
        path: PathBuf::from("big.bin"),
        max_bytes: 42,
    };
    assert!(format!("{too_large}").contains("42"));

    let sig = DownloadError::SignatureInvalid {
        path: PathBuf::from("s.bin"),
        source: SignatureError::VerificationFailed("nope".to_owned()),
    };
    assert!(format!("{sig}").contains("s.bin"));
}

/// Absolute path to a checked-in model fixture under `models/`.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// Raw `.minisig` text for the small checked-in ecapa model.
fn fixture_signature() -> String {
    fs::read_to_string(fixture_path("ecapa_tdnn_mel.onnx.minisig"))
        .expect("ecapa signature fixture must be checked in")
}

/// Copy the small ecapa model into a fresh temp dir, returning the dir
/// (keep alive), the cached path, and its signature text. `None` (test
/// skips) when the gitignored model blob is not present locally.
fn cached_fixture_model() -> Option<(TempDir, PathBuf, String)> {
    let src = fixture_path("ecapa_tdnn_mel.onnx");
    if !src.exists() {
        eprintln!("skip: models/ecapa_tdnn_mel.onnx missing");
        return None;
    }
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("ecapa_tdnn_mel.onnx");
    fs::copy(src, &dest).expect("copy fixture");
    Some((dir, dest, fixture_signature()))
}

fn sha256_of_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(fs::read(path).unwrap());
    format!("{:x}", h.finalize())
}

#[test]
fn max_download_bytes_uses_declared_size_with_slack() {
    assert_eq!(max_download_bytes(None), DEFAULT_MAX_MODEL_BYTES);
    assert_eq!(max_download_bytes(Some(0)), DEFAULT_MAX_MODEL_BYTES);
    assert_eq!(max_download_bytes(Some(1_000_000)), 2_000_000);
    // 600 MiB declared → 1.2 GiB would exceed the global ceiling → clamp.
    let six_hundred_mib = 600 * 1024 * 1024;
    assert_eq!(
        max_download_bytes(Some(six_hundred_mib)),
        DEFAULT_MAX_MODEL_BYTES
    );
}

#[test]
fn aborts_when_stream_exceeds_cap() {
    // 100 bytes through a 10-byte cap: write_capped must abort with TooLarge
    // and delete the .partial.
    let dir = TempDir::new().unwrap();
    let tmp = dir.path().join(".big.partial");
    let mut file = fs::File::create(&tmp).unwrap();
    let mut noop = |_: &[u8]| {};
    let err = write_capped(
        std::io::Cursor::new(vec![0u8; 100]),
        &mut file,
        &tmp,
        10,
        &mut noop,
    )
    .expect_err("stream over the cap must abort");
    assert!(matches!(err, DownloadError::TooLarge { max_bytes: 10, .. }));
    drop(file);
    assert!(!tmp.exists(), ".partial must be deleted on cap overflow");
}
