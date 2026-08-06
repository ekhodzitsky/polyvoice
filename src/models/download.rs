//! HTTP download with streamed SHA-256 and optional Minisign verification.

use crate::models::verify::{SignatureError, verify_minisign};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// Errors from `download_with_checksum` and `verify_sha256`.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("network error fetching {url}: {source}")]
    Network {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("checksum mismatch for {path}: expected {expected:.16}…, computed {actual:.16}…")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("signature invalid for {path}: {source}")]
    SignatureInvalid {
        path: PathBuf,
        #[source]
        source: SignatureError,
    },
    #[error("refusing to fetch model over a non-https URL: {url}")]
    InsecureScheme { url: String },
    #[error("download for {path} exceeded the {max_bytes}-byte cap")]
    TooLarge { path: PathBuf, max_bytes: u64 },
}

/// { !url.is_empty() && expected_sha256.len() == 64 }
/// `pub fn download_with_checksum( url: &str, expected_sha256: &str, dest: &Path, ) -> Result<bool, DownloadError>`
/// { ret.as_ref().map_or(true, |&downloaded| if downloaded { dest.exists() } else { true }) }
/// Stream `url` to `dest` and verify the SHA-256 matches `expected_sha256`.
///
/// Idempotent: if `dest` already exists with the correct hash, returns Ok(false)
/// immediately. Otherwise downloads, hashes while streaming (so 200+ MB files
/// don't blow up RAM), and on hash mismatch deletes the partial file and returns
/// an error. Returns `Ok(true)` if a download happened, `Ok(false)` if cached.
///
/// Backwards-compatibility wrapper: delegates to [`download_with_checksum_and_signature`]
/// with `signature: None`.
pub fn download_with_checksum(
    url: &str,
    expected_sha256: &str,
    dest: &Path,
) -> Result<bool, DownloadError> {
    download_with_checksum_and_signature(url, expected_sha256, None, dest)
}

/// { !url.is_empty() && expected_sha256.len() == 64 }
/// `pub fn download_with_checksum_and_signature( url: &str, expected_sha256: &str, signature: Option<&str>, dest: &Path, ) -> Result<bool, DownloadError>`
/// { ret.as_ref().map_or(true, |&downloaded| if downloaded { dest.exists() } else { true }) }
/// Stream `url` to `dest`, verify SHA-256, and optionally verify a Minisign signature.
///
/// When `signature` is `Some(sig_text)`, the signature is verified both on cache
/// hits and after fresh downloads. If verification fails, the temp file is deleted
/// and `DownloadError::SignatureInvalid` is returned.
///
/// Streams everything in 64 KiB chunks; does not load the whole model into memory.
pub fn download_with_checksum_and_signature(
    url: &str,
    expected_sha256: &str,
    signature: Option<&str>,
    dest: &Path,
) -> Result<bool, DownloadError> {
    download_with_checksum_signature_and_cap(
        url,
        expected_sha256,
        signature,
        dest,
        DEFAULT_MAX_MODEL_BYTES,
    )
}

/// Default absolute ceiling for a single streamed model download (1 GiB).
///
/// Bounds a disk-exhaustion DoS for manifest entries that do not declare a
/// `size`. It sits well above any real polyvoice model (the largest shipped
/// weights are ~250 MiB), so legitimate downloads are unaffected.
pub(crate) const DEFAULT_MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;

/// Streaming size cap for one model given an optional manifest-declared size.
///
/// When `declared_size` is set and positive, the cap is `2 × size` (slack for
/// upstream drift) still clamped to [`DEFAULT_MAX_MODEL_BYTES`]. Missing or
/// zero size falls back to the global 1 GiB ceiling.
pub(crate) fn max_download_bytes(declared_size: Option<u64>) -> u64 {
    match declared_size {
        Some(n) if n > 0 => n.saturating_mul(2).min(DEFAULT_MAX_MODEL_BYTES),
        _ => DEFAULT_MAX_MODEL_BYTES,
    }
}

/// Like [`download_with_checksum_and_signature`] but with an explicit streaming
/// size cap and an enforced `https://` scheme.
///
/// * Rejects any non-`https://` URL with [`DownloadError::InsecureScheme`]
///   before opening the network. Cache hits transmit nothing and are still
///   served — the scheme is only required when bytes are actually fetched.
/// * Aborts and deletes the `.partial` file if the stream exceeds `max_bytes`,
///   returning [`DownloadError::TooLarge`], so a hostile or buggy endpoint
///   cannot fill the disk before the SHA-256 check runs.
pub(crate) fn download_with_checksum_signature_and_cap(
    url: &str,
    expected_sha256: &str,
    signature: Option<&str>,
    dest: &Path,
    max_bytes: u64,
) -> Result<bool, DownloadError> {
    // Stage 1 — cache hit: verify SHA-256, then the signature if present. No
    // network here, so the URL scheme is irrelevant.
    if serve_cache_hit(dest, expected_sha256, signature)? {
        return Ok(false);
    }

    // Stage 2 — a real fetch will happen: require https:// so weights are
    // never pulled in cleartext (integrity must not rest on the same-manifest
    // hash alone).
    require_https(url)?;

    // Stage 3 — prepare the destination directory and the sibling `.partial`
    // path the download streams into.
    let tmp = prepare_partial_path(dest)?;

    // Stage 4 — pre-parse the Minisign public key and signature so malformed
    // input fails fast, before any network access.
    let prepared = signature
        .map(|sig_text| PreparedSignature::new(dest, sig_text))
        .transpose()?;
    let mut verifier = DownloadVerifier::new(dest, prepared.as_ref())?;

    // Stage 5 — stream the body into the `.partial` file, hashing and
    // signature-verifying each chunk; aborts past `max_bytes`.
    fetch_into_partial(url, &tmp, max_bytes, &mut verifier)?;

    // Stage 6 — verify the checksum, then the signature (both delete the
    // `.partial` on failure).
    verifier.finish(&tmp, dest, expected_sha256)?;

    // Stage 7 — atomic rename, so a partial file is never seen as cached.
    fs::rename(&tmp, dest).map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    Ok(true)
}

/// Cache-hit short-circuit: `dest` already exists with the expected SHA-256.
/// When a signature is present it is verified on the hit as well. Returns
/// `Ok(true)` if the cached file serves the request (no network needed).
fn serve_cache_hit(
    dest: &Path,
    expected_sha256: &str,
    signature: Option<&str>,
) -> Result<bool, DownloadError> {
    if !(dest.exists() && verify_sha256(dest, expected_sha256).is_ok()) {
        return Ok(false);
    }
    if let Some(sig) = signature {
        verify_minisign(dest, sig).map_err(|e| DownloadError::SignatureInvalid {
            path: dest.to_path_buf(),
            source: e,
        })?;
    }
    Ok(true)
}

/// Reject any non-`https://` URL before opening the network. Cache hits
/// transmit nothing and never reach this check.
fn require_https(url: &str) -> Result<(), DownloadError> {
    if !url
        .get(..8)
        .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
    {
        return Err(DownloadError::InsecureScheme {
            url: url.to_owned(),
        });
    }
    Ok(())
}

/// Create `dest`'s parent directory if needed and return the sibling
/// `.partial` path the download is staged through.
fn prepare_partial_path(dest: &Path) -> Result<PathBuf, DownloadError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| DownloadError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let mut tmp = dest.to_path_buf();
    let original_name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    tmp.set_file_name(format!(".{original_name}.partial"));
    Ok(tmp)
}

/// Minisign public key + signature parsed up front; owns the values the
/// streaming verifier borrows.
struct PreparedSignature {
    public_key: minisign_verify::PublicKey,
    signature: minisign_verify::Signature,
}

impl PreparedSignature {
    fn new(dest: &Path, sig_text: &str) -> Result<Self, DownloadError> {
        let public_key =
            minisign_verify::PublicKey::from_base64(crate::models::verify::SIGNING_PUBKEY_BASE64)
                .map_err(|e| DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: SignatureError::BadPublicKey(format!("{e:?}")),
            })?;
        let signature = minisign_verify::Signature::decode(sig_text).map_err(|e| {
            DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: SignatureError::BadSignature(format!("{e:?}")),
            }
        })?;
        Ok(Self {
            public_key,
            signature,
        })
    }

    fn stream_verifier(
        &self,
        dest: &Path,
    ) -> Result<minisign_verify::StreamVerifier<'_>, DownloadError> {
        self.public_key.verify_stream(&self.signature).map_err(|e| {
            DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: SignatureError::VerificationFailed(format!("{e:?}")),
            }
        })
    }
}

/// Streaming integrity state fed chunk-by-chunk while the download is
/// written: a SHA-256 hasher plus, when the manifest carries a signature, a
/// Minisign stream verifier.
struct DownloadVerifier<'a> {
    hasher: Sha256,
    minisign: Option<minisign_verify::StreamVerifier<'a>>,
}

impl<'a> DownloadVerifier<'a> {
    fn new(dest: &Path, prepared: Option<&'a PreparedSignature>) -> Result<Self, DownloadError> {
        let minisign = match prepared {
            Some(p) => Some(p.stream_verifier(dest)?),
            None => None,
        };
        Ok(Self {
            hasher: Sha256::new(),
            minisign,
        })
    }

    fn update(&mut self, chunk: &[u8]) {
        self.hasher.update(chunk);
        if let Some(v) = self.minisign.as_mut() {
            v.update(chunk);
        }
    }

    /// Compare the streamed SHA-256 against `expected_sha256`, then finalize
    /// the Minisign verifier — in that order. Deletes `tmp` on failure.
    fn finish(self, tmp: &Path, dest: &Path, expected_sha256: &str) -> Result<(), DownloadError> {
        let actual = format!("{:x}", self.hasher.finalize());
        if actual != expected_sha256 {
            let _ = fs::remove_file(tmp);
            return Err(DownloadError::ChecksumMismatch {
                path: dest.to_path_buf(),
                expected: expected_sha256.to_owned(),
                actual,
            });
        }
        if let Some(mut v) = self.minisign {
            v.finalize().map_err(|e| {
                let _ = fs::remove_file(tmp);
                DownloadError::SignatureInvalid {
                    path: dest.to_path_buf(),
                    source: SignatureError::VerificationFailed(format!("{e:?}")),
                }
            })?;
        }
        Ok(())
    }
}

/// Stream `url` into `tmp`, feeding each chunk to `verifier` and enforcing
/// the `max_bytes` cap (see [`write_capped`]).
fn fetch_into_partial(
    url: &str,
    tmp: &Path,
    max_bytes: u64,
    verifier: &mut DownloadVerifier<'_>,
) -> Result<(), DownloadError> {
    let resp = ureq::get(url).call().map_err(|e| DownloadError::Network {
        url: url.to_owned(),
        source: Box::new(e),
    })?;
    let reader = BufReader::new(resp.into_body().into_reader());
    let mut file = fs::File::create(tmp).map_err(|e| DownloadError::Io {
        path: tmp.to_path_buf(),
        source: e,
    })?;
    write_capped(reader, &mut file, tmp, max_bytes, &mut |chunk| {
        verifier.update(chunk);
    })?;
    file.flush().map_err(|e| DownloadError::Io {
        path: tmp.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

/// Stream `reader` into `file` in 64 KiB chunks, calling `on_chunk` for each
/// chunk (used for SHA-256 and signature updates), and aborting with
/// [`DownloadError::TooLarge`] — after deleting `tmp` — if more than `max_bytes`
/// are read. The cap is checked before each write, so the on-disk `.partial`
/// never exceeds the limit.
fn write_capped<R: Read>(
    mut reader: R,
    file: &mut fs::File,
    tmp: &Path,
    max_bytes: u64,
    on_chunk: &mut dyn FnMut(&[u8]),
) -> Result<(), DownloadError> {
    let mut buf = [0u8; 64 * 1024];
    let mut written: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadError::Io {
            path: tmp.to_path_buf(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        written += n as u64;
        if written > max_bytes {
            let _ = fs::remove_file(tmp);
            return Err(DownloadError::TooLarge {
                path: tmp.to_path_buf(),
                max_bytes,
            });
        }
        file.write_all(&buf[..n]).map_err(|e| DownloadError::Io {
            path: tmp.to_path_buf(),
            source: e,
        })?;
        on_chunk(&buf[..n]);
    }
    Ok(())
}

/// { expected.len() == 64 }
/// `pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), DownloadError>`
/// { true }
/// Compute the SHA-256 of `path` and compare against `expected`. Streams the file
/// (does not load it into RAM).
pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), DownloadError> {
    let f = fs::File::open(path).map_err(|e| DownloadError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual == expected {
        Ok(())
    } else {
        Err(DownloadError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        })
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
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
}
