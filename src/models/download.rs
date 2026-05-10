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
}

impl From<SignatureError> for DownloadError {
    fn from(source: SignatureError) -> Self {
        DownloadError::SignatureInvalid {
            path: PathBuf::from("(unknown)"),
            source,
        }
    }
}

/// { TODO: precondition }
/// pub fn download_with_checksum( url: &str, expected_sha256: &str, dest: &Path, ) -> Result<bool, DownloadError>
/// { TODO: postcondition }
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

/// { TODO: precondition }
/// pub fn download_with_checksum_and_signature( url: &str, expected_sha256: &str, signature: Option<&str>, dest: &Path, ) -> Result<bool, DownloadError>
/// { TODO: postcondition }
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
    // Cache hit: verify SHA-256, then signature if present.
    if dest.exists() && verify_sha256(dest, expected_sha256).is_ok() {
        if let Some(sig) = signature {
            verify_minisign(dest, sig).map_err(|e| DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: e,
            })?;
        }
        return Ok(false);
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| DownloadError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    // Download to a sibling .partial file, then rename — gives atomic on-success
    // semantics so a partial file is never seen as cached.
    let mut tmp = dest.to_path_buf();
    let original_name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    tmp.set_file_name(format!(".{original_name}.partial"));

    // Pre-parse Minisign public key and signature so we fail fast before the network.
    let public_key = if signature.is_some() {
        Some(
            minisign_verify::PublicKey::from_base64(crate::models::verify::SIGNING_PUBKEY_BASE64)
                .map_err(|e| DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: SignatureError::BadPublicKey(format!("{e:?}")),
            })?,
        )
    } else {
        None
    };
    let sig = if let Some(sig_text) = signature {
        Some(minisign_verify::Signature::decode(sig_text).map_err(|e| {
            DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: SignatureError::BadSignature(format!("{e:?}")),
            }
        })?)
    } else {
        None
    };
    let mut verifier = if let (Some(pk), Some(s)) = (&public_key, &sig) {
        Some(
            pk.verify_stream(s)
                .map_err(|e| DownloadError::SignatureInvalid {
                    path: dest.to_path_buf(),
                    source: SignatureError::VerificationFailed(format!("{e:?}")),
                })?,
        )
    } else {
        None
    };

    let resp = ureq::get(url).call().map_err(|e| DownloadError::Network {
        url: url.to_owned(),
        source: Box::new(e),
    })?;
    let reader = resp.into_body().into_reader();
    let mut reader = BufReader::new(reader);
    let mut file = fs::File::create(&tmp).map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).map_err(|e| DownloadError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        if let Some(ref mut v) = verifier {
            v.update(&buf[..n]);
        }
    }
    file.flush().map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    drop(file);

    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_sha256 {
        let _ = fs::remove_file(&tmp);
        return Err(DownloadError::ChecksumMismatch {
            path: dest.to_path_buf(),
            expected: expected_sha256.to_owned(),
            actual,
        });
    }

    if let Some(mut v) = verifier {
        v.finalize().map_err(|e| {
            let _ = fs::remove_file(&tmp);
            DownloadError::SignatureInvalid {
                path: dest.to_path_buf(),
                source: SignatureError::VerificationFailed(format!("{e:?}")),
            }
        })?;
    }

    fs::rename(&tmp, dest).map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    Ok(true)
}

/// { TODO: precondition }
/// pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), DownloadError>
/// { TODO: postcondition }
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
}
