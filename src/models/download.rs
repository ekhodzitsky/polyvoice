//! HTTP download with streamed SHA-256 verification.

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
}

/// Stream `url` to `dest` and verify the SHA-256 matches `expected_sha256`.
///
/// Idempotent: if `dest` already exists with the correct hash, returns Ok(false)
/// immediately. Otherwise downloads, hashes while streaming (so 200+ MB files
/// don't blow up RAM), and on hash mismatch deletes the partial file and returns
/// an error. Returns `Ok(true)` if a download happened, `Ok(false)` if cached.
pub fn download_with_checksum(
    url: &str,
    expected_sha256: &str,
    dest: &Path,
) -> Result<bool, DownloadError> {
    if dest.exists() && verify_sha256(dest, expected_sha256).is_ok() {
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
    fs::rename(&tmp, dest).map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    Ok(true)
}

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
}
