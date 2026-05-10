//! Minisign signature verification for model artifacts.
//!
//! Uses `minisign-verify` (zero dependencies, pure Rust) to stream-verify
//! downloaded ONNX files against an embedded project public key.

use minisign_verify::{PublicKey, Signature};
use std::io::{BufReader, Read};
use std::path::Path;

/// The base64 public key baked into the binary at compile time.
pub(crate) const SIGNING_PUBKEY_BASE64: &str = include_str!("signing.pub.base64");

/// Errors from the signature verification step.
#[derive(Debug, thiserror::Error)]
pub enum SignatureError {
    #[error("invalid public key: {0}")]
    BadPublicKey(String),
    #[error("invalid signature text: {0}")]
    BadSignature(String),
    #[error("signature verification failed: {0}")]
    VerificationFailed(String),
}

/// Verify `path` against a Minisign signature string (the raw `.minisig` text).
///
/// Streams the file in 64 KiB chunks; does not load the whole model into memory.
pub fn verify_minisign(path: &Path, sig_text: &str) -> Result<(), SignatureError> {
    let public_key = PublicKey::from_base64(SIGNING_PUBKEY_BASE64)
        .map_err(|e| SignatureError::BadPublicKey(format!("{e:?}")))?;

    let signature =
        Signature::decode(sig_text).map_err(|e| SignatureError::BadSignature(format!("{e:?}")))?;

    let mut verifier = public_key
        .verify_stream(&signature)
        .map_err(|e| SignatureError::VerificationFailed(format!("{e:?}")))?;

    let file = std::fs::File::open(path)
        .map_err(|e| SignatureError::VerificationFailed(format!("io open: {e}")))?;
    let mut reader = BufReader::new(file);
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| SignatureError::VerificationFailed(format!("io read: {e}")))?;
        if n == 0 {
            break;
        }
        verifier.update(&buf[..n]);
    }

    verifier
        .finalize()
        .map_err(|e| SignatureError::VerificationFailed(format!("{e:?}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: generate a temporary file with known content and sign it with the
    /// project's signing key, returning (temp_dir, path, signature_text).
    /// The TempDir must be kept alive so the file isn't deleted.
    /// Returns `None` when `minisign` CLI or `models/signing.key` are missing
    /// so tests gracefully skip on CI runners that lack them.
    fn sign_temp_file(content: &[u8]) -> Option<(TempDir, std::path::PathBuf, String)> {
        if std::process::Command::new("minisign")
            .arg("--version")
            .output()
            .is_err()
        {
            return None;
        }
        if !std::path::Path::new("models/signing.key").exists() {
            return None;
        }

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.bin");
        fs::write(&path, content).unwrap();

        let sig_path = dir.path().join("test.bin.minisig");
        let status = std::process::Command::new("minisign")
            .args([
                "-S",
                "-s",
                "models/signing.key",
                "-m",
                path.to_str().unwrap(),
                "-x",
                sig_path.to_str().unwrap(),
                "-t",
                "test signature",
            ])
            .status()
            .expect("minisign spawn failed");
        assert!(status.success(), "minisign signing failed");

        let sig_text = fs::read_to_string(&sig_path).expect("signature file must exist");
        Some((dir, path, sig_text))
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn valid_signature_passes() {
        let Some((_dir, path, sig_text)) = sign_temp_file(b"polyvoice test content") else {
            return;
        };
        verify_minisign(&path, &sig_text).expect("valid signature must verify");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn tampered_signature_fails() {
        let Some((_dir, path, mut sig_text)) = sign_temp_file(b"polyvoice test content") else {
            return;
        };
        // Corrupt one character in the base64 signature line (second line).
        let lines: Vec<&str> = sig_text.lines().collect();
        assert!(lines.len() >= 2);
        let mut chars: Vec<char> = lines[1].chars().collect();
        if chars[0] == 'A' {
            chars[0] = 'B';
        } else {
            chars[0] = 'A';
        }
        let corrupted: String = chars.into_iter().collect();
        sig_text = sig_text
            .lines()
            .enumerate()
            .map(|(i, l)| if i == 1 { &corrupted[..] } else { l })
            .collect::<Vec<_>>()
            .join("\n");
        let err = verify_minisign(&path, &sig_text).expect_err("tampered signature must fail");
        let msg = format!("{err}").to_lowercase();
        assert!(
            msg.contains("verification failed")
                || msg.contains("bad signature")
                || msg.contains("invalid signature text"),
            "error should indicate signature problem, got: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn tampered_file_fails() {
        let Some((_dir, path, sig_text)) = sign_temp_file(b"polyvoice test content") else {
            return;
        };
        // Mutate one byte in the file.
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = bytes[0].wrapping_add(1);
        fs::write(&path, &bytes).unwrap();
        let err = verify_minisign(&path, &sig_text).expect_err("tampered file must fail");
        assert!(
            format!("{err}")
                .to_lowercase()
                .contains("verification failed"),
            "error should indicate verification failure, got: {err}"
        );
    }

    #[test]
    fn malformed_signature_text_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.bin");
        fs::write(&path, b"irrelevant").unwrap();
        let err = verify_minisign(&path, "totally not a signature").expect_err("must fail");
        let msg = format!("{err}").to_lowercase();
        assert!(
            msg.contains("bad signature") || msg.contains("invalid signature text"),
            "error should indicate bad signature, got: {err}"
        );
    }

    #[test]
    fn empty_signature_text_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.bin");
        fs::write(&path, b"irrelevant").unwrap();
        let err = verify_minisign(&path, "").expect_err("must fail");
        let msg = format!("{err}").to_lowercase();
        assert!(
            msg.contains("bad signature") || msg.contains("invalid signature text"),
            "error should indicate bad signature, got: {err}"
        );
    }
}
