//! CI guard: clustering hyperparameters must not branch on dataset name.
//!
//! The cVBx robustness claim (and polyvoice's anti-overfit policy) is a single
//! global hyperparameter set across corpora. This test scans the clustering /
//! pipeline config surfaces for forbidden per-dataset conditionals.

use std::fs;
use std::path::PathBuf;

/// Source roots that own clustering configuration. Expanding this list is fine;
/// adding a dataset-name branch inside any of them is not.
const SCAN_ROOTS: &[&str] = &[
    "src/clusterer",
    "src/ahc",
    "src/cluster",
    "src/pipeline_v2/config.rs",
    "src/pipeline_v2/builder.rs",
];

/// Patterns that indicate per-dataset hyperparameter branching.
const FORBIDDEN: &[&str] = &[
    "dataset == \"ami\"",
    "dataset == \"voxconverse\"",
    "dataset == \"AMI\"",
    "dataset == \"VoxConverse\"",
    "if dataset",
    "match dataset",
    "per_dataset",
    "per-dataset",
    "ami_threshold",
    "vox_threshold",
    "ami_fa",
    "vox_fa",
];

fn collect_rs_files(path: &std::path::Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path.to_path_buf());
        }
        return;
    }
    if !path.is_dir() {
        return;
    }
    let entries = fs::read_dir(path).unwrap_or_else(|e| panic!("read_dir {}: {e}", path.display()));
    for entry in entries.flatten() {
        collect_rs_files(&entry.path(), out);
    }
}

#[test]
fn no_per_dataset_clustering_hyperparameters() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for root in SCAN_ROOTS {
        collect_rs_files(&manifest.join(root), &mut files);
    }
    assert!(
        !files.is_empty(),
        "scan found no .rs files under {SCAN_ROOTS:?}"
    );

    let mut hits: Vec<String> = Vec::new();
    for path in &files {
        let text = fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("read {}: {e}", path.display());
        });
        // Skip this guard file itself if it ever lands under a scan root.
        if path.ends_with("global_hyperparam_guard.rs") {
            continue;
        }
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            // Allow comments that *mention* the policy (forbid implementing it).
            if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("//!")
            {
                continue;
            }
            for pat in FORBIDDEN {
                if line.contains(pat) {
                    hits.push(format!("{}:{}: {line}", path.display(), lineno + 1));
                }
            }
        }
    }

    assert!(
        hits.is_empty(),
        "per-dataset clustering hyperparameter branches are forbidden \
         (one global set across corpora). Offending lines:\n{}",
        hits.join("\n")
    );
}
