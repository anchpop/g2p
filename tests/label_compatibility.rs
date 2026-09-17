//! A review tripwire, not part of the runtime identity. Label-preserving
//! refactors refresh this baseline; label changes also require a version bump.
use std::path::Path;
use xxhash_rust::xxh3::Xxh3;

const REVIEWED_VERSION: &str = "0.5.0";
const REVIEWED_SOURCE_DIGEST: &str = "a965d070b519b870";

fn label_source_digest(root: &Path) -> String {
    // Include all files, not just Rust: Mandarin weights/vocabulary are label
    // inputs too. Whole-file bytes intentionally make comments trigger review.
    // The test/baseline itself stays outside these trees (no self-reference).
    let mut files = Vec::new();
    for input in [
        "src",
        "g2p-types/src",
        "Cargo.toml",
        "g2p-types/Cargo.toml",
        "Cargo.lock",
        "build.rs",
    ] {
        for entry in walkdir::WalkDir::new(root.join(input)) {
            let entry = entry.expect("read label source tree");
            assert!(
                !entry.file_type().is_symlink(),
                "symlinked label sources are not supported by the review guard: {}",
                entry.path().display()
            );
            if entry.file_type().is_file() {
                let relative = entry
                    .path()
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace('\\', "/");
                files.push((relative, entry.into_path()));
            }
        }
    }
    files.sort();
    let mut digest = Xxh3::new();
    for (relative, path) in files {
        let bytes = std::fs::read(path).expect("read label source file");
        digest.update(relative.as_bytes());
        digest.update(b"\0");
        digest.update(&(bytes.len() as u64).to_le_bytes());
        digest.update(&bytes);
    }
    format!("{:016x}", digest.digest())
}

#[test]
fn label_producing_sources_have_been_reviewed() {
    let actual_digest = label_source_digest(Path::new(env!("CARGO_MANIFEST_DIR")));
    assert_eq!(
        (env!("CARGO_PKG_VERSION"), actual_digest.as_str()),
        (REVIEWED_VERSION, REVIEWED_SOURCE_DIGEST),
        "Label-producing code changed. Confirm labels are unchanged and refresh the \
         reviewed digest, or bump the package version and refresh both baselines \
         if labels changed: identity must change when labels do. A version bump \
         never disables this guard."
    );
}
