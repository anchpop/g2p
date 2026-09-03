//! The compiled espeak-ng data (phoneme tables, dictionaries, voices) is
//! embedded in the binary at build time. espeak reads it from disk, so on
//! first use it is unpacked into a per-build cache directory, keyed by the
//! source digest, and reused thereafter.

use std::io;
use std::path::{Path, PathBuf};

static ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/espeak-ng-data.zst"));

/// The per-build cache directory (`<cache>/g2p/<espeak digest>`), created if
/// needed: where the espeak data and the Thai Python project are unpacked.
pub(crate) fn cache_root() -> io::Result<PathBuf> {
    let mut last_err = None;
    for base in candidate_roots() {
        let dir = base.join("g2p").join(crate::ESPEAK_DIGEST);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("no writable cache directory")))
}

/// Directory to hand to `espeak_ng_InitializePath` (espeak appends
/// `espeak-ng-data` itself). Unpacks the embedded data if needed.
pub fn ensure_unpacked() -> io::Result<PathBuf> {
    let mut last_err = None;
    for base in candidate_roots() {
        match unpack_into(&base.join("g2p").join(crate::ESPEAK_DIGEST)) {
            Ok(dir) => return Ok(dir),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("no writable cache directory")))
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(cache) = dirs::cache_dir() {
        roots.push(cache);
    }
    roots.push(std::env::temp_dir());
    roots
}

/// Unpack into `dir` atomically: extract to a sibling temp dir, then rename.
/// A concurrent unpack of the same build races harmlessly — whoever renames
/// first wins and the other discards its copy.
fn unpack_into(dir: &Path) -> io::Result<PathBuf> {
    let marker = dir.join(".unpacked");
    if marker.exists() {
        return Ok(dir.to_path_buf());
    }
    let parent = dir
        .parent()
        .ok_or_else(|| io::Error::other("cache dir has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        dir.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;

    let blob = zstd::decode_all(ARCHIVE)?;
    let data_root = tmp.join("espeak-ng-data");
    for (rel, bytes) in entries(&blob) {
        let path = data_root.join(rel);
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(path, bytes)?;
    }
    std::fs::write(tmp.join(".unpacked"), b"")?;

    match std::fs::rename(&tmp, dir) {
        Ok(()) => {}
        Err(_) if marker.exists() => {
            let _ = std::fs::remove_dir_all(&tmp);
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }
    }
    Ok(dir.to_path_buf())
}

/// Iterate the archive written by `build.rs::pack_dir`.
fn entries(blob: &[u8]) -> impl Iterator<Item = (&str, &[u8])> {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        if pos >= blob.len() {
            return None;
        }
        let path_len = u32::from_le_bytes(blob[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let rel = std::str::from_utf8(&blob[pos..pos + path_len]).expect("archive path is utf-8");
        pos += path_len;
        let size = u64::from_le_bytes(blob[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let bytes = &blob[pos..pos + size];
        pos += size;
        Some((rel, bytes))
    })
}
