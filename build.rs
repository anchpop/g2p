//! Build the espeak-ng fork (git submodule at `espeak-ng/`) as a static
//! library, compile its phoneme/dictionary data, and pack that data into a
//! single zstd blob the library embeds with `include_bytes!`. The result is a
//! self-contained binary: nothing to install, no data path to configure, and
//! no way for a consumer to accidentally run against mainline espeak.
//!
//! Requires `cmake` and a C compiler on the build machine.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest_dir.join("espeak-ng");
    assert!(
        src.join("CMakeLists.txt").exists(),
        "espeak-ng sources missing at {} — run `git submodule update --init`",
        src.display()
    );
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Everything that can change phoneme output: the translation engine, the
    // pronunciation rules/dictionaries, the phoneme tables, and the voice
    // definitions. Also what triggers a rebuild.
    let output_relevant = [
        "src/libespeak-ng",
        "src/ucd-tools",
        "src/include",
        "dictsource",
        "phsource",
        "espeak-ng-data",
        "cmake",
        "CMakeLists.txt",
    ];
    for sub in output_relevant {
        println!("cargo:rerun-if-changed={}", src.join(sub).display());
    }

    let dst = cmake::Config::new(&src)
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        // Audio backends are irrelevant to phoneme output; leaving them out
        // avoids optional link dependencies (and libstdc++ for speechPlayer).
        .define("USE_MBROLA", "OFF")
        .define("USE_LIBSONIC", "OFF")
        .define("USE_LIBPCAUDIO", "OFF")
        .define("USE_KLATT", "OFF")
        .define("USE_SPEECHPLAYER", "OFF")
        .define("USE_ASYNC", "OFF")
        .define("ENABLE_TESTS", "OFF")
        .define("ESPEAK_COMPAT", "OFF")
        // `data` compiles phondata/intonations/dictionaries with the freshly
        // built espeak-ng binary; it depends on the library, so this builds
        // both.
        .build_target("data")
        .build();
    let build = dst.join("build");

    println!(
        "cargo:rustc-link-search=native={}",
        build.join("src/libespeak-ng").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("src/ucd-tools").display()
    );
    println!("cargo:rustc-link-lib=static=espeak-ng");
    println!("cargo:rustc-link-lib=static=ucd");
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() != "windows" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    let data_dir = build.join("espeak-ng-data");
    assert!(
        data_dir.join("phontab").exists(),
        "espeak-ng data build produced no phontab in {}",
        data_dir.display()
    );
    let blob = pack_dir(&data_dir);
    let compressed = zstd::encode_all(blob.as_slice(), 15).expect("zstd compression failed");
    std::fs::write(out_dir.join("espeak-ng-data.zst"), compressed).unwrap();

    // Identity of this build's phoneme output: a digest of every source file
    // that can affect it. Deterministic across machines and compilers (unlike
    // hashing the built artifacts), so cached artifacts stamped with it can be
    // shared between hosts, and any fork change invalidates them.
    let digest = digest_sources(&src, &output_relevant);
    println!("cargo:rustc-env=G2P_ESPEAK_DIGEST={digest:016x}");
    println!(
        "cargo:rustc-env=G2P_ESPEAK_COMMIT={}",
        git_head(&src).unwrap_or_else(|| "unknown".to_string())
    );
}

/// Sorted list of every regular file under `root`, as (relative path, absolute path).
fn files_under(root: &Path) -> Vec<(String, PathBuf)> {
    let mut files: Vec<(String, PathBuf)> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            let rel = e
                .path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            (rel, e.path().to_path_buf())
        })
        .collect();
    files.sort();
    files
}

/// Trivial archive: for each file, `u32 path length, path bytes, u64 size,
/// contents`. Unpacked by `src/data.rs`.
fn pack_dir(dir: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    for (rel, path) in files_under(dir) {
        let bytes = std::fs::read(&path).unwrap();
        out.extend_from_slice(&(rel.len() as u32).to_le_bytes());
        out.extend_from_slice(rel.as_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    out
}

fn digest_sources(src: &Path, subs: &[&str]) -> u64 {
    use xxhash_rust::xxh3::Xxh3;
    let mut h = Xxh3::new();
    for sub in subs {
        let root = src.join(sub);
        if root.is_file() {
            h.update(sub.as_bytes());
            h.update(&std::fs::read(&root).unwrap());
            continue;
        }
        for (rel, path) in files_under(&root) {
            h.update(sub.as_bytes());
            h.update(b"/");
            h.update(rel.as_bytes());
            h.update(b"\0");
            h.update(&std::fs::read(&path).unwrap());
        }
    }
    h.digest()
}

fn git_head(src: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(src)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}
