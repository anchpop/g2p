//! Shared mandarin labels.

/// One syllable's labels.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Syllable {
    /// The character it came from.
    pub char: char,
    /// Pinyin with tone digit, as g2pM emits it (`u:` → `v`, `r5` → `er5`).
    pub pinyin: String,
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`: the tone number (1–5) on the tone-bearing
    /// phone, `None` elsewhere.
    pub tone: Vec<Option<u8>>,
}
