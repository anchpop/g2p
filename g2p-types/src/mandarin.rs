//! Shared Mandarin labels.

/// One syllable's labels.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Syllable {
    /// The character it came from.
    pub char: char,
    /// Pinyin with tone digit, as g2pM emits it (`u:` → `v`, `r5` → `er5`).
    pub pinyin: String,
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`: the surface tone (1–5), after sandhi, on the tone-bearing
    /// phone, `None` elsewhere.
    pub tone: Vec<Option<u8>>,
}
