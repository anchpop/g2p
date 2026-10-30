//! Hindi word/syllable structure.

use crate::Stress;

/// A syllable span within a word's phoneme list (`[start, end)`), with its
/// mora weight and stress from Roy's rules.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Syllable {
    pub start: usize,
    pub end: usize,
    pub nucleus: usize,
    pub moras: u8,
    pub stressed: bool,
}

/// One Devanagari word's labels.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Word {
    /// UTF-8 byte range in the original input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_span: Option<(usize, usize)>,
    pub phonemes: Vec<String>,
    pub stress: Vec<Stress>,
    pub syllables: Vec<Syllable>,
    /// One entry per orthographic schwa: whether the classifier kept it.
    pub schwa_retained: Vec<bool>,
}
