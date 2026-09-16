//! Shared Korean labels.

/// One utterance's labels.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Labels {
    /// The pronunciation as post-sandhi Hangul, one token per input word,
    /// clauses separated by ` | ` (값이 안 좋아, 라디오 → "갑씨 안 조아 |
    /// 라디오"). Readable; not for scoring.
    pub raw: String,
    pub phonemes: Vec<String>,
    /// All `None`: Korean has no lexical stress.
    pub stress: Vec<crate::Stress>,
    /// `[start, end)` per input word (어절).
    pub word_spans: Vec<(usize, usize)>,
}
