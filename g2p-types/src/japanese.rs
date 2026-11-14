//! Shared Japanese labels.

use crate::Pitch;

/// Labels for one utterance.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Labels {
    /// Phone ranges for NJD-node words, when recoverable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub word_spans: Vec<(usize, usize)>,
    /// Original-input byte ranges, parallel to words, or empty when uncertain.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub word_text_spans: Vec<(usize, usize)>,
    /// Original-input byte range of each phone's NJD node (both nodes for a
    /// long consonant across two), parallel to `phonemes`, or empty when
    /// uncertain.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phoneme_text_spans: Vec<(usize, usize)>,
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`; `None` on phones that bear no mora.
    pub pitch: Vec<Option<Pitch>>,
    /// If set, `pitch` is empty and this says why the accent factor is not
    /// trustworthy for this utterance (the phones still are).
    pub accent_withheld: Option<String>,
    /// OpenJTalk's own phone strings, for diagnostics and parity checks.
    pub native_phones: Vec<String>,
}
