//! Shared japanese labels.

use crate::Pitch;

/// Labels for one utterance.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Labels {
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`; `None` on phones that bear no mora.
    pub pitch: Vec<Option<Pitch>>,
    /// If set, `pitch` is empty and this says why the accent factor is not
    /// trustworthy for this utterance (the phones still are).
    pub accent_withheld: Option<String>,
    /// OpenJTalk's own phone strings, for diagnostics and parity checks.
    pub native_phones: Vec<String>,
}
