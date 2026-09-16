//! Shared thai labels.

/// One utterance's labels.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Labels {
    /// vachana's IPA string as returned.
    pub raw: String,
    pub phonemes: Vec<String>,
    pub stress: Vec<crate::Stress>,
    /// Tone class per phoneme: 1 mid, 2 low, 3 falling, 4 high, 5 rising on
    /// vowels (the tone-bearing phone of each syllable), `None` on consonants.
    pub tone: Vec<Option<u8>>,
    /// `[start, end)` per word (vachana's space-separated tokens).
    pub word_spans: Vec<(usize, usize)>,
}
