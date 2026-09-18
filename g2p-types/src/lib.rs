//! Shared pronunciation labels, independent of native engines and backend features.

pub mod hindi;
pub mod japanese;
pub mod korean;
mod language;
pub mod mandarin;
pub mod parse;
pub mod thai;

pub use hindi::Syllable;
pub use language::Language;
pub use parse::{Parsed, Stress};

/// Tokyo pitch-accent factor for one mora-bearing phone (Japanese).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pitch {
    /// 1-based accent phrase index within the utterance.
    pub phrase: u8,
    /// 1-based mora index within the accent phrase.
    pub mora: u8,
    pub phrase_moras: u8,
    /// Accent nucleus mora (0 = heiban), the NJD value.
    pub nucleus: u8,
    /// Realized level: 0 = L, 1 = H. The trained target.
    pub level: u8,
}

/// Phonemization of one utterance.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Phonemized {
    /// Readable source IPA, including any stress marks and word boundaries.
    /// Not for scoring; imported dictionary IPA retains its token separators.
    pub raw: String,
    /// Phoneme tokens supplied by the engine or imported from tokenized IPA.
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`; empty when unknown.
    pub stress: Vec<Stress>,
    /// `[start, end)` ranges into `phonemes`, one per word when known.
    pub word_spans: Vec<(usize, usize)>,
    /// Syllable spans (absolute indices into `phonemes`) for backends that
    /// compute them — Hindi. Empty for espeak languages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub syllables: Vec<Syllable>,
    /// Lexical tone per phoneme for tone languages — Mandarin: the tone
    /// number (1–5) on each syllable's tone-bearing phone, `None` elsewhere.
    /// Parallel to `phonemes`; empty for languages without tone labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tone: Vec<Option<u8>>,
    /// Tokyo pitch-accent factor per phoneme — Japanese. Parallel to
    /// `phonemes`; empty when withheld or for other languages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pitch: Vec<Option<Pitch>>,
    /// Why `pitch` is empty although the language has accent labels
    /// (Japanese): the phones are fine, the accent factor is not trusted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accent_withheld: Option<String>,
}

impl Phonemized {
    /// Import whitespace-separated IPA tokens, with optional `|` word boundaries.
    /// Preserves each token exactly (including affricates, diphthongs and marks).
    /// This is a structural conversion, not validation against a model vocabulary
    /// or a language-specific G2P transformation. Prosodic annotations are unknown.
    /// Without `|`, the input is treated as one word. Empty words are omitted.
    pub fn from_ipa_tokens(ipa: &str) -> Self {
        let mut result = Self {
            raw: ipa.to_owned(),
            ..Self::default()
        };
        for word in ipa.split('|') {
            let start = result.phonemes.len();
            result
                .phonemes
                .extend(word.split_whitespace().map(str::to_owned));
            let end = result.phonemes.len();
            if end > start {
                result.word_spans.push((start, end));
            }
        }
        result
    }
}

/// Where a language's phoneme labels come from. One table for both yap and
/// lexide: which G2P a language may use is a correctness constraint, not a
/// preference — targets from a different source than the model's training
/// labels disagree about the phoneme inventory, and nothing downstream can
/// tell (Hindi scored against espeak `hi` measured as the worst language by
/// a wide margin before this was understood).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LabelSource {
    /// Our espeak-ng fork. Voice selection is private to the engine.
    Espeak,
    /// The ported `schwa-stress-hin` chain ([`hindi`]).
    Hindi,
    /// The ported g2pM + pinyin-to-IPA chain ([`mandarin`]).
    Mandarin,
    /// OpenJTalk via `jpreprocess` ([`japanese`]).
    Japanese,
    /// vachana-thai, run as an embedded pinned Python project ([`thai`]);
    /// needs `uv` at runtime.
    Thai,
    /// g2pk2 + mecab-ko, run as an embedded pinned Python project
    /// ([`korean`]); needs `uv` at runtime.
    Korean,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imported_ipa_preserves_tokens_without_inventing_prosody() {
        let target = Phonemized::from_ipa_tokens(" | ˈt͡ʃ oʊ | ts ãː || ");
        assert_eq!(target.phonemes, ["ˈt͡ʃ", "oʊ", "ts", "ãː"]);
        assert_eq!(target.word_spans, [(0, 2), (2, 4)]);
        assert!(target.stress.is_empty() && target.tone.is_empty() && target.pitch.is_empty());
        assert!(Phonemized::from_ipa_tokens(" | ").phonemes.is_empty());
    }
}
