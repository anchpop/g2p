//! Shared pronunciation labels, independent of native engines and backend features.

pub mod hindi;
pub mod japanese;
pub mod korean;
pub mod mandarin;
pub mod parse;
pub mod thai;

pub use hindi::{LabelVersion as HindiLabels, Syllable};
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
    /// espeak's own IPA output — stress marks and word boundaries intact,
    /// clauses joined with single spaces. Readable; not for scoring.
    pub raw: String,
    /// Model-label tokenization of `raw` (see [`parse`]).
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`.
    pub stress: Vec<Stress>,
    /// `[start, end)` ranges into `phonemes`, one per word espeak emitted.
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

/// Where a language's phoneme labels come from. One table for both yap and
/// lexide: which G2P a language may use is a correctness constraint, not a
/// preference — targets from a different source than the model's training
/// labels disagree about the phoneme inventory, and nothing downstream can
/// tell (Hindi scored against espeak `hi` measured as the worst language by
/// a wide margin before this was understood).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LabelSource {
    /// Our espeak-ng fork, with this voice. Table entries can borrow static
    /// names; deserialization owns the name without requiring static input.
    Espeak(std::borrow::Cow<'static, str>),
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

/// A language's pronunciation variety, independent of backend voice names.
/// Currently only Spanish accepts a non-default variety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Variety {
    /// Preserve the language's established default labels.
    #[default]
    Default,
    /// Spanish with seseo (espeak `es-419`).
    LatinAmerican,
    /// European Spanish with distinción (espeak `es`).
    European,
}
