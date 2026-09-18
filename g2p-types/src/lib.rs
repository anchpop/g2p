//! Shared pronunciation labels, independent of native engines and backend features.

pub mod hindi;
pub mod japanese;
pub mod korean;
mod language;
mod phoneme;
pub use phoneme::{Phoneme, UnknownPhoneme};
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
    pub phonemes: Vec<Phoneme>,
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
    /// Cantonese/Vietnamese preserve eSpeak tone codes (1–7), including
    /// contextual code 7 (Cantonese high fall; Vietnamese clause-final ngang).
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
    /// Import space-separated phoneme tokens, with `|` word boundaries.
    /// Stress marks and syllable/liaison separators are retained in `raw`,
    /// not treated as segmental phones. Prosody remains unaligned. Every
    /// remaining token must belong to the inventory; unknown phones are errors.
    pub fn from_ipa_tokens(ipa: &str) -> Result<Self, UnknownPhoneme> {
        let words = ipa
            .split('|')
            .map(|word| {
                word.split_whitespace()
                    .map(|token| {
                        token
                            .chars()
                            .filter(|c| !matches!(c, 'ˈ' | 'ˌ' | '.' | '‿'))
                            .collect::<String>()
                    })
                    .filter(|token| !token.is_empty())
                    .map(|token| token.parse())
                    .collect::<Result<Vec<Phoneme>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut result = Self::from_words(words);
        result.raw = ipa.to_owned();
        Ok(result)
    }

    /// Construct a target directly from typed words, without reparsing IPA.
    /// Prosody is unknown; use the engine output when those annotations exist.
    pub fn from_words(words: impl IntoIterator<Item = Vec<Phoneme>>) -> Self {
        let mut result = Self::default();
        for word in words {
            let start = result.phonemes.len();
            result.phonemes.extend(word);
            let end = result.phonemes.len();
            if end > start {
                result.word_spans.push((start, end));
            }
        }
        result.raw = result
            .word_spans
            .iter()
            .map(|&(start, end)| result.phonemes[start..end].join(""))
            .collect::<Vec<_>>()
            .join(" ");
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
        let target = Phonemized::from_ipa_tokens(" | t͡ʃ oʊ | ts ãː || ").unwrap();
        assert_eq!(
            target
                .phonemes
                .iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>(),
            ["t͡ʃ", "oʊ", "ts", "ãː"]
        );
        assert_eq!(target.raw, " | t͡ʃ oʊ | ts ãː || ");
        assert_eq!(target.word_spans, [(0, 2), (2, 4)]);
        assert!(target.stress.is_empty() && target.tone.is_empty() && target.pitch.is_empty());
        assert!(
            Phonemized::from_ipa_tokens(" | ")
                .unwrap()
                .phonemes
                .is_empty()
        );
        assert!(Phonemized::from_ipa_tokens("unknown").is_err());
    }
}
