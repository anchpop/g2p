//! Hindi label conventions and word/syllable structure.

use crate::Stress;

/// Which label convention to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelVersion {
    /// Byte-identical to lexide's `schwa-stress-hin` (provider schema 5): what
    /// the deployed pronunciation model was trained on. Use this to score
    /// against that model.
    Legacy,
    /// Legacy plus the audited corrections:
    /// * `/ə/` beside `/ɦ/` is `[ɛ]` (शहर, कहना, बहन, जगह) and यह/वह are
    ///   `[jeː]`/`[ʋoː]` — Legacy wrote `ə` in 39% of corpus rows. Applied
    ///   uniformly: it is near-categorical in the native and function words
    ///   that carry most of those tokens, while careful readings of Sanskrit
    ///   compounds (आग्रह, असहयोग) may keep `[ə]`; per-clip realization is
    ///   an acoustic-narrowing question, not a G2P one;
    /// * word-final short ɪ/ʊ are `iː`/`uː` (no length contrast there);
    /// * anusvara before a velar is `ŋ` (संकट), as before other stops it is
    ///   already homorganic — Legacy nasalized the vowel before क/ख only;
    /// * ज्ञ is `[ɡj]` (ज्ञान), not `d͡ʒɲ`;
    /// * a schwa deletion that would create an unpronounceable consonant run
    ///   (दुश्मनों → `ʃmn`) is undone;
    /// * digits and Latin script are an error rather than silently missing
    ///   from the labels while present in the audio.
    Current,
}

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
    pub phonemes: Vec<String>,
    pub stress: Vec<Stress>,
    pub syllables: Vec<Syllable>,
    /// One entry per orthographic schwa: whether the classifier kept it.
    pub schwa_retained: Vec<bool>,
}
