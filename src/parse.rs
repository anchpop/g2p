//! Tokenize espeak-ng's `--ipa -x` output into phonemes, stress, and word
//! spans — the segmentation the lexide pronunciation model was trained on.
//!
//! This is the one place that segmentation is defined. A model trained on
//! labels segmented one way cannot be scored against targets segmented
//! another, and the mismatch is silent (shapes match, distances compute,
//! results are simply wrong). Every consumer must go through this parser.
//!
//! Rules:
//! * Stress markers `ˈ`/`ˌ` are not tokens; the stress attaches to the next
//!   vowel nucleus.
//! * Word boundaries (space, tab, newline, `|`, `_`, `-`) are not tokens;
//!   they delimit `word_spans`.
//! * Continuation diacritics (length, nasalization, dental, syllabic, …)
//!   append to the preceding token, so `ɛ̃`/`iː`/`t̪` are single tokens.
//! * Palatalization `ʲ` appends to a preceding *consonant* (Russian `tʲ`,
//!   `ɫʲ`, `ʃʲ`) but stays its own token after a vowel, where espeak uses it
//!   for a hiatus glide (Italian "io" = `iʲo`).
//! * Everything else — vowel or consonant, including each half of a
//!   diphthong — is its own token.
//! * espeak brackets language switches with markers like `(en)` or `(en-us)`;
//!   these are stripped before tokenizing. Left in, the parentheses would be
//!   junk tokens and the letters would pass as real phonemes.

use serde::{Deserialize, Serialize};

/// IPA vowels (monophthongs and near-variants espeak emits across our
/// languages). Vowels carry stress; `ʲ` never folds onto one.
pub const IPA_VOWELS: &str = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ";

/// Combining marks and modifier letters that continue the preceding token:
/// length (ː ˑ), retracted, lowered, non-syllabic, voiceless (below/above),
/// nasalized, centralized, dental, syllabic, raised, and pharyngealization.
pub const CONTINUATIONS: &str = "ːˑ̠̞̯̥̪̩̝̃̊̈ˤ";

pub const WORD_BOUNDARIES: &str = " \t\n|_-";

/// Lexical stress of a token, as espeak marked it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stress {
    None,
    Primary,
    Secondary,
}

impl Stress {
    /// The integer code lexide's corpus files use (0/1/2).
    pub fn code(self) -> u8 {
        match self {
            Stress::None => 0,
            Stress::Primary => 1,
            Stress::Secondary => 2,
        }
    }
}

/// A parsed utterance. `phonemes` and `stress` are parallel; each
/// `word_spans` entry is a half-open `[start, end)` index range into them
/// for one word espeak emitted (empty words are dropped).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Parsed {
    pub phonemes: Vec<String>,
    pub stress: Vec<Stress>,
    pub word_spans: Vec<(usize, usize)>,
}

fn is_vowel(c: char) -> bool {
    IPA_VOWELS.contains(c)
}

fn starts_with_vowel(token: &str) -> bool {
    token.chars().next().is_some_and(is_vowel)
}

/// Remove `(xx)` / `(xx-yy)` language-switch markers (2–4 ASCII letters per
/// part, any case).
pub fn strip_language_markers(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '('
            && let Some(len) = marker_len(&chars[i..])
        {
            i += len;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Length of a language marker starting at `s[0] == '('`, if there is one.
fn marker_len(s: &[char]) -> Option<usize> {
    let mut i = 1;
    let letters = |i: &mut usize| {
        let start = *i;
        while *i < s.len() && s[*i].is_ascii_alphabetic() {
            *i += 1;
        }
        (2..=4).contains(&(*i - start))
    };
    if !letters(&mut i) {
        return None;
    }
    if i < s.len() && s[i] == '-' {
        i += 1;
        if !letters(&mut i) {
            return None;
        }
    }
    (i < s.len() && s[i] == ')').then_some(i + 1)
}

/// Parse one utterance of espeak `--ipa -x` output.
pub fn parse(raw: &str) -> Parsed {
    let raw = strip_language_markers(raw);
    let mut p = Parsed::default();
    let mut word_start = 0usize;
    let mut pending: Option<Stress> = None;
    let mut current = Stress::None;
    let mut in_vowel = false;

    for ch in raw.chars() {
        if ch == 'ˈ' {
            pending = Some(Stress::Primary);
            in_vowel = false;
        } else if ch == 'ˌ' {
            pending = Some(Stress::Secondary);
            in_vowel = false;
        } else if WORD_BOUNDARIES.contains(ch) {
            if p.phonemes.len() > word_start {
                p.word_spans.push((word_start, p.phonemes.len()));
            }
            word_start = p.phonemes.len();
            pending = None;
            current = Stress::None;
            in_vowel = false;
        } else if is_vowel(ch) {
            if let Some(s) = pending.take() {
                current = s;
            } else if !in_vowel {
                current = Stress::None;
            }
            in_vowel = true;
            p.phonemes.push(ch.to_string());
            p.stress.push(current);
        } else if CONTINUATIONS.contains(ch) {
            match p.phonemes.last_mut() {
                Some(last) => last.push(ch),
                // Stray diacritic with nothing to attach to: keep it rather
                // than dropping detail silently.
                None => {
                    p.phonemes.push(ch.to_string());
                    p.stress.push(Stress::None);
                }
            }
        } else if ch == 'ʲ' {
            match p.phonemes.last_mut() {
                Some(last) if !starts_with_vowel(last) => last.push(ch),
                _ => {
                    in_vowel = false;
                    current = Stress::None;
                    p.phonemes.push(ch.to_string());
                    p.stress.push(Stress::None);
                }
            }
        } else {
            in_vowel = false;
            current = Stress::None;
            p.phonemes.push(ch.to_string());
            p.stress.push(Stress::None);
        }
    }
    if p.phonemes.len() > word_start {
        p.word_spans.push((word_start, p.phonemes.len()));
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phonemes(raw: &str) -> Vec<String> {
        parse(raw).phonemes
    }

    #[test]
    fn stress_and_boundaries_are_not_tokens() {
        assert_eq!(phonemes("ˈɔ̃ n ɛ"), vec!["ɔ̃", "n", "ɛ"]);
        assert_eq!(phonemes("sˈiː aɪ"), vec!["s", "iː", "a", "ɪ"]);
        assert_eq!(phonemes("wˌi\nɡˈoʊ"), vec!["w", "i", "ɡ", "o", "ʊ"]);
        assert_eq!(phonemes("ːa"), vec!["ː", "a"]);
    }

    #[test]
    fn palatalization_folds_onto_consonants_only() {
        assert_eq!(
            phonemes("tʲinʲ ɫʲ iʲo"),
            vec!["tʲ", "i", "nʲ", "ɫʲ", "i", "ʲ", "o"]
        );
    }

    #[test]
    fn stress_attaches_to_the_next_nucleus() {
        let p = parse("bɔ̃ʒˈuʁ mˌadam");
        assert_eq!(
            p.phonemes,
            vec!["b", "ɔ̃", "ʒ", "u", "ʁ", "m", "a", "d", "a", "m"]
        );
        assert_eq!(
            p.stress,
            vec![
                Stress::None,
                Stress::None,
                Stress::None,
                Stress::Primary,
                Stress::None,
                Stress::None,
                Stress::Secondary,
                Stress::None,
                Stress::None,
                Stress::None,
            ]
        );
        assert_eq!(p.word_spans, vec![(0, 5), (5, 10)]);
    }

    #[test]
    fn diphthong_second_half_keeps_stress() {
        // A vowel directly after a stressed vowel is the same nucleus.
        let p = parse("ˈaɪ");
        assert_eq!(p.stress, vec![Stress::Primary, Stress::Primary]);
    }

    #[test]
    fn language_switch_markers_are_stripped() {
        assert_eq!(strip_language_markers("(en)fˈʊtbɔːl(fr)"), "fˈʊtbɔːl");
        assert_eq!(strip_language_markers("(en-us)a(fr-FR)"), "a");
        // Not markers: too short, too long, or unterminated.
        assert_eq!(strip_language_markers("(e)a"), "(e)a");
        assert_eq!(strip_language_markers("(abcde)a"), "(abcde)a");
        assert_eq!(strip_language_markers("(en"), "(en");
        assert_eq!(phonemes("(en)fˈʊt(fr)"), vec!["f", "ʊ", "t"]);
    }

    #[test]
    fn empty_words_leave_no_spans() {
        let p = parse("  a  b ");
        assert_eq!(p.word_spans, vec![(0, 1), (1, 2)]);
        assert!(parse("").word_spans.is_empty());
    }
}
