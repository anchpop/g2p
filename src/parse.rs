//! Tokenize espeak-ng's `--ipa -x` output into phonemes, stress, and word
//! spans — versioned labels for pronunciation-model training and scoring.
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
//! * The framed engine path merges affricates inside an actual espeak phone,
//!   and selected English/German/Brazilian Portuguese/Czech vowel units.
//!   Plain `parse` retains the legacy character segmentation: raw IPA alone
//!   cannot distinguish an affricate from adjacent stop/fricative phones.
//! * espeak brackets language switches with markers like `(en)` or `(en-us)`;
//!   these are stripped before tokenizing. Left in, the parentheses would be
//!   junk tokens and the letters would pass as real phonemes.

pub use g2p_types::parse::{Parsed, Stress};

/// IPA vowels (monophthongs and near-variants espeak emits across our
/// languages). Vowels carry stress; `ʲ` never folds onto one.
pub const IPA_VOWELS: &str = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ";

/// Combining marks and modifier letters that continue the preceding token:
/// length (ː ˑ), retracted, lowered, non-syllabic, voiceless (below/above),
/// nasalized, centralized, dental, syllabic, raised, and pharyngealization.
pub const CONTINUATIONS: &str = "ːˑ̠̞̯̥̪̩̝̃̊̈ˤ";

pub const WORD_BOUNDARIES: &str = " \t\n|_-";

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

/// Parse unframed `--ipa -x` output with legacy character segmentation.
/// Use [`crate::phonemize`] for current model labels: it also reads the
/// engine's phoneme boundaries, which cannot be recovered from raw IPA.
pub fn parse(raw: &str) -> Parsed {
    parse_impl(raw, None)
}

/// Private trace separator (not a word boundary). The renderer suppresses
/// separators before modifiers, matching our continuation/palatalization rule.
pub(crate) const PHONE_SEPARATOR: char = '\u{1f}';

pub(crate) fn parse_framed(raw: &str, language: &str) -> Parsed {
    let mut parsed = parse_impl(raw, Some(language));
    // Use the requested language even for eSpeak's code-switched loanwords.
    // These replacements are one-to-one, preserving every aligned field.
    for phone in &mut parsed.phonemes {
        let replacement = match (
            language.split('-').next().unwrap_or(language),
            phone.as_str(),
        ) {
            ("en", "ɐ" | "ᵻ") => "ə",
            ("fr", "uː" | "ʊ") => "u",
            ("fr", "ɔː" | "ɒ") => "ɔ",
            ("fr", "ɑː" | "aː" | "ʌ" | "ɐ") => "a",
            ("fr", "oː") => "o",
            ("fr", "iː" | "ɪ") => "i",
            ("fr", "yː") => "y",
            ("fr", "eː") => "e",
            ("fr", "ɜː") => "œ",
            ("it", "ɪ") => "i",
            ("it", "ʊ") => "u",
            _ => continue,
        };
        *phone = replacement.to_owned();
    }
    parsed
}

#[derive(Clone)]
struct Origin {
    phone: usize,
    // Stress and language switches are barriers even for cross-phone merges.
    barrier: usize,
    language: String,
}

fn parse_impl(raw: &str, language: Option<&str>) -> Parsed {
    let framed = language.is_some();
    let default_language = language.unwrap_or("").to_ascii_lowercase();
    let mut origin = Origin {
        phone: 0,
        barrier: 0,
        language: default_language.clone(),
    };
    let mut origins = Vec::new();
    let chars: Vec<char> = raw.chars().collect();
    let mut p = Parsed::default();
    let mut word_start = 0usize;
    let mut pending: Option<Stress> = None;
    let mut current = Stress::None;
    let mut in_vowel = false;

    let mut i = 0;
    while i < chars.len() {
        // Some Danish output spells IPA open-e with Greek epsilon.
        let ch = if chars[i] == 'ε' { 'ɛ' } else { chars[i] };
        if ch == '('
            && let Some(len) = marker_len(&chars[i..])
        {
            if framed {
                origin.language = chars[i + 1..i + len - 1]
                    .iter()
                    .collect::<String>()
                    .to_ascii_lowercase();
                // Switch markers name phoneme tables, not regional voices.
                // On return to (pt), restore the original pt-br policy.
                if default_language.split('-').next() == Some(origin.language.as_str()) {
                    origin.language.clone_from(&default_language);
                }
                origin.barrier += 1;
                origin.phone += 1;
            }
            i += len;
            continue;
        }
        i += 1;
        if framed && ch == PHONE_SEPARATOR {
            origin.phone += 1;
            // Do not reset stress: adjacent vowels historically share it.
            continue;
        }
        if "\"().?^".contains(ch) {
            // Punctuation/syllable separators are not phones. Discard attached
            // modifiers too: `.ː` must not lengthen the preceding vowel.
            while i < chars.len()
                && (CONTINUATIONS.contains(chars[i]) || chars[i] == PHONE_SEPARATOR)
            {
                i += 1;
            }
            origin.barrier += 1;
            in_vowel = false;
            continue;
        }
        let before = p.phonemes.len();
        if ch == 'ˈ' {
            origin.barrier += 1;
            pending = Some(Stress::Primary);
            in_vowel = false;
        } else if ch == 'ˌ' {
            origin.barrier += 1;
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
        if p.phonemes.len() > before {
            origins.push(origin.clone());
        }
    }
    if p.phonemes.len() > word_start {
        p.word_spans.push((word_start, p.phonemes.len()));
    }
    if framed { merge_units(p, &origins) } else { p }
}

fn english(language: &str) -> bool {
    language == "en" || language.starts_with("en-")
}

fn vowel_unit(unit: &str, language: &str) -> bool {
    if english(language) {
        matches!(
            unit,
            "eɪ" | "aɪ" | "ɔɪ" | "aʊ" | "oʊ" | "əʊ" | "ɑːɹ" | "ɔːɹ" | "ɪɹ" | "ɛɹ" | "ʊɹ"
        )
    } else {
        match language {
            "de" => matches!(unit, "aɪ" | "aʊ" | "ɔʏ" | "ɔø"),
            "pt-br" => matches!(
                unit,
                "aʊ" | "eɪ" | "oʊ" | "aɪ" | "ɐ̃ʊ̃" | "ɐ̃ɪ̃" | "ɐ̃j" | "õɪ̃" | "ũɪ̃"
            ),
            "cs" => matches!(unit, "eɪ" | "oʊ" | "aʊ"),
            _ => false,
        }
    }
}

fn affricate(left: &str, right: &str) -> bool {
    // Keep all existing length/continuation/palatalization detail. In
    // particular Italian dzː and Russian tʃʲ remain single decorated units.
    let base = |s: &str| {
        s.chars()
            .filter(|c| !CONTINUATIONS.contains(*c) && *c != 'ʲ')
            .collect::<String>()
    };
    matches!(
        (base(left).as_str(), base(right).as_str()),
        ("t", "ʃ" | "s" | "ɕ" | "ʂ" | "θ")
            | ("d", "ʒ" | "z" | "ʑ" | "ʐ" | "ð")
            | ("p", "f")
            | ("b", "v")
            | ("k", "x")
            | ("ɡ", "ɣ")
            | ("ʈ", "ʂ")
            | ("ɖ", "ʐ")
    )
}

fn merge_units(p: Parsed, origins: &[Origin]) -> Parsed {
    let mut out = Parsed::default();
    for (start, end) in p.word_spans {
        let word_start = out.phonemes.len();
        let mut i = start;
        while i < end {
            let mut unit = p.phonemes[i].clone();
            let mut count = 1;
            if i + 1 < end {
                // Some tables spell affricates with an explicit IPA tie.
                // Legacy parsing keeps that tie as a third token; consume it
                // only when ALL three pieces belong to the same engine phone.
                let tied = i + 2 < end && matches!(p.phonemes[i + 1].as_str(), "͡" | "͜");
                let width = if tied { 3 } else { 2 };
                let last = i + width - 1;
                let a = &origins[i];
                let b = &origins[last];
                let right = &p.phonemes[last];
                let joined = p.phonemes[i..=last].concat();
                let same_phone = a.phone == b.phone;
                // espeak spells more/ear as separate vowel + r phones and
                // mãe as ɐ̃ + j. Merge only a coda, never before another vowel
                // (mirror/hero). The trace has no full syllabification, so
                // this conservative exception intentionally misses some cases.
                let coda = i + 2 == end || !starts_with_vowel(&p.phonemes[i + 2]);
                let cross_phone = coda
                    && ((english(&a.language) && right == "ɹ")
                        || (a.language == "pt-br" && joined == "ɐ̃j"));
                if a.barrier == b.barrier
                    && a.language == b.language
                    && ((same_phone && affricate(&unit, right))
                        || (!tied
                            && (same_phone || cross_phone)
                            && vowel_unit(&joined, &a.language)))
                {
                    unit = joined;
                    count = width;
                }
            }
            out.phonemes.push(unit);
            out.stress.push(p.stress[i]);
            i += count;
        }
        out.word_spans.push((word_start, out.phonemes.len()));
    }
    out
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
    fn pronunciation_is_ready_for_consumers() {
        let p = parse_framed("ˈɐ .ː ˌᵻ ɪ ɐ̯ hʲ", "en-us");
        assert_eq!(p.phonemes, ["ə", "ə", "ɪ", "ɐ̯", "hʲ"]);
        assert_eq!(p.stress[..2], [Stress::Primary, Stress::Secondary]);
        assert_eq!(p.word_spans, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        assert_eq!(
            parse_framed("(en)uː ɔː ɑː oː aː iː yː eː ɜː ɪ ʊ ʌ ɒ ɐ", "fr-fr").phonemes,
            [
                "u", "ɔ", "a", "o", "a", "i", "y", "e", "œ", "i", "u", "a", "ɔ", "a"
            ]
        );
        assert_eq!(parse_framed("(en)ɪ ʊ ɪː", "it").phonemes, ["i", "u", "ɪː"]);
        assert_eq!(parse_framed("ˈε", "da").stress, [Stress::Primary]);
        assert_eq!(parse_framed("a.ːi", "en-us").phonemes, ["a", "i"]);
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

#[cfg(test)]
mod framed_tests {
    use super::*;

    fn framed(trace: &str, language: &str) -> Parsed {
        parse_framed(&trace.replace(';', &PHONE_SEPARATOR.to_string()), language)
    }

    #[test]
    fn separator_is_not_whitespace_or_a_word_boundary() {
        assert!(!PHONE_SEPARATOR.is_whitespace());
        assert!(!WORD_BOUNDARIES.contains(PHONE_SEPARATOR));
    }

    #[test]
    fn every_vowel_unit_and_its_near_misses() {
        let inventories = [
            (
                "en-us",
                vec![
                    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɑːɹ", "ɔːɹ", "ɪɹ", "ɛɹ", "ʊɹ",
                ],
            ),
            (
                "en-gb",
                vec![
                    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɑːɹ", "ɔːɹ", "ɪɹ", "ɛɹ", "ʊɹ",
                ],
            ),
            ("de", vec!["aɪ", "aʊ", "ɔʏ", "ɔø"]),
            (
                "pt-br",
                vec!["aʊ", "eɪ", "oʊ", "aɪ", "ɐ̃ʊ̃", "ɐ̃ɪ̃", "ɐ̃j", "õɪ̃", "ũɪ̃"],
            ),
            ("cs", vec!["eɪ", "oʊ", "aʊ"]),
        ];
        for (language, units) in inventories {
            for unit in units {
                let legacy = parse(unit).phonemes;
                assert_eq!(legacy.len(), 2, "{unit}");
                let p = framed(&format!("ˈ{unit}"), language);
                assert_eq!(p.phonemes, [unit], "{language}: {unit}");
                assert_eq!(p.stress, [Stress::Primary]);
                assert_eq!(p.word_spans, [(0, 1)]);
                // Identical spelling in another language is not sufficient.
                assert_eq!(framed(unit, "pl").phonemes, legacy);
                for boundary in WORD_BOUNDARIES.chars() {
                    let p = framed(&legacy.join(&boundary.to_string()), language);
                    assert_eq!(p.phonemes, legacy, "{language}: {unit} {boundary:?}");
                    assert_eq!(p.word_spans, [(0, 1), (1, 2)]);
                }
                assert_eq!(framed(&legacy.join("ˈ"), language).phonemes, legacy);
                assert_eq!(framed(&legacy.join("ˌ"), language).phonemes, legacy);
                // Ordinary diphthongs must be one engine phone; only coda-r
                // and the attested mãe glide have narrow cross-phone rules.
                if !unit.ends_with('ɹ') && unit != "ɐ̃j" {
                    assert_eq!(framed(&legacy.join(";"), language).phonemes, legacy);
                }
            }
        }
    }

    #[test]
    fn every_affricate_requires_an_actual_phone() {
        for unit in [
            "tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "tʂ", "dʐ", "pf", "bv", "tθ", "dð", "kx", "ɡɣ",
            "tʃʲ", "dzː", "tːs", "ʈʂ", "ɖʐ", "t͡s", "t͡ʃ", "t͡sʲ", "d͡z", "d͡zʲ", "ʈ͡ʂ", "ɖ͡ʐ", "d͡ʒ",
            "t͜s",
        ] {
            let legacy = parse(unit).phonemes;
            for language in [
                "en-us", "it", "pt-br", "ru", "de", "fa", "ar", "pl", "fr-fr",
            ] {
                assert_eq!(framed(unit, language).phonemes, [unit]);
                assert_eq!(framed(unit, language).stress, [Stress::None]);
                for delimiter in [";", " ", "ˈ", "ˌ", "(en)"] {
                    if legacy.len() == 3 {
                        // A separator on EITHER side of the tie disqualifies
                        // the complete three-part unit, not only two at once.
                        for split in 1..3 {
                            let raw = format!(
                                "{}{}{}",
                                legacy[..split].concat(),
                                delimiter,
                                legacy[split..].concat()
                            );
                            assert_eq!(framed(&raw, language).phonemes, legacy);
                        }
                    }
                    assert_eq!(
                        framed(&legacy.join(delimiter), language).phonemes,
                        legacy,
                        "{unit} in {language} with {delimiter:?}"
                    );
                }
            }
        }
        for near_miss in ["tɹ", "dɹ", "ps", "ks", "tʰ", "dʱ", "iʲo"] {
            assert_eq!(
                framed(near_miss, "en-us").phonemes,
                parse(near_miss).phonemes
            );
        }
    }

    #[test]
    fn coda_exceptions_do_not_swallow_onsets_or_switches() {
        for unit in ["ɑːɹ", "ɔːɹ", "ɪɹ", "ɛɹ", "ʊɹ"] {
            let parts = parse(unit).phonemes;
            let split = parts.join(";");
            assert_eq!(framed(&split, "en-us").phonemes, [unit]);
            assert_eq!(
                framed(&format!("{split};ə"), "en-us").phonemes,
                [parts[0].as_str(), "ɹ", "ə"]
            );
            assert_eq!(framed(&parts.join("(en)"), "en-us").phonemes, parts);
        }
        assert_eq!(framed("ˈɐ̃;j", "pt-br").phonemes, ["ɐ̃j"]);
        assert_eq!(framed("ɐ̃;j;a", "pt-br").phonemes, ["ɐ̃", "j", "a"]);
        assert_eq!(framed("ɐ̃;j", "pt").phonemes, ["ɐ̃", "j"]);
    }

    #[test]
    fn stress_length_tones_and_word_indices_stay_aligned() {
        let p = framed(";tʃ;ˈeɪ;t;s ;dʒ;ˌaʊ ;ˈa;ɪ ;iː;5", "en-us");
        assert_eq!(
            p.phonemes,
            ["tʃ", "eɪ", "t", "s", "dʒ", "aʊ", "a", "ɪ", "iː", "5"]
        );
        assert_eq!(p.word_spans, [(0, 4), (4, 6), (6, 8), (8, 10)]);
        assert_eq!(
            p.stress,
            [
                Stress::None,
                Stress::Primary,
                Stress::None,
                Stress::None,
                Stress::None,
                Stress::Secondary,
                Stress::Primary,
                Stress::Primary,
                Stress::None,
                Stress::None
            ]
        );
        for raw in ["iː5", "aˈɪ", "tʲ;ɪ;ɫʲ", "ː;a", "", ";;;", "  "] {
            assert_eq!(framed(raw, "cmn"), parse(&raw.replace(';', "")));
        }
    }

    #[test]
    fn language_switches_change_only_merge_policy() {
        let p = framed("aɪ;(en)ˈaɪ;(fr)aɪ", "fr-fr");
        assert_eq!(p.phonemes, ["a", "i", "aɪ", "a", "i"]);
        assert_eq!(
            p.stress,
            [
                Stress::None,
                Stress::None,
                Stress::Primary,
                Stress::Primary,
                Stress::Primary
            ]
        );
        assert_eq!(framed("a(en)ɪ", "en-us").phonemes, ["a", "ɪ"]);
        assert_eq!(
            framed("(en)ˈaɪ (pt)m;ˈɐ̃;j", "pt-br").phonemes,
            ["aɪ", "m", "ɐ̃j"]
        );
    }
}
