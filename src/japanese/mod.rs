//! Japanese grapheme-to-phoneme: OpenJTalk, via the `jpreprocess` rewrite.
//!
//! espeak's `ja` voice is not the label source for Japanese: kanji readings
//! are lexical, not derivable from the glyphs, and pitch accent needs a
//! dictionary. lexide labels Japanese with pyopenjtalk (OpenJTalk's text
//! front end over the NAIST dictionary). `jpreprocess` is a Rust rewrite of
//! that front end with the same dictionary bundled, so this module runs it
//! in-process and then applies lexide's `japanese_labels`:
//!
//! * OpenJTalk phones map to IPA (`sh` → `ɕ`, `N` → `ɴ`, devoiced `I`/`U` →
//!   `i̥`/`ɯ̥ᵝ`, …); the sokuon closure phone `cl` becomes `ː` on the
//!   following obstruent rather than a token of its own;
//! * each mora-bearing phone gets a Tokyo pitch level (0 = L, 1 = H) derived
//!   from its accent phrase's nucleus position and mora count, plus the raw
//!   position fields the acoustic validator reasons about;
//! * accent supervision is withheld — phones kept — when NJD and the HTS
//!   labels disagree on phrase boundaries, when the utterance is a fragment
//!   whose first content word is a particle/auxiliary/suffix (Pimsleur
//!   backward-buildup drills), or when there is no content word at all;
//! * a closure that cannot geminate (あっ、) or ends the utterance (早っ) has
//!   no phone to attach to, so the row is refused.
//!
//! Digits and Latin letters are read by the front end (十九時, エーオーエル),
//! so unlike Hindi and Mandarin they are not refused.

use crate::{Error, Pitch};
use jpreprocess::{JPreprocess, SystemDictionaryConfig, kind::JPreprocessDictionaryKind};
use std::sync::OnceLock;

pub use g2p_types::japanese::Labels;

fn phone_ipa(phone: &str) -> Option<&'static str> {
    Some(match phone {
        "a" => "a",
        "i" => "i",
        "I" => "i̥",
        "u" => "ɯᵝ",
        "U" => "ɯ̥ᵝ",
        "e" => "e",
        "o" => "o",
        "k" => "k",
        "ky" => "kʲ",
        "g" => "ɡ",
        "gy" => "ɡʲ",
        "s" => "s",
        "sh" => "ɕ",
        "z" => "z",
        "j" => "dʑ",
        "t" => "t",
        "ty" => "tʲ",
        "ch" => "tɕ",
        "ts" => "ts",
        "d" => "d",
        "dy" => "dʲ",
        "n" => "n",
        "ny" => "ɲ",
        "h" => "h",
        "hy" => "ç",
        "f" => "ɸ",
        "b" => "b",
        "by" => "bʲ",
        "p" => "p",
        "py" => "pʲ",
        "m" => "m",
        "my" => "mʲ",
        "y" => "j",
        "r" => "ɾ",
        "ry" => "ɾʲ",
        "w" => "w",
        "N" => "ɴ",
        "v" => "v",
        _ => return None,
    })
}

/// Phones a sokuon can geminate: the obstruents. A closure before anything
/// else is a glottal cutoff, not a geminate.
fn geminable(phone: &str) -> bool {
    matches!(
        phone,
        "k" | "ky"
            | "g"
            | "gy"
            | "s"
            | "sh"
            | "z"
            | "j"
            | "t"
            | "ty"
            | "ch"
            | "ts"
            | "d"
            | "dy"
            | "h"
            | "hy"
            | "f"
            | "b"
            | "by"
            | "p"
            | "py"
            | "v"
    )
}

/// Realized pitch of one mora of a Tokyo accent phrase (0 = L, 1 = H): the
/// nucleus position plus the initial-lowering rule. Odaka and heiban give the
/// same in-phrase contour; they differ only on the following particle, which
/// this phrase's audio does not carry.
pub fn tokyo_pitch_level(mora: u8, nucleus: u8, phrase_moras: u8) -> u8 {
    if phrase_moras <= 1 {
        return (nucleus == 1) as u8;
    }
    if nucleus == 1 {
        return (mora == 1) as u8;
    }
    if nucleus == 0 {
        return (mora != 1) as u8;
    }
    (2 <= mora && mora <= nucleus) as u8
}

fn engine() -> Result<&'static JPreprocess<jpreprocess::DefaultTokenizer>, Error> {
    static ENGINE: OnceLock<Result<JPreprocess<jpreprocess::DefaultTokenizer>, String>> =
        OnceLock::new();
    ENGINE
        .get_or_init(|| {
            SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic)
                .load()
                .map(|dict| JPreprocess::with_dictionaries(dict, None))
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| Error::Init(format!("jpreprocess: {e}")))
}

struct AccentPhrase {
    nucleus: u8,
    moras: u8,
}

/// Label one utterance.
pub fn phonemize(text: &str) -> Result<Labels, Error> {
    let jp = engine()?;
    let mut njd = jp
        .text_to_njd(text)
        .map_err(|e| Error::Synth(format!("jpreprocess: {e}")))?;
    njd.preprocess();

    // Accent phrases from the NJD features, before JPCommon rewrites a flat
    // accent (0) to the mora count: heiban and final-mora accent must stay
    // distinguishable.
    let mut phrases: Vec<AccentPhrase> = Vec::new();
    let mut boundary = true;
    let mut first_content: Option<String> = None;
    for node in &njd.nodes {
        let moras = node.get_pron().mora_size();
        let pos = node.get_pos().to_string();
        if moras == 0 {
            // Punctuation can force a short pause even when the next word
            // keeps chain_flag = 1.
            boundary = true;
            continue;
        }
        if first_content.is_none() && !pos.starts_with("記号") {
            first_content = Some(pos.clone());
        }
        if boundary || phrases.is_empty() || node.get_chain_flag() != Some(true) {
            phrases.push(AccentPhrase {
                nucleus: node.get_pron().accent() as u8,
                moras: moras as u8,
            });
        } else {
            phrases.last_mut().unwrap().moras += moras as u8;
        }
        boundary = false;
    }

    // Keep each native phone attached to its NJD node, and verify the complete
    // sequence against JPCommon below (not merely a matching phone count).
    let mut node_phones: Vec<(String, usize)> = Vec::new();
    let mut surfaces = Vec::new();
    for node in &njd.nodes {
        if node.get_pron().mora_size() == 0 {
            continue;
        }
        let owner = surfaces.len();
        surfaces.push(node.get_string().to_owned());
        for mora in node.get_pron().moras() {
            let (consonant, vowel) = mora.phonemes();
            if let Some(c) = consonant {
                let mut phone = c.to_string();
                if phone == "-" {
                    phone = node_phones
                        .last()
                        .map(|(p, _)| p.clone())
                        .unwrap_or_default();
                }
                node_phones.push((phone, owner));
            }
            if let Some(v) = vowel {
                let phone = if v.to_string() == "-" {
                    node_phones
                        .last()
                        .map(|(p, _)| p.clone())
                        .unwrap_or_default()
                } else {
                    v.to_string()
                };
                node_phones.push((phone, owner));
            }
        }
    }
    let text_spans = normalized_spans(text, &surfaces);

    // make_label re-parses each NJD node's reading and unwraps it; kana that
    // does not split into moras (ヴぁ: katakana ヴ with a hiragana small ぁ)
    // panics there. The engine is read-only, so recovering is sound.
    let labels =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| jp.make_label(njd.into())))
            .map_err(|_| Error::Unlabelable("japanese_unparsable_kana".into()))?;
    if labels.len() < 2 {
        return Ok(Labels::default());
    }
    let native: Vec<String> = labels[1..labels.len() - 1]
        .iter()
        .map(|l| l.phoneme.c.clone().unwrap_or_default())
        .collect();

    let aligned = native
        .iter()
        .filter(|p| !matches!(p.as_str(), "pau" | "sil"))
        .eq(node_phones.iter().map(|(p, _)| p));
    let mut native_index = 0;
    let mut owners = Vec::new();
    let mut closure_owner = None;
    let mut phonemes: Vec<String> = Vec::new();
    let mut pitch: Vec<Option<Pitch>> = Vec::new();
    let mut geminate = false;
    let mut phrase = 0usize;
    let mut breath_group = 0usize;
    let mut current_key: Option<(usize, u8)> = None;
    for (i, phone) in native.iter().enumerate() {
        if phone == "pau" || phone == "sil" {
            breath_group += 1;
            current_key = None;
            continue;
        }
        let mut owner = node_phones
            .get(native_index)
            .map(|(_, owner)| (*owner, *owner));
        native_index += 1;
        if phone == "cl" {
            closure_owner = owner;
            if geminate {
                return Err(Error::Unlabelable("japanese_double_closure".into()));
            }
            geminate = true;
            continue;
        }
        let Some(mapped) = phone_ipa(phone) else {
            return Err(Error::Unlabelable(format!(
                "japanese_unknown_phone:{phone}"
            )));
        };
        let mut mapped = mapped.to_string();
        if geminate {
            owner = closure_owner
                .zip(owner)
                .map(|(closure, current)| (closure.0, current.1));
            if !geminable(phone) {
                return Err(Error::Unlabelable("japanese_ungeminable_closure".into()));
            }
            mapped.push('ː');
            geminate = false;
        }
        let label = &labels[i + 1];
        let (Some(mora), Some(ap)) = (&label.mora, &label.accent_phrase_curr) else {
            return Err(Error::Synth(format!(
                "jpreprocess label for {phone:?} lacks mora/accent-phrase fields"
            )));
        };
        let key = (breath_group, ap.accent_phrase_position_forward);
        if current_key != Some(key) {
            phrase += 1;
            current_key = Some(key);
        }
        if phrase == 0 || phrase > phrases.len() {
            return Err(Error::Synth("jpreprocess accent-phrase mismatch".into()));
        }
        let info = &phrases[phrase - 1];
        let bears_mora = mapped.starts_with(|c| "aiɯeo".contains(c)) || mapped == "ɴ";
        pitch.push(bears_mora.then(|| Pitch {
            phrase: phrase as u8,
            mora: mora.position_forward,
            phrase_moras: info.moras,
            nucleus: info.nucleus,
            level: tokyo_pitch_level(mora.position_forward, info.nucleus, info.moras),
        }));
        phonemes.push(mapped);
        owners.push(owner);
    }
    if geminate {
        return Err(Error::Unlabelable("japanese_terminal_sokuon".into()));
    }

    let mut word_spans: Vec<(usize, usize)> = Vec::new();
    let mut word_text_spans = Vec::new();
    if aligned {
        let mut previous = None;
        for (index, owner) in owners.into_iter().enumerate() {
            if previous == owner && !word_spans.is_empty() {
                word_spans.last_mut().unwrap().1 = index + 1;
            } else {
                word_spans.push((index, index + 1));
                word_text_spans.push(
                    owner
                        .and_then(|(start, end)| {
                            Some((text_spans.get(start)?.0, text_spans.get(end)?.1))
                        })
                        .unwrap_or_default(),
                );
            }
            previous = owner;
        }
    }
    if word_text_spans.contains(&(0, 0)) {
        word_text_spans.clear();
    }
    let mut out = Labels {
        word_spans,
        word_text_spans,
        phonemes,
        pitch: Vec::new(),
        accent_withheld: None,
        native_phones: native,
    };
    let inconsistent = pitch
        .iter()
        .flatten()
        .any(|p| !(1 <= p.mora && p.mora <= p.phrase_moras) || p.nucleus > p.phrase_moras);
    if inconsistent || phrase != phrases.len() {
        out.accent_withheld = Some("njd_hts_phrase_boundary_mismatch".into());
        return Ok(out);
    }
    match first_content.as_deref() {
        None => out.accent_withheld = Some("japanese_accent_no_content_word".into()),
        Some(pos) => {
            let mut parts = pos.split(',');
            let major = parts.next().unwrap_or("");
            let group1 = parts.next().unwrap_or("");
            if major == "助詞" || major == "助動詞" || group1 == "接尾" {
                out.accent_withheld = Some("japanese_accent_head_truncated".into());
            }
        }
    }
    if out.accent_withheld.is_none() {
        out.pitch = pitch;
    }
    Ok(out)
}

// Map normalization groups back to original UTF-8 boundaries. Rewrites of the
// NJD surfaces themselves (such as number readings) must still align exactly.
fn normalized_spans(text: &str, surfaces: &[String]) -> Vec<(usize, usize)> {
    use jpreprocess::normalize_text_for_naist_jdic as normalize;
    let mut normalized = String::new();
    let mut boundaries = std::collections::BTreeMap::new();
    let mut chars = text.char_indices().peekable();
    while let Some((byte, c)) = chars.next() {
        boundaries.insert(normalized.len(), byte);
        let mut end = byte + c.len_utf8();
        // NAIST normalization combines a following voicing mark with its kana.
        // Keep that exact input group indivisible in the byte map.
        if let Some(&(next, mark)) = chars.peek()
            && matches!(
                mark,
                '\u{3099}' | '\u{309a}' | '\u{309b}' | '\u{309c}' | '\u{ff9e}' | '\u{ff9f}'
            )
        {
            end = next + mark.len_utf8();
            chars.next();
        }
        normalized.push_str(&normalize(&text[byte..end]));
    }
    boundaries.insert(normalized.len(), text.len());
    if normalized != normalize(text) {
        return Vec::new();
    }
    crate::attribution::align(&normalized, surfaces.iter().map(String::as_str))
        .into_iter()
        .map(|(start, end)| Some((*boundaries.get(&start)?, *boundaries.get(&end)?)))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_phones_and_gemination() {
        let l = phonemize("学校").unwrap();
        assert_eq!(l.phonemes, ["ɡ", "a", "kː", "o", "o"]);
    }

    #[test]
    fn pitch_accent_distinguishes_hashi() {
        // 箸 (chopsticks) is atamadaka HL; 橋 (bridge) is LH.
        let hashi = |s: &str| {
            let l = phonemize(s).unwrap();
            assert!(l.accent_withheld.is_none(), "{l:?}");
            l.pitch
                .iter()
                .flatten()
                .map(|p| p.level)
                .collect::<Vec<_>>()
        };
        assert_eq!(hashi("箸"), [1, 0]);
        assert_eq!(hashi("橋"), [0, 1]);
    }

    #[test]
    fn fragments_withhold_accent_but_keep_phones() {
        let l = phonemize("ません").unwrap();
        assert!(!l.phonemes.is_empty());
        assert_eq!(
            l.accent_withheld.as_deref(),
            Some("japanese_accent_head_truncated")
        );
        assert!(l.pitch.is_empty());
    }

    #[test]
    fn terminal_sokuon_is_refused() {
        assert!(matches!(
            phonemize("早っ"),
            Err(Error::Unlabelable(r)) if r == "japanese_terminal_sokuon"
        ));
    }

    #[test]
    fn unparsable_kana_is_refused() {
        assert!(matches!(
            phonemize("ヴぁかやろう"),
            Err(Error::Unlabelable(r)) if r == "japanese_unparsable_kana"
        ));
    }

    #[test]
    fn numbers_and_latin_are_read() {
        let l = phonemize("19時にAOLと").unwrap();
        assert!(
            l.native_phones.join(" ").contains("j u u k u j i"),
            "{:?}",
            l.native_phones
        );
    }
}
