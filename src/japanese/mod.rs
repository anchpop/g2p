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

/// Labels for one utterance.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

    let labels = jp.make_label(njd.into());
    if labels.len() < 2 {
        return Ok(Labels::default());
    }
    let native: Vec<String> = labels[1..labels.len() - 1]
        .iter()
        .map(|l| l.phoneme.c.clone().unwrap_or_default())
        .collect();

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
        if phone == "cl" {
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
    }
    if geminate {
        return Err(Error::Unlabelable("japanese_terminal_sokuon".into()));
    }

    let mut out = Labels {
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
    fn numbers_and_latin_are_read() {
        let l = phonemize("19時にAOLと").unwrap();
        assert!(
            l.native_phones.join(" ").contains("j u u k u j i"),
            "{:?}",
            l.native_phones
        );
    }
}
