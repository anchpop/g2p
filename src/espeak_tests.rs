//! End-to-end tests against the embedded espeak-ng fork. No environment
//! setup needed: the engine and its data are inside the test binary.

use crate as g2p;
use g2p::{Stress, phonemize};

fn bare(text: &str, voice: &str) -> Vec<String> {
    phonemize(text, voice).unwrap().phonemes
}

#[test]
fn french_liaison_is_phrase_level() {
    // The motivating case: "on est" is /ɔ̃ n ɛ/ in connected speech — the
    // nasal vowel survives and the liaison /n/ appears. No per-word
    // dictionary produces that.
    let p = bare("on est", "fr-fr");
    assert!(p.contains(&"ɔ̃".to_string()), "{p:?}");
    assert!(p.contains(&"n".to_string()), "{p:?}");
    assert!(p.contains(&"ɛ".to_string()), "{p:?}");
}

#[test]
fn raw_keeps_boundaries_and_stress() {
    let r = phonemize("on est là", "fr-fr").unwrap();
    assert!(r.raw.contains(' '), "{:?}", r.raw);
    assert!(r.raw.contains('ˈ'), "{:?}", r.raw);
    assert!(!r.raw.contains('\n'));
    assert_eq!(r.phonemes.len(), r.stress.len());
    assert!(r.stress.contains(&Stress::Primary));
    assert_eq!(r.word_spans.last().unwrap().1, r.phonemes.len());
}

#[test]
fn leading_dash_is_text_not_options() {
    // Subtitle dialogue dashes are ~16% of film cues; the CLI once parsed
    // "- Bonjour" as flags and returned nothing.
    assert_eq!(bare("- Bonjour", "fr-fr"), bare("Bonjour", "fr-fr"));
}

#[test]
fn commas_do_not_split_an_utterance() {
    // espeak emits one line per clause; the crate joins them back into one
    // utterance, so a comma sentence yields one result, not two.
    let r = phonemize("Oui, bien sûr.", "fr-fr").unwrap();
    assert!(r.word_spans.len() >= 3, "{r:?}");
}

#[test]
fn newlines_are_spaces() {
    assert_eq!(bare("on\nest", "fr-fr"), bare("on est", "fr-fr"));
}

#[test]
fn empty_and_punctuation_only_input_is_ok() {
    assert!(bare("", "fr-fr").is_empty());
    assert!(bare("...", "fr-fr").is_empty());
}

#[test]
fn unknown_voice_is_an_error() {
    assert!(matches!(
        phonemize("hello", "xx-nope"),
        Err(g2p::Error::UnknownVoice(_))
    ));
    // And the engine still works afterwards.
    assert!(!bare("hello", "en-us").is_empty());
}

#[test]
fn switching_voices_leaves_no_state_behind() {
    let a1 = phonemize("Je ne sais pas.", "fr-fr").unwrap();
    let _ = phonemize("I don't know.", "en-us").unwrap();
    let _ = phonemize("Я не знаю.", "ru").unwrap();
    let a2 = phonemize("Je ne sais pas.", "fr-fr").unwrap();
    assert_eq!(a1, a2);
}

#[test]
fn language_switch_markers_do_not_leak_letters() {
    // A loanword makes espeak switch voices mid-sentence and bracket it as
    // "(en)…(fr)"; neither the parentheses nor the codes may become tokens.
    let r = phonemize("le football", "fr-fr").unwrap();
    assert!(!r.phonemes.iter().any(|p| p == "(" || p == ")"), "{r:?}");
}

#[test]
fn russian_palatalization_is_one_token() {
    let p = bare("тень", "ru");
    assert!(p.iter().any(|t| t.ends_with('ʲ')), "{p:?}");
    assert!(!p.iter().any(|t| t == "ʲ"), "{p:?}");
}

#[test]
fn mandarin_has_tones() {
    // Tone marks come from the pitch pass the CLI runs; the shortcut
    // `espeak_TextToPhonemes` API would miss tone sandhi.
    let r = phonemize("你好", "cmn").unwrap();
    assert!(!r.phonemes.is_empty(), "{r:?}");
}

#[test]
fn identity_is_stable_and_specific() {
    let id = g2p::identity();
    assert!(id.starts_with("g2p/"), "{id}");
    assert!(id.contains("espeak-ng/"), "{id}");
    assert_eq!(id, g2p::identity());
}

#[test]
fn concurrent_calls_are_serialized_and_correct() {
    let expected = bare("Bonjour madame", "fr-fr");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let expected = expected.clone();
            std::thread::spawn(move || {
                for _ in 0..20 {
                    assert_eq!(bare("Bonjour madame", "fr-fr"), expected);
                    assert!(!bare("good morning", "en-us").is_empty());
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn requested_words_use_merged_inventory() {
    for (voice, text, unit) in [
        ("en-us", "day", "eɪ"),
        ("en-us", "my", "aɪ"),
        ("en-us", "boy", "ɔɪ"),
        ("en-us", "now", "aʊ"),
        ("en-us", "go", "oʊ"),
        ("en-gb", "go", "əʊ"),
        ("en-us", "car", "ɑːɹ"),
        ("en-us", "more", "ɔːɹ"),
        ("en-us", "ear", "ɪɹ"),
        ("en-us", "air", "ɛɹ"),
        ("en-us", "tour", "ʊɹ"),
        ("de", "Häuser", "ɔø"),
        ("de", "Haus", "aʊ"),
        ("de", "Eis", "aɪ"),
        ("pt-br", "pão", "ɐ̃ʊ̃"),
        ("pt-br", "mãe", "ɐ̃j"),
        ("pt-br", "põe", "õɪ̃"),
        ("pt-br", "muito", "ũɪ̃"),
        ("pt-br", "mau", "aʊ"),
        ("pt-br", "sei", "eɪ"),
        ("pt-br", "sou", "oʊ"),
        ("pt-br", "pai", "aɪ"),
        ("cs", "dej", "eɪ"),
        ("cs", "ou", "oʊ"),
        ("cs", "auto", "aʊ"),
        ("en-us", "church", "tʃ"),
        ("en-us", "judge", "dʒ"),
        ("it", "cielo", "tʃ"),
        ("it", "giorno", "dʒ"),
        ("it", "zio", "dz"),
        ("it", "mezzo", "dzː"),
        ("pt-br", "tchau", "tʃ"),
        ("pt-br", "dia", "dʒ"),
        ("ru", "чай", "tʃʲ"),
        ("ru", "царь", "ts"),
        ("de", "Zeit", "ts"),
        ("de", "Pfad", "pf"),
        ("fa", "چای", "tʃ"),
        ("ar", "جميل", "dʒ"),
    ] {
        let p = phonemize(text, voice).unwrap();
        assert!(
            p.phonemes.iter().any(|p| p == unit),
            "{voice} {text}: {p:?}, expected {unit}"
        );
        assert_eq!(p.phonemes.len(), p.stress.len());
        assert_eq!(p.word_spans.last().unwrap().1, p.phonemes.len());
        assert_eq!(p.raw, g2p::phonemize_raw(text, voice).unwrap());
        assert!(!p.raw.contains('\u{1f}'));
    }
}

#[test]
fn source_artifacts_are_fixed_without_global_digit_or_caret_stripping() {
    let fa = phonemize("قهوه", "fa").unwrap();
    assert_eq!(fa.raw, "qˈahveː");
    assert_eq!(fa.phonemes, ["q", "a", "h", "v", "eː"]);
    let ru = phonemize("царь", "ru").unwrap();
    assert_eq!(ru.raw, "tsˈɑrɪ");
    assert_eq!(ru.phonemes, ["ts", "ɑ", "r", "ɪ"]);
    assert_eq!(g2p::parse::parse("q1 ɪ^").phonemes, ["q", "1", "ɪ", "^"]);
}

#[test]
fn actual_phone_and_word_boundaries_protect_clusters_and_onsets() {
    let p = phonemize("cat ship", "en-us").unwrap();
    assert!(!p.phonemes.iter().any(|p| p == "tʃ"), "{p:?}");
    let p = phonemize("cats", "en-us").unwrap();
    assert!(!p.phonemes.iter().any(|p| p == "ts"), "{p:?}");
    for word in ["mirror", "hero"] {
        let p = phonemize(word, "en-us").unwrap();
        assert!(!p.phonemes.iter().any(|p| p == "ɪɹ"), "{p:?}");
    }
    let p = phonemize("day my. Boy now!", "en-us").unwrap();
    assert_eq!(p.word_spans.len(), 4, "{p:?}");
    let mut end = 0;
    for (start, next) in p.word_spans {
        assert_eq!(start, end);
        assert!(next > start);
        end = next;
    }
    assert_eq!(end, p.phonemes.len());
}

#[test]
fn voice_aliases_use_resolved_language() {
    assert_eq!(bare("go", "English (America)"), bare("go", "en-us"));
    assert_eq!(bare("go", "en-us+f3"), bare("go", "en-us"));
    assert!(bare("go", "English (Great Britain)").contains(&"əʊ".into()));
}

#[test]
fn british_nonrhotic_outputs_are_not_invented() {
    for (text, expected) in [
        ("car", vec!["k", "ɑː"]),
        ("air", vec!["e", "ə"]),
        ("tour", vec!["t", "ʊ", "ə"]),
    ] {
        assert_eq!(bare(text, "en-gb"), expected);
    }
}

#[test]
fn explicit_ipa_ties_remain_inside_affricate_tokens() {
    for (voice, text, expected) in [
        (
            "lv",
            "cits četri",
            vec!["t͡s", "i", "t", "s", "t͡ʃ", "e", "t", "r", "i"],
        ),
        (
            "be",
            "цар дзякуй",
            vec!["t͡s", "a", "r", "d͡zʲ", "a", "k", "u", "j"],
        ),
        ("ps", "چای", vec!["t͡ʃ", "aː", "iː"]),
    ] {
        let p = phonemize(text, voice).unwrap();
        assert_eq!(p.phonemes, expected, "{voice} {text}: {p:?}");
        assert_eq!(p.stress.len(), p.phonemes.len());
        assert_eq!(p.word_spans.last().unwrap().1, p.phonemes.len());
        assert!(p.raw.contains('͡'));
    }
}

#[test]
fn typed_varieties_replay_the_corresponding_engine_output() {
    use g2p::{PhonemizeRequest, Variety, phonemize_language};
    for (lang, variety, voice, text) in [
        ("spa", Variety::Default, "es", "cinco"),
        ("spa", Variety::European, "es", "cinco"),
        ("spa", Variety::LatinAmerican, "es-419", "cinco"),
        ("por", Variety::Default, "pt-br", "dia noite"),
        ("por", Variety::Brazilian, "pt-br", "dia noite"),
        ("por", Variety::European, "pt", "dia noite"),
    ] {
        let actual =
            phonemize_language(PhonemizeRequest::new(lang, text).variety(variety)).unwrap();
        let expected = phonemize(text, voice).unwrap();
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
    }
}

#[test]
fn every_mapping_is_reachable_and_defaults_preserve_engine_output() {
    use g2p::{PhonemizeRequest, Variety, phonemize_language};
    let mut selections = Vec::new();
    for &(lang, variety, voice) in crate::voices::ESPEAK_VOICES {
        assert_eq!(crate::label_source(lang), Some(g2p::LabelSource::Espeak));
        assert_eq!(crate::variety_voice(lang, variety).unwrap(), voice);
        assert!(
            !selections.contains(&(lang, variety)),
            "duplicate selection: {lang} {variety:?}"
        );
        selections.push((lang, variety));
        assert!(
            crate::voices::ESPEAK_VOICES
                .iter()
                .any(|(language, candidate, _)| {
                    *language == lang && *candidate == Variety::Default
                }),
            "missing default for {lang}"
        );
        if variety == Variety::Default {
            let actual = phonemize_language(PhonemizeRequest::new(lang, "")).unwrap();
            assert_eq!(actual, phonemize("", voice).unwrap(), "{lang}");
        }
    }
}
