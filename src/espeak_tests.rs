//! End-to-end tests against the embedded espeak-ng fork. No environment
//! setup needed: the engine and its data are inside the test binary.

use crate as g2p;
use g2p::{Stress, phonemize_espeak};

fn bare(text: &str, voice: &str) -> Vec<String> {
    phonemize_espeak(text, voice)
        .unwrap()
        .phonemes
        .iter()
        .map(ToString::to_string)
        .collect()
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
    let r = phonemize_espeak("on est là", "fr-fr").unwrap();
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
    let r = phonemize_espeak("Oui, bien sûr.", "fr-fr").unwrap();
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
        phonemize_espeak("hello", "xx-nope"),
        Err(g2p::Error::UnknownVoice(_))
    ));
    // And the engine still works afterwards.
    assert!(!bare("hello", "en-us").is_empty());
}

#[test]
fn switching_voices_leaves_no_state_behind() {
    let a1 = phonemize_espeak("Je ne sais pas.", "fr-fr").unwrap();
    let _ = phonemize_espeak("I don't know.", "en-us").unwrap();
    let _ = phonemize_espeak("Я не знаю.", "ru").unwrap();
    let a2 = phonemize_espeak("Je ne sais pas.", "fr-fr").unwrap();
    assert_eq!(a1, a2);
}

#[test]
fn language_switch_markers_do_not_leak_letters() {
    // A loanword makes espeak switch voices mid-sentence and bracket it as
    // "(en)…(fr)"; neither the parentheses nor the codes may become tokens.
    let r = phonemize_espeak("le football", "fr-fr").unwrap();
    assert!(
        !r.phonemes
            .iter()
            .any(|p| p.as_str() == "(" || p.as_str() == ")"),
        "{r:?}"
    );
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
    // This unsupported raw eSpeak route emits inline tone digits. Production
    // Mandarin uses the dedicated backend and a separate tone field.
    let raw = g2p::phonemize_espeak_raw("你好", "cmn").unwrap();
    assert!(raw.chars().any(|c| c.is_ascii_digit()), "{raw:?}");
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
        ("pt-br", "põe", "õɪ̃"),
        ("pt-br", "muito", "ũɪ̃"),
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
        let p = phonemize_espeak(text, voice).unwrap();
        assert!(
            p.phonemes.iter().any(|p| p.as_str() == unit),
            "{voice} {text}: {p:?}, expected {unit}"
        );
        assert_eq!(p.phonemes.len(), p.stress.len());
        assert_eq!(p.word_spans.last().unwrap().1, p.phonemes.len());
        assert_eq!(p.raw, g2p::phonemize_espeak_raw(text, voice).unwrap());
        assert!(!p.raw.contains('\u{1f}'));
    }
}

#[test]
fn source_artifacts_are_fixed_and_digits_are_preserved() {
    let fa = phonemize_espeak("قهوه", "fa").unwrap();
    assert_eq!(fa.raw, "qˈahveː");
    assert_eq!(
        fa.phonemes.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        ["q", "a", "h", "v", "eː"]
    );
    let ru = phonemize_espeak("царь", "ru").unwrap();
    assert_eq!(ru.raw, "tsˈɑrɪ");
    assert_eq!(
        ru.phonemes.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        ["ts", "ɑ", "r", "ɪ"]
    );
    assert_eq!(g2p::parse::parse("q1 ɪ^").phonemes, ["q", "1", "ɪ"]);
}

#[test]
fn actual_phone_and_word_boundaries_protect_clusters_and_onsets() {
    let p = phonemize_espeak("cat ship", "en-us").unwrap();
    assert!(!p.phonemes.iter().any(|p| p.as_str() == "tʃ"), "{p:?}");
    let p = phonemize_espeak("cats", "en-us").unwrap();
    assert!(!p.phonemes.iter().any(|p| p.as_str() == "ts"), "{p:?}");
    for word in ["mirror", "hero"] {
        let p = phonemize_espeak(word, "en-us").unwrap();
        assert!(!p.phonemes.iter().any(|p| p.as_str() == "ɪɹ"), "{p:?}");
    }
    let p = phonemize_espeak("day my. Boy now!", "en-us").unwrap();
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
        let p = phonemize_espeak(text, voice).unwrap();
        assert_eq!(
            p.phonemes.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
            expected,
            "{voice} {text}: {p:?}"
        );
        assert_eq!(p.stress.len(), p.phonemes.len());
        assert_eq!(p.word_spans.last().unwrap().1, p.phonemes.len());
        assert!(p.raw.contains('͡'));
    }
}

#[test]
fn typed_languages_replay_the_corresponding_engine_output() {
    use g2p::{Language, phonemize};
    for (language, voice, text) in [
        (Language::SpanishEuro, "es", "cinco"),
        (Language::SpanishLatinAmerica, "es-419", "cinco"),
        (Language::PortugueseBrazil, "pt-br", "dia noite"),
        (Language::PortugueseEuro, "pt", "dia noite"),
    ] {
        assert_eq!(
            phonemize(language, text).unwrap(),
            phonemize_espeak(text, voice).unwrap()
        );
    }
}

#[test]
fn every_mapping_is_reachable() {
    let mut selections = Vec::new();
    for &(language, voice) in crate::voices::ESPEAK_VOICES {
        assert_eq!(
            crate::label_source(language.code()),
            Some(g2p::LabelSource::Espeak)
        );
        assert_eq!(crate::engine_voice(language), voice);
        assert!(
            !selections.contains(&language),
            "duplicate selection: {language:?}"
        );
        selections.push(language);
        assert_eq!(
            g2p::phonemize(language, "").unwrap(),
            phonemize_espeak("", voice).unwrap()
        );
    }
}

#[test]
fn english_refuses_hangul_instead_of_switching_to_korean() {
    for text in [
        "아니요, 그렇지만 그것은 충분해요. Listen and repeat.",
        "Hello ᄀ",
        "Hello ㄱ",
        "Hello ꥠ",
        "Hello ힰ",
        "Hello ﾡ",
    ] {
        assert!(matches!(g2p::phonemize_lang("eng", text),
            Err(g2p::Error::Unlabelable(reason)) if reason.starts_with("english_hangul:")));
    }
    assert!(g2p::phonemize_lang("eng", "Listen and repeat.").is_ok());
}

#[test]
fn cantonese_and_vietnamese_tones_are_aligned_metadata() {
    for (language, text, phones, tones) in [
        (
            g2p::Language::Cantonese,
            "你好",
            vec!["n", "e", "i", "h", "o", "u"],
            vec![None, None, Some(5), None, None, Some(2)],
        ),
        (
            g2p::Language::Vietnamese,
            "Xin chào",
            vec!["s", "i", "n", "tʃ", "aː", "w"],
            vec![None, Some(1), None, None, Some(2), None],
        ),
    ] {
        let result = g2p::phonemize(language, text).unwrap();
        assert_eq!(
            result
                .phonemes
                .iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>(),
            phones
        );
        assert_eq!(result.tone, tones);
        assert_eq!(result.stress.len(), result.phonemes.len());
        assert_eq!(result.word_spans, [(0, 3), (3, 6)]);
        assert!(result.raw.chars().any(|c| c.is_ascii_digit()));
        assert!(g2p::phonemize(language, "...").unwrap().tone.is_empty());
    }
}
