//! Unified language dispatch preserves the existing voice and backend paths.
use g2p::{Error, Language, phonemize, phonemize_lang};

fn assert_same_bytes(left: g2p::Phonemized, right: g2p::Phonemized) {
    assert_eq!(
        serde_json::to_vec(&left).unwrap(),
        serde_json::to_vec(&right).unwrap()
    );
}

#[test]
fn spanish_varieties_distinguish_seseo() {
    for (language, initial) in [
        (Language::SpanishEuro, "θ"),
        (Language::SpanishLatinAmerica, "s"),
    ] {
        let result = phonemize(language, "cinco").unwrap();
        assert_eq!(result.phonemes.first().unwrap().as_str(), initial);
    }
}

#[test]
fn espeak_wrappers_preserve_full_default_output() {
    for (lang, text) in [
        ("eng", "church cats more mirror"),
        ("deu", "Häuser Zeit"),
        ("fra", "on est"),
        ("ita", "pizza"),
        ("por", "mãe pão"),
        ("spa", "cinco"),
        ("rus", "царь"),
    ] {
        assert_backend_identity(lang, text);
    }
}

fn assert_backend_identity(lang: &str, text: &str) {
    let result = phonemize(Language::from_code(lang).unwrap(), text).unwrap();
    assert!(!result.phonemes.is_empty());
    assert_same_bytes(result, phonemize_lang(lang, text).unwrap());
}

#[test]
fn hindi_default_remains_the_trained_current_output() {
    assert_backend_identity("hin", "यह शहर");
    let result = phonemize_lang("hin", "यह शहर").unwrap();
    assert_eq!(
        result
            .phonemes
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>(),
        ["j", "eː", "ʃ", "ɛː", "ɦ", "ɛː", "ɾ"]
    );
    let words = g2p::hindi::phonemize("यह शहर").unwrap();
    assert_eq!(
        result
            .phonemes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        words
            .into_iter()
            .flat_map(|w| w.phonemes)
            .collect::<Vec<_>>()
    );
    assert_eq!(g2p::hindi::word("यह").unwrap().phonemes, ["j", "eː"]);
    assert!(matches!(
        phonemize_lang("hin", "19 यह AOL"),
        Err(Error::Unlabelable(_))
    ));
    assert!(matches!(
        g2p::hindi::phonemize("19 यह AOL"),
        Err(Error::Unlabelable(_))
    ));
    assert_eq!(
        g2p::hindi::word("ज्ञान").unwrap().phonemes,
        ["ɡ", "j", "aː", "n"]
    );
}

#[test]
fn mandarin_default_preserves_full_output() {
    assert_backend_identity("zho-hans", "你好");
}

#[cfg(feature = "japanese")]
#[test]
fn japanese_default_preserves_full_output() {
    assert_backend_identity("jpn", "こんにちは");
}

#[cfg(not(feature = "japanese"))]
#[test]
fn japanese_without_feature_stays_unsupported() {
    assert!(matches!(
        phonemize(Language::Japanese, "こんにちは"),
        Err(Error::UnsupportedLanguage(lang)) if lang == "jpn"
    ));
    let _: g2p_types::japanese::Labels = g2p::japanese::Labels::default();
}

#[test]
#[ignore = "needs uv on PATH and network on first run"]
fn thai_default_preserves_full_output() {
    assert_backend_identity("tha", "สวัสดี");
}

#[test]
#[ignore = "needs uv on PATH and network on first run"]
fn korean_default_preserves_full_output() {
    assert_backend_identity("kor", "안녕하세요");
}

#[test]
fn portuguese_default_is_brazilian_and_european_is_supported() {
    let default = phonemize_lang("por", "dia noite").unwrap();
    let brazilian = phonemize(Language::PortugueseBrazil, "dia noite").unwrap();
    let european = phonemize(Language::PortugueseEuro, "dia noite").unwrap();
    assert_same_bytes(default.clone(), brazilian);
    assert_ne!(default.phonemes, european.phonemes);
    assert!(!european.phonemes.is_empty());
}
