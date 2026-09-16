//! Unified language dispatch preserves the existing voice and backend paths.
use g2p::{Error, PhonemizeRequest, Variety, phonemize_lang, phonemize_language};

fn assert_same_bytes(left: g2p::Phonemized, right: g2p::Phonemized) {
    assert_eq!(
        serde_json::to_vec(&left).unwrap(),
        serde_json::to_vec(&right).unwrap()
    );
}

#[test]
fn request_defaults_and_variety_setter() {
    let request = PhonemizeRequest::new("spa", "cinco");
    assert_eq!(request.variety, Variety::Default);
    assert_eq!(
        request.variety(Variety::LatinAmerican).variety,
        Variety::LatinAmerican
    );
}

#[test]
fn spanish_varieties_distinguish_seseo() {
    for (variety, initial) in [
        (Variety::Default, "θ"),
        (Variety::European, "θ"),
        (Variety::LatinAmerican, "s"),
    ] {
        let result =
            phonemize_language(PhonemizeRequest::new("spa", "cinco").variety(variety)).unwrap();
        assert_eq!(result.phonemes.first().unwrap(), initial);
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
    let result = phonemize_language(PhonemizeRequest::new(lang, text)).unwrap();
    assert!(!result.phonemes.is_empty());
    assert_same_bytes(result, phonemize_lang(lang, text).unwrap());
}

#[test]
fn hindi_default_remains_the_trained_current_output() {
    assert_backend_identity("hin", "यह शहर");
    let result = phonemize_lang("hin", "यह शहर").unwrap();
    assert_eq!(result.phonemes, ["j", "eː", "ʃ", "ɛː", "ɦ", "ɛː", "ɾ"]);
    let words = g2p::hindi::phonemize("यह शहर").unwrap();
    assert_eq!(
        result.phonemes,
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
        phonemize_language(PhonemizeRequest::new("jpn", "こんにちは")),
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
fn unsupported_varieties_are_rejected_before_running_backends() {
    for lang in ["eng", "fra", "hin", "zho-hans", "jpn", "tha", "kor"] {
        for variety in [
            Variety::LatinAmerican,
            Variety::European,
            Variety::Brazilian,
        ] {
            assert!(matches!(
                phonemize_language(PhonemizeRequest::new(lang, "").variety(variety)),
                Err(Error::VarietyNotApplicable { lang: actual_lang, variety: actual_variety })
                    if actual_lang == lang && actual_variety == variety
            ));
        }
    }
    for (lang, variety) in [("spa", Variety::Brazilian), ("por", Variety::LatinAmerican)] {
        assert!(matches!(
            phonemize_language(PhonemizeRequest::new(lang, "").variety(variety)),
            Err(Error::VarietyNotApplicable { .. })
        ));
    }
}

#[test]
fn portuguese_default_is_brazilian_and_european_is_supported() {
    let request = PhonemizeRequest::new("por", "dia noite");
    let default = phonemize_language(request).unwrap();
    let brazilian = phonemize_language(request.variety(Variety::Brazilian)).unwrap();
    let european = phonemize_language(request.variety(Variety::European)).unwrap();
    assert_same_bytes(default.clone(), brazilian);
    assert_ne!(default.phonemes, european.phonemes);
    assert!(!european.phonemes.is_empty());
}

#[test]
fn unknown_language_stays_unsupported_with_every_variety() {
    for variety in [
        Variety::Default,
        Variety::LatinAmerican,
        Variety::European,
        Variety::Brazilian,
    ] {
        let error = phonemize_language(PhonemizeRequest::new("xx-nope", "hello").variety(variety))
            .unwrap_err();
        assert!(matches!(&error, Error::UnsupportedLanguage(lang) if lang == "xx-nope"));
        assert_eq!(
            error.to_string(),
            phonemize_lang("xx-nope", "hello").unwrap_err().to_string()
        );
    }
}
