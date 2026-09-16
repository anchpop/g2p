//! Unified language dispatch preserves the existing voice and backend paths.
use g2p::{
    Error, HindiLabels, PhonemizeRequest, Variety, VoiceChoice, phonemize, phonemize_lang,
    phonemize_lang_with, phonemize_language,
};

fn assert_same_bytes(left: g2p::Phonemized, right: g2p::Phonemized) {
    assert_eq!(
        serde_json::to_vec(&left).unwrap(),
        serde_json::to_vec(&right).unwrap()
    );
}

#[test]
fn request_defaults_and_setters_have_one_voice_choice() {
    let request = PhonemizeRequest::new("spa", "cinco");
    assert_eq!(request.hindi_labels, HindiLabels::Current);
    assert_eq!(request.voice, VoiceChoice::Variety(Variety::Default));
    assert_eq!(
        request.variety(Variety::LatinAmerican).voice("es").voice,
        VoiceChoice::Raw("es")
    );
    assert_eq!(
        request.voice("es").variety(Variety::LatinAmerican).voice,
        VoiceChoice::Variety(Variety::LatinAmerican)
    );
}

#[test]
fn spanish_varieties_preserve_raw_voice_output_and_distinguish_seseo() {
    for (variety, voice, initial) in [
        (Variety::Default, "es", "θ"),
        (Variety::European, "es", "θ"),
        (Variety::LatinAmerican, "es-419", "s"),
    ] {
        let result =
            phonemize_language(PhonemizeRequest::new("spa", "cinco").variety(variety)).unwrap();
        assert_eq!(result.phonemes.first().unwrap(), initial);
        assert_same_bytes(result, phonemize("cinco", voice).unwrap());
    }
}

#[test]
fn espeak_wrappers_preserve_full_default_output() {
    for (lang, text, voice) in [
        ("eng", "church cats more mirror", "en-us"),
        ("deu", "Häuser Zeit", "de"),
        ("fra", "on est", "fr-fr"),
        ("ita", "pizza", "it"),
        ("por", "mãe pão", "pt-br"),
        ("spa", "cinco", "es"),
        ("rus", "царь", "ru"),
    ] {
        assert_backend_identity(lang, text);
        assert_same_bytes(
            phonemize_lang(lang, text).unwrap(),
            phonemize(text, voice).unwrap(),
        );
    }
}

fn assert_backend_identity(lang: &str, text: &str) {
    let result = phonemize_language(PhonemizeRequest::new(lang, text)).unwrap();
    assert!(!result.phonemes.is_empty());
    assert_same_bytes(result.clone(), phonemize_lang(lang, text).unwrap());
    // The Hindi label version is irrelevant for all other backends.
    for labels in [HindiLabels::Current, HindiLabels::Legacy] {
        if lang != "hin" || labels == HindiLabels::Current {
            assert_same_bytes(
                result.clone(),
                phonemize_lang_with(lang, text, labels).unwrap(),
            );
        }
    }
}

#[test]
fn hindi_default_preserves_full_output() {
    assert_backend_identity("hin", "यह शहर");
    for labels in [HindiLabels::Current, HindiLabels::Legacy] {
        let result = phonemize_language(PhonemizeRequest {
            hindi_labels: labels,
            ..PhonemizeRequest::new("hin", "यह शहर")
        })
        .unwrap();
        assert_same_bytes(
            result.clone(),
            phonemize_lang_with("hin", "यह शहर", labels).unwrap(),
        );
        let words = g2p::hindi::phonemize("यह शहर", labels).unwrap();
        assert_eq!(
            result.phonemes,
            words
                .into_iter()
                .flat_map(|w| w.phonemes)
                .collect::<Vec<_>>()
        );
    }
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
fn non_spanish_varieties_are_rejected_before_running_backends() {
    for lang in ["eng", "fra", "por", "hin", "zho-hans", "jpn", "tha", "kor"] {
        for variety in [Variety::LatinAmerican, Variety::European] {
            assert!(matches!(
                phonemize_language(PhonemizeRequest::new(lang, "").variety(variety)),
                Err(Error::VarietyNotApplicable { lang: actual_lang, variety: actual_variety })
                    if actual_lang == lang && actual_variety == variety
            ));
        }
    }
}

#[test]
fn explicit_voice_replaces_even_inapplicable_variety() {
    for lang in ["spa", "fra", "por"] {
        for variety in [Variety::Default, Variety::LatinAmerican, Variety::European] {
            let result = phonemize_language(
                PhonemizeRequest::new(lang, "cinco")
                    .variety(variety)
                    .voice("es-419"),
            )
            .unwrap();
            assert_same_bytes(result, phonemize("cinco", "es-419").unwrap());
        }
    }
    assert!(matches!(
        phonemize_language(PhonemizeRequest::new("fra", "bonjour").variety(Variety::European).voice("xx-no-such-voice")),
        Err(Error::UnknownVoice(voice)) if voice == "xx-no-such-voice"
    ));
}

#[test]
fn every_backend_rejects_voice_overrides_before_running() {
    for lang in ["hin", "zho-hans", "jpn", "tha", "kor"] {
        assert!(matches!(
            phonemize_language(PhonemizeRequest::new(lang, "").variety(Variety::European).voice("es-419")),
            Err(Error::VoiceNotApplicable { lang: actual_lang, voice })
                if actual_lang == lang && voice == "es-419"
        ));
    }
}

#[test]
fn unknown_language_stays_unsupported_even_with_voice_or_variety() {
    for voice in [
        VoiceChoice::default(),
        VoiceChoice::Variety(Variety::LatinAmerican),
        VoiceChoice::Variety(Variety::European),
        VoiceChoice::Raw("es-419"),
    ] {
        let error = phonemize_language(PhonemizeRequest {
            voice,
            ..PhonemizeRequest::new("xx-nope", "hello")
        })
        .unwrap_err();
        assert!(matches!(&error, Error::UnsupportedLanguage(lang) if lang == "xx-nope"));
        assert_eq!(
            error.to_string(),
            phonemize_lang("xx-nope", "hello").unwrap_err().to_string()
        );
    }
}
