//! Unified language dispatch preserves the existing voice and backend paths.
use g2p::{Error, HindiCanon, phonemize, phonemize_lang, phonemize_lang_with, phonemize_language};

#[test]
fn spanish_voice_override_preserves_full_output_and_seseo() {
    let text = "zapato";
    let overridden = phonemize_language("spa", Some("es-419"), text, HindiCanon::Current).unwrap();
    assert_eq!(overridden, phonemize(text, "es-419").unwrap());
    assert!(overridden.phonemes.iter().any(|p| p == "s"));
    assert!(!overridden.phonemes.iter().any(|p| p == "θ"));

    let default = phonemize_language("spa", None, text, HindiCanon::Current).unwrap();
    assert_eq!(default, phonemize_lang("spa", text).unwrap());
    assert_eq!(default, phonemize(text, "es").unwrap());
    assert!(default.phonemes.iter().any(|p| p == "θ"));
}

fn assert_backend_identity(lang: &str, text: &str) {
    let result = phonemize_language(lang, None, text, HindiCanon::Current).unwrap();
    assert!(!result.phonemes.is_empty());
    assert_eq!(result, phonemize_lang(lang, text).unwrap());
    assert_eq!(
        result,
        phonemize_lang_with(lang, text, HindiCanon::Current).unwrap()
    );
}

#[test]
fn hindi_default_preserves_full_output() {
    assert_backend_identity("hin", "यह शहर");
    for canon in [HindiCanon::Current, HindiCanon::Legacy] {
        let result = phonemize_language("hin", None, "यह शहर", canon).unwrap();
        assert_eq!(result, phonemize_lang_with("hin", "यह शहर", canon).unwrap());
        let words = g2p::hindi::phonemize("यह शहर", canon).unwrap();
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
        phonemize_language("jpn", None, "こんにちは", HindiCanon::Current),
        Err(Error::UnsupportedLanguage(lang)) if lang == "jpn"
    ));
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
fn every_backend_rejects_voice_overrides_before_running() {
    for lang in ["hin", "zho-hans", "jpn", "tha", "kor"] {
        assert!(matches!(
            phonemize_language(lang, Some("es-419"), "", HindiCanon::Current),
            Err(Error::VoiceNotApplicable { lang: actual_lang, voice })
                if actual_lang == lang && voice == "es-419"
        ));
    }
}

#[test]
fn unknown_language_stays_unsupported_even_with_voice() {
    for voice in [None, Some("es-419")] {
        let error = phonemize_language("xx-nope", voice, "hello", HindiCanon::Current).unwrap_err();
        assert!(matches!(&error, Error::UnsupportedLanguage(lang) if lang == "xx-nope"));
        assert_eq!(
            error.to_string(),
            phonemize_lang("xx-nope", "hello").unwrap_err().to_string()
        );
    }
}
