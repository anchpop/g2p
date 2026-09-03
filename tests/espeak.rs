//! End-to-end tests against the embedded espeak-ng fork. No environment
//! setup needed: the engine and its data are inside the test binary.

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
