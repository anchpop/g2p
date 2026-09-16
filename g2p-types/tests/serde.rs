use g2p_types::{HindiLabels, LabelSource, Parsed, Phonemized, Pitch, Stress, Syllable};
use serde::{Serialize, de::DeserializeOwned};
use std::borrow::Cow;
use std::fmt::Debug;

fn roundtrip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: T) {
    let json = serde_json::to_string(&value).unwrap();
    assert_eq!(serde_json::from_str::<T>(&json).unwrap(), value);
}

#[test]
fn label_sources_deserialize_from_owned_input() {
    for source in [
        LabelSource::Espeak("fr-fr".into()),
        LabelSource::Espeak(format!("{}-{}", "es", 419).into()),
        LabelSource::Hindi,
        LabelSource::Mandarin,
        LabelSource::Japanese,
        LabelSource::Thai,
        LabelSource::Korean,
    ] {
        roundtrip(source);
    }
    let source = {
        let input = String::from(r#"{"Espeak":"custom-voice"}"#);
        serde_json::from_str::<LabelSource>(&input).unwrap()
    };
    assert!(matches!(source, LabelSource::Espeak(Cow::Owned(voice)) if voice == "custom-voice"));
}

#[test]
fn existing_serialized_shapes_are_preserved() {
    assert_eq!(
        serde_json::to_string(&HindiLabels::Legacy).unwrap(),
        "\"legacy\""
    );
    assert_eq!(
        serde_json::to_string(&HindiLabels::Current).unwrap(),
        "\"current\""
    );
    for (stress, name, code) in [
        (Stress::None, "None", 0),
        (Stress::Primary, "Primary", 1),
        (Stress::Secondary, "Secondary", 2),
    ] {
        assert_eq!(stress.code(), code);
        assert_eq!(serde_json::to_value(stress).unwrap(), name);
        roundtrip(stress);
    }
    let empty = Phonemized::default();
    assert_eq!(
        serde_json::to_value(&empty).unwrap(),
        serde_json::json!({
            "raw": "", "phonemes": [], "stress": [], "word_spans": []
        })
    );
    roundtrip(empty);
}

#[test]
fn all_shared_labels_roundtrip_without_engines() {
    let syllable = Syllable {
        start: 0,
        end: 1,
        nucleus: 0,
        moras: 2,
        stressed: true,
    };
    let pitch = Pitch {
        phrase: 1,
        mora: 1,
        phrase_moras: 2,
        nucleus: 0,
        level: 0,
    };
    roundtrip(Phonemized {
        raw: "a".into(),
        phonemes: vec!["a".into()],
        stress: vec![Stress::Primary],
        word_spans: vec![(0, 1)],
        syllables: vec![syllable.clone()],
        tone: vec![Some(1)],
        pitch: vec![Some(pitch.clone())],
        accent_withheld: Some("diagnostic".into()),
    });
    roundtrip(Parsed {
        phonemes: vec!["a".into()],
        stress: vec![Stress::Primary],
        word_spans: vec![(0, 1)],
    });
    roundtrip(g2p_types::hindi::Word {
        phonemes: vec!["a".into()],
        stress: vec![Stress::Primary],
        syllables: vec![syllable],
        schwa_retained: vec![true, false],
    });
    roundtrip(g2p_types::mandarin::Syllable {
        char: '你',
        pinyin: "ni3".into(),
        phonemes: vec!["n".into(), "i".into()],
        tone: vec![None, Some(3)],
    });
    roundtrip(g2p_types::japanese::Labels {
        phonemes: vec!["a".into()],
        pitch: vec![Some(pitch)],
        accent_withheld: None,
        native_phones: vec!["a".into()],
    });
    roundtrip(g2p_types::thai::Labels {
        raw: "a".into(),
        phonemes: vec!["a".into()],
        stress: vec![Stress::Primary],
        tone: vec![Some(1)],
        word_spans: vec![(0, 1)],
    });
    roundtrip(g2p_types::korean::Labels {
        raw: "나".into(),
        phonemes: vec!["n".into(), "a".into()],
        stress: vec![Stress::None; 2],
        word_spans: vec![(0, 2)],
    });
}
