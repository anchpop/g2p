use g2p_types::{LabelSource, Parsed, Phonemized, Pitch, Stress, Syllable};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;

fn roundtrip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: T) {
    let json = serde_json::to_string(&value).unwrap();
    assert_eq!(serde_json::from_str::<T>(&json).unwrap(), value);
}

#[test]
fn label_sources_are_copy_and_serialize_as_unit_variants() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<LabelSource>();
    for source in [
        LabelSource::Espeak,
        LabelSource::Hindi,
        LabelSource::Mandarin,
        LabelSource::Japanese,
        LabelSource::Thai,
        LabelSource::Korean,
    ] {
        roundtrip(source);
    }
    assert_eq!(serde_json::to_value(LabelSource::Espeak).unwrap(), "Espeak");
    assert_eq!(
        serde_json::from_str::<LabelSource>("\"Espeak\"").unwrap(),
        LabelSource::Espeak
    );
}

#[test]
fn existing_serialized_shapes_are_preserved() {
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

#[test]
fn varieties_use_snake_case() {
    use g2p_types::Variety;
    assert_eq!(Variety::default(), Variety::Default);
    for (variety, name) in [
        (Variety::Default, "default"),
        (Variety::LatinAmerican, "latin_american"),
        (Variety::European, "european"),
        (Variety::Brazilian, "brazilian"),
    ] {
        assert_eq!(serde_json::to_value(variety).unwrap(), name);
        roundtrip(variety);
    }
    assert!(serde_json::from_str::<Variety>("\"LatinAmerican\"").is_err());
}
