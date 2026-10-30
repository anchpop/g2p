use g2p::{Phonemized, phonemize_lang};

fn slices<'a>(text: &'a str, p: &Phonemized) -> Vec<&'a str> {
    assert_eq!(p.word_text_spans.len(), p.word_spans.len());
    let mut previous = 0;
    p.word_text_spans
        .iter()
        .map(|&(start, end)| {
            assert!(previous <= start && start < end);
            previous = end;
            &text[start..end]
        })
        .collect()
}

#[test]
fn engine_provenance_survives_joins_splits_and_clause_boundaries() {
    for (lang, text, expected) in [
        ("fra", "Bonjour ! Ça va ?", vec!["Bonjour", "Ça", "va"]),
        ("fra", "L’homme n’a rien.", vec!["L’homme", "n’a", "rien"]),
        ("fra", "c'est un ami", vec!["c'est", "un", "ami"]),
        ("eng", "23", vec!["23"]),
        ("fra", "23", vec!["23"]),
        ("eng", "in the house", vec!["in the", "house"]),
        ("rus", "Том - плакса.", vec!["Том", "плакса"]),
    ] {
        let p = phonemize_lang(lang, text).unwrap();
        assert_eq!(slices(text, &p), expected, "{lang}: {text}");
    }
    let text = "hello ".repeat(350);
    let p = phonemize_lang("eng", &text).unwrap();
    assert_eq!(slices(&text, &p).len(), 350);
    for (lang, text) in [
        ("eng", "We live in Russia."),
        ("eng", "They are no longer my neighbors."),
        ("eng", "Each of the three boys won a prize."),
        (
            "eng",
            "The blackbird found a little red worm in the green grass.",
        ),
        ("deu", "Er verliert keine Zeit."),
        ("ita", "Mio padre non è un insegnante, ma un dottore."),
        ("ita", "Perché siete tutti tristi?"),
        ("ita", "Prendi mia moglie - per favore!"),
        ("ita", "Stasera"),
        ("rus", "Дельфин выпрыгнул из воды."),
        ("rus", "Я хочу перед вами извиниться."),
        ("rus", "Это было в прошлом году в начале октября."),
        ("spa", "Vd. es el gran productor que todo lo dirige."),
    ] {
        let p = phonemize_lang(lang, text).unwrap();
        assert!(!slices(text, &p).is_empty(), "{lang}: {text}");
    }
}

#[test]
fn hindi_joiners_and_mandarin_punctuation_keep_source_words() {
    for (lang, text, expected) in [
        ("hin", "जल्‍दी चलो", vec!["जल्‍दी", "चलो"]),
        ("hin", "जल्‌दी चलो", vec!["जल्‌दी", "चलो"]),
        ("zho-hans", "你好•中国！", vec!["你", "好", "中", "国"]),
    ] {
        let p = phonemize_lang(lang, text).unwrap();
        assert_eq!(slices(text, &p), expected);
    }
}

#[cfg(feature = "japanese")]
#[test]
fn japanese_closure_merges_its_two_owners_and_maps_kana_normalization() {
    for text in ["主人はすぐ戻ってきます。", "ｶﾞｯｺｳに行く。", "コーヒーです。"]
    {
        let p = phonemize_lang("jpn", text).unwrap();
        assert!(!slices(text, &p).is_empty(), "{text}");
        assert_eq!(p.pitch.len(), p.phonemes.len());
    }
    let text = "瑠璃鶇が空を背負っている。";
    let p = phonemize_lang("jpn", text).unwrap();
    assert!(p.word_text_spans.is_empty()); // NJD drops the unknown 鶇.
}

#[test]
#[ignore = "needs uv on PATH and network on first run"]
fn python_backends_keep_eojeol_and_repeated_thai_owners() {
    let text = "못 만났어, 라디오!";
    let p = phonemize_lang("kor", text).unwrap();
    assert_eq!(slices(text, &p), ["못", "만났어", "라디오"]);
    for text in [
        "เร็วๆ",
        "เด็กๆโตเร็วมาก",
        "อยู่ห่างๆ เขาไว้",
        "ลาร์สๆ ดูนี่สิ",
        "เเม็กซิม!",
    ] {
        let p = phonemize_lang("tha", text).unwrap();
        assert!(!slices(text, &p).is_empty(), "{text}");
    }
    let text = "เร็วๆ";
    let p = phonemize_lang("tha", text).unwrap();
    assert_eq!(slices(text, &p), [text]);
}
