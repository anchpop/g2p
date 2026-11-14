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

/// The text each run of phonemes was spoken from, after checking that every
/// phoneme's span lies within its word's.
fn phoneme_sources<'a>(lang: &str, text: &'a str, readings: &[(&str, &str)]) -> Vec<&'a str> {
    let readings: Vec<_> = readings
        .iter()
        .map(|&(of, reading)| {
            let start = text.find(of).unwrap();
            g2p::Reading {
                start,
                end: start + of.len(),
                reading: reading.into(),
            }
        })
        .collect();
    let language = g2p::Language::from_code(lang).unwrap();
    let hints = g2p::Hints {
        readings: &readings,
        ..Default::default()
    };
    let p = g2p::phonemize_with(language, text, &hints).unwrap();
    assert_eq!(p.phoneme_text_spans.len(), p.phonemes.len(), "{text}");
    for (&(start, end), &(word_start, word_end)) in p.word_spans.iter().zip(&p.word_text_spans) {
        for &(s, e) in &p.phoneme_text_spans[start..end] {
            assert!(word_start <= s && e <= word_end, "{text}");
        }
    }
    let mut sources: Vec<_> = p
        .phoneme_text_spans
        .iter()
        .map(|&(s, e)| &text[s..e])
        .collect();
    sources.dedup();
    sources
}

#[test]
fn espeak_phonemes_keep_the_letters_they_were_spoken_from() {
    for (lang, text, expected) in [
        // An elided clitic is a prefix of its word's translation; the stem
        // is a dictionary entry (est) or the letter groups its rules
        // matched (a m i).
        ("fra", "c'est l'ami", vec!["c", "est", "l", "a", "m", "i"]),
        // Italian's geminate across the apostrophe stays with the clitic
        // whose rule spells both halves: dell + anno is d e lː | a nː o.
        ("ita", "dell'anno", vec!["dell", "a", "nn", "o"]),
        // A whole-word dictionary entry marks the start of each part after
        // an apostrophe with ||+ ("com'è kom||+'E"); a word the rules read
        // whole keeps its letter groups even when a suffix rule (-a) said
        // to keep the stem's first translation.
        ("ita", "com'è", vec!["com", "è"]),
        ("ita", "cos'ha", vec!["c", "o", "s", "a"]),
        // A removed suffix (English 's) and compound parts.
        ("eng", "Tom's", vec!["T", "o", "m", "s"]),
        ("deu", "Hausaufgabe", vec!["Haus", "auf", "gabe"]),
        // Hyphenated words split at the hyphen.
        ("rus", "по-фински", vec!["по", "ф", "и", "н", "с", "к", "и"]),
        // Removing an accent the rules lack moves the letters after it back
        // a byte or more; their sources are still the letters as written.
        ("eng", "Māori", vec!["M", "āo", "r", "i"]),
        ("eng", "Kōbe", vec!["K", "ō", "b"]),
        ("fra", "l'aîné", vec!["l", "aî", "n", "é"]),
    ] {
        assert_eq!(phoneme_sources(lang, text, &[]), expected, "{lang}: {text}");
    }
}

#[test]
fn dictionary_phrases_split_at_their_word_breaks() {
    // Phrase entries spell a ||+ before each word they consume ("(in the)
    // %In||+D@2"): the phrase stays one phonological word, so its weak forms
    // and vowel reductions are unchanged, but each word keeps its phonemes.
    for (lang, text, expected) in [
        ("eng", "in the house", vec!["in", "the", "house"]),
        ("eng", "I had to go", vec!["I", "had", "to", "go"]),
        ("fra", "et en plus", vec!["et", "en", "plus"]),
        ("fra", "parce qu'il", vec!["parce", "qu'il"]),
        ("rus", "не было", vec!["не", "было"]),
    ] {
        assert_eq!(phoneme_sources(lang, text, &[]), expected, "{lang}: {text}");
    }
    // A || would make было a word of its own, with its own stressed vowel.
    let p = phonemize_lang("rus", "не было").unwrap();
    assert_eq!(p.raw.trim(), "nʲˈebɨɫʌ");
}

#[test]
fn word_level_backends_keep_each_character_or_word() {
    for (lang, text, expected) in [
        ("zho-hans", "你好", vec!["你", "好"]),
        // The danda is punctuation, not part of the word before it.
        ("hin", "वह है।", vec!["वह", "है"]),
    ] {
        assert_eq!(phoneme_sources(lang, text, &[]), expected, "{lang}: {text}");
    }
}

#[cfg(feature = "japanese")]
#[test]
fn japanese_phonemes_keep_their_mora() {
    for (text, readings, expected) in [
        // Kana say one mora each; the long consonant of 行って is spoken
        // from っ and て, and a lone kanji takes the moras left over.
        ("はねでも", vec![], vec!["は", "ね", "で", "も"]),
        (
            "行ってきます",
            vec![],
            vec!["行", "って", "て", "き", "ま", "す"],
        ),
        // Kanji take the moras of their supplied readings.
        (
            "現金でも",
            vec![("現", "げん"), ("金", "きん")],
            vec!["現", "金", "で", "も"],
        ),
    ] {
        assert_eq!(phoneme_sources("jpn", text, &readings), expected, "{text}");
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

/// Byte ranges of `words`, found in order in `text`.
fn ranges(text: &str, words: &[&str]) -> Vec<(usize, usize)> {
    let mut cursor = 0;
    words
        .iter()
        .map(|w| {
            let start = cursor + text[cursor..].find(w).unwrap();
            cursor = start + w.len();
            (start, cursor)
        })
        .collect()
}

#[test]
#[ignore = "needs uv on PATH and network on first run"]
fn thai_uses_the_callers_words() {
    let thai = g2p::Language::from_code("tha").unwrap();
    // vachana's segmenter reads มา + นะ as มานะ "perseverance"; the given
    // words keep their own phonemes and spans, punctuation is unspoken, and
    // ๆ repeats its word.
    let text = "เดี๋ยวครูมานะ ดีๆ!";
    let words = ranges(text, &["เดี๋ยว", "ครู", "มา", "นะ", "ดีๆ", "!"]);
    let hints = g2p::Hints {
        words: &words,
        ..Default::default()
    };
    let p = g2p::phonemize_with(thai, text, &hints).unwrap();
    assert_eq!(slices(text, &p), ["เดี๋ยว", "ครู", "มา", "นะ", "ดีๆ"]);
    let (start, end) = p.word_spans[2];
    assert_eq!(
        p.phonemes[start..end]
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>(),
        ["m", "aː"]
    );
    let (start, end) = p.word_spans[4];
    assert_eq!(end - start, 4); // d iː d iː
    // An unknown word TLTK transcribes only partly (ดี|พโรส as dii0) is
    // read as vachana's segmenter splits it, so no syllable goes missing.
    let text = "ดีพโรส";
    let words = [(0, text.len())];
    let hints = g2p::Hints {
        words: &words,
        ..Default::default()
    };
    let p = g2p::phonemize_with(thai, text, &hints).unwrap();
    assert_eq!(p.tone.iter().flatten().count(), 4, "{:?}", p.raw);
    // Without words, vachana's own segmentation is used.
    let text = "เดี๋ยวครูมานะ ดีๆ!";
    let p = phonemize_lang("tha", text).unwrap();
    assert!(slices(text, &p).contains(&"มานะ"));
}

#[test]
fn caller_words_must_tile_the_text_and_are_thai_only() {
    let refused = |lang: &str, text: &str, words: &[(usize, usize)]| {
        let hints = g2p::Hints {
            words,
            ..Default::default()
        };
        let language = g2p::Language::from_code(lang).unwrap();
        match g2p::phonemize_with(language, text, &hints) {
            Err(g2p::Error::Unlabelable(reason)) => reason,
            other => panic!("{other:?}"),
        }
    };
    assert_eq!(
        refused("fra", "on est", &[(0, 2), (3, 6)]),
        "words_unsupported:fra"
    );
    let text = "มานะ";
    // overlapping, out of order, off a character boundary, leaving a letter out
    for words in [
        vec![(0, 6), (3, 12)],
        vec![(6, 12), (0, 6)],
        vec![(0, 4)],
        vec![(0, 6)],
    ] {
        assert!(refused("tha", text, &words).starts_with("words_invalid:"));
    }
}
