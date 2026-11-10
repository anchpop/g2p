//! Supplied readings: make the NJD nodes say what the caller already knows.
//!
//! The front end runs as usual first. Wherever its words already spell the
//! supplied readings, nothing changes, so a caller that passes readings the
//! dictionary agrees with gets exactly the labels it would have got without
//! them. Each word (or run of words a reading straddles) that disagrees is
//! replaced before the accent and sound-change rules run:
//!
//! * by the dictionary's own entry for that surface with the supplied reading
//!   (方 ほう → かた), which keeps its accent and pronunciation form;
//! * failing that, by a word that just says the supplied kana (a name, a
//!   colloquial form, or text the tokenizer mis-split). Nothing then knows
//!   that word's accent, so the utterance's accent is withheld.

use super::{Engine, normalized_spans};
use crate::{Error, Reading};
use jpreprocess::NJD;
use jpreprocess_core::pronunciation::Pronunciation;
use jpreprocess_dictionary::{
    dictionary::word_encoding::JPreprocessDictionaryWordEncoding, word_data::get_word_data,
};

/// A stretch of words whose reading is not the supplied one.
struct Mismatch {
    start: usize,
    end: usize,
    /// What the stretch must be read as, in katakana.
    read: String,
    /// The same with long vowels written ー, as dictionary pronunciations
    /// are. `None` when the readings do not say which vowels are long.
    pron: Option<String>,
}

/// Katakana and ー only: the form in which text, readings and NJD fields compare.
fn kana(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c {
            'ぁ'..='ゖ' => char::from_u32(c as u32 + 0x60),
            'ァ'..='ヺ' | 'ー' => Some(c),
            _ => None,
        })
        .collect()
}

/// A reading as the dictionary would write its pronunciation: おう → オー,
/// えい → エー, and a doubled vowel → ー. Applied only inside one supplied
/// reading, because the same kana across a morpheme boundary (子牛 こ・うし)
/// are two vowels.
fn long_vowels(katakana: &str) -> String {
    const ROWS: [&str; 5] = [
        "アカガサザタダナハバパマヤラワァャヮ",
        "イキギシジチヂニヒビピミリィ",
        "ウクグスズツヅヌフブプムユルゥュヴ",
        "エケゲセゼテデネヘベペメレェ",
        "オコゴソゾトドノホボポモヨロヲォョ",
    ];
    let mut previous = None;
    katakana
        .chars()
        .map(|c| {
            if matches!(
                (previous, c),
                (Some(0), 'ア')
                    | (Some(1), 'イ')
                    | (Some(2), 'ウ')
                    | (Some(3), 'エ' | 'イ')
                    | (Some(4), 'オ' | 'ウ')
            ) {
                return 'ー';
            }
            if c != 'ー' {
                previous = ROWS.iter().position(|row| row.contains(c));
            }
            c
        })
        .collect()
}

/// Can `target` be spelled by taking each word's reading or its
/// pronunciation? The pronunciation is what is spoken, and is where sound
/// changes the reading field lacks (一回 イチ → イッ) show up.
fn spells(words: &[(String, String)], target: &str) -> bool {
    match words.split_first() {
        None => target.is_empty(),
        // Usually the two are the same kana; trying it twice would double
        // the work at every word of a long stretch.
        Some(((read, pron), rest)) => [Some(read), (pron != read).then_some(pron)]
            .into_iter()
            .flatten()
            .any(|option| {
                !option.is_empty()
                    && target
                        .strip_prefix(option.as_str())
                        .is_some_and(|tail| spells(rest, tail))
            }),
    }
}

/// An NJD node and the bytes of `text` it stands for.
struct Word {
    node: usize,
    start: usize,
    end: usize,
}

/// Every node that has a surface, located in `text`.
fn words(text: &str, njd: &NJD) -> Result<Vec<Word>, Error> {
    let (nodes, surfaces): (Vec<_>, Vec<_>) = njd
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !node.get_string().is_empty())
        .map(|(index, node)| (index, node.get_string().to_owned()))
        .unzip();
    let spans = normalized_spans(text, &surfaces);
    if spans.len() != nodes.len() {
        return Err(Error::Unlabelable("japanese_reading_alignment".into()));
    }
    Ok(nodes
        .into_iter()
        .zip(spans)
        .map(|(node, (start, end))| Word { node, start, end })
        .collect())
}

/// Every stretch of preprocessed words that does not spell its readings.
/// `readings` is ordered and non-overlapping; `forced` are the stretches
/// already replaced by a word that says the supplied kana.
fn mismatches(
    text: &str,
    njd: &NJD,
    readings: &[&Reading],
    forced: &[(usize, usize)],
) -> Result<Vec<Mismatch>, Error> {
    let words = words(text, njd)?;
    let mut out = Vec::new();
    let mut consumed = 0;
    let mut first = 0;
    while first < words.len() {
        // A reading is indivisible, so the words it straddles are judged together.
        let start = words[first].start;
        let mut end = words[first].end;
        let mut after = first + 1;
        let touching = |end: usize| {
            readings
                .iter()
                .filter(move |r| r.start < end && r.end > start)
        };
        while let Some(reach) = touching(end).map(|r| r.end).max()
            && reach > end
        {
            let Some(next) = words.get(after) else {
                return Err(Error::Unlabelable("japanese_reading_span".into()));
            };
            end = next.end;
            after += 1;
        }
        if touching(end).next().is_some() {
            let mut read = String::new();
            let mut pron = Some(String::new());
            let mut at = start;
            while at < end {
                if let Some(reading) = readings.iter().find(|r| r.start == at) {
                    let supplied = kana(&reading.reading);
                    let long = long_vowels(&supplied);
                    // A reading that takes in okurigana does not say where
                    // it starts, and おう is two vowels in 思う but one in 行こう.
                    if long != supplied && !kana(&text[reading.start..reading.end]).is_empty() {
                        pron = None;
                    }
                    if let Some(pron) = &mut pron {
                        pron.push_str(&long);
                    }
                    read.push_str(&supplied);
                    at = reading.end;
                    consumed += 1;
                    continue;
                }
                let c = text[at..].chars().next().unwrap();
                let spoken = kana(c.encode_utf8(&mut [0; 4]));
                // Anything else that is pronounced (a kanji, a digit, a
                // voicing mark) has no reading here to compare or to say.
                if spoken.is_empty()
                    && (c.is_alphanumeric() || matches!(c, '\u{3099}'..='\u{309c}' | 'ﾞ' | 'ﾟ'))
                {
                    return Err(Error::Unlabelable(format!(
                        "japanese_reading_partial:{}",
                        &text[start..end]
                    )));
                }
                read.push_str(&spoken);
                if let Some(pron) = &mut pron {
                    pron.push_str(&spoken);
                }
                at += c.len_utf8();
            }
            let said: Vec<_> = words[first..after]
                .iter()
                .map(|word| {
                    let node = &njd.nodes[word.node];
                    (
                        kana(node.get_read().unwrap_or_default()),
                        kana(&node.get_pron().to_string()),
                    )
                })
                .collect();
            // A forced word has only its pronunciation to go by, and a rule
            // after tokenization (digits, です) may have rewritten it.
            let agrees = if forced.contains(&(start, end)) {
                Some(said.iter().map(|(_, pron)| pron.as_str()).collect()) == pron
            } else {
                spells(&said, &read)
            };
            if !agrees {
                out.push(Mismatch {
                    start,
                    end,
                    read,
                    pron,
                });
            }
        }
        first = after;
    }
    // A reading over text the front end made no word of was never reached.
    if consumed != readings.len() {
        return Err(Error::Unlabelable("japanese_reading_span".into()));
    }
    Ok(out)
}

/// The dictionary's cheapest entry for `surface` that is read `read`.
fn entry(engine: &Engine, surface: &str, read: &str) -> Option<NJD> {
    let prefix = &engine.dictionary.prefix_dictionary;
    prefix
        .find_surface_iter(surface)
        .filter_map(|word| {
            let bytes = get_word_data(
                &prefix.words_idx_data,
                &prefix.words_data,
                Some(word.word_id.id as usize),
            )?;
            let entry = JPreprocessDictionaryWordEncoding::deserialize(bytes).ok()?;
            let njd = NJD::from_iter([(surface, &entry)]);
            let spelled: String = njd
                .nodes
                .iter()
                .map(|node| kana(node.get_read().unwrap_or_default()))
                .collect();
            (spelled == read).then_some((word.word_cost, njd))
        })
        .min_by_key(|(cost, _)| *cost)
        .map(|(_, njd)| njd)
}

/// Replace each mismatching stretch of the not-yet-preprocessed `njd`.
/// Returns the stretches replaced by a forced word rather than an entry.
fn patch(
    engine: &Engine,
    text: &str,
    njd: &mut NJD,
    wrong: &[Mismatch],
    use_dictionary: bool,
) -> Result<Vec<(usize, usize)>, Error> {
    let words = words(text, njd)?;
    let mut forced = Vec::new();
    for mismatch in wrong.iter().rev() {
        let misaligned = || Error::Unlabelable("japanese_reading_alignment".into());
        let first = words
            .iter()
            .find(|word| word.start == mismatch.start)
            .ok_or_else(misaligned)?
            .node;
        let last = words
            .iter()
            .find(|word| word.end == mismatch.end)
            .ok_or_else(misaligned)?
            .node;
        let surface: String = njd.nodes[first..=last]
            .iter()
            .map(|node| node.get_string())
            .collect();
        let replacement = use_dictionary
            .then(|| entry(engine, &surface, &mismatch.read))
            .flatten();
        let nodes = match replacement {
            Some(entry) => entry.nodes,
            None => {
                forced.push((mismatch.start, mismatch.end));
                let pron = mismatch.pron.as_deref().ok_or_else(|| {
                    Error::Unlabelable(format!("japanese_reading_vowel_length:{}", mismatch.read))
                })?;
                let mut node = njd.nodes[first].clone();
                node.replace_string(&surface);
                node.set_read(&mismatch.read);
                node.set_pron(Pronunciation::parse(pron, 0).map_err(|_| {
                    Error::Unlabelable(format!("japanese_unparsable_reading:{}", mismatch.read))
                })?);
                vec![node]
            }
        };
        njd.nodes.splice(first..=last, nodes);
    }
    Ok(forced)
}

/// The preprocessed NJD for `text`, agreeing with every reading, and whether
/// that took a word the dictionary does not have.
pub(super) fn resolve(
    engine: &Engine,
    text: &str,
    readings: &[Reading],
) -> Result<(NJD, bool), Error> {
    let mut njd = engine
        .jp
        .text_to_njd(text)
        .map_err(|e| Error::Synth(format!("jpreprocess: {e}")))?;
    if readings.is_empty() {
        njd.preprocess();
        return Ok((njd, false));
    }
    let mut readings: Vec<_> = readings.iter().collect();
    readings.sort_by_key(|r| r.start);
    if readings
        .iter()
        .any(|r| text.get(r.start..r.end).is_none_or(str::is_empty))
        || readings.windows(2).any(|pair| pair[0].end > pair[1].start)
    {
        return Err(Error::Unlabelable("japanese_reading_span".into()));
    }
    // Anything `kana` would drop (a decomposed voicing mark, Latin) would be
    // silently left unsaid.
    if let Some(reading) = readings
        .iter()
        .find(|r| kana(&r.reading).chars().count() != r.reading.chars().count())
    {
        return Err(Error::Unlabelable(format!(
            "japanese_reading_not_kana:{}",
            reading.reading
        )));
    }
    let mut forced = Vec::new();
    // Dictionary entries first, then forced words for whatever the rules
    // after tokenization still read differently.
    for use_dictionary in [true, false] {
        let mut preprocessed = NJD {
            nodes: njd.nodes.clone(),
        };
        preprocessed.preprocess();
        let wrong = mismatches(text, &preprocessed, &readings, &forced)?;
        if wrong.is_empty() {
            return Ok((preprocessed, !forced.is_empty()));
        }
        forced.extend(patch(engine, text, &mut njd, &wrong, use_dictionary)?);
    }
    njd.preprocess();
    match mismatches(text, &njd, &readings, &forced)?.first() {
        None => Ok((njd, !forced.is_empty())),
        Some(mismatch) => Err(Error::Unlabelable(format!(
            "japanese_reading_unenforceable:{}",
            mismatch.read
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_vowels_follow_dictionary_pronunciations() {
        for (reading, pron) in [
            ("ソウスケ", "ソースケ"),
            ("センセイ", "センセー"),
            ("オオキイ", "オーキー"),
            ("クウキ", "クーキ"),
            ("ホントニ", "ホントニ"),
            ("チョウチョ", "チョーチョ"),
            // ン and ッ end the vowel a following kana could lengthen.
            ("シンイ", "シンイ"),
        ] {
            assert_eq!(long_vowels(reading), pron);
        }
    }

    #[test]
    fn spells_takes_reading_or_pronunciation_per_word() {
        let word = |read: &str, pron: &str| (read.to_owned(), pron.to_owned());
        let week = [word("イチ", "イッ"), word("シュウカン", "シューカン")];
        assert!(spells(&week, "イッシュウカン"));
        assert!(!spells(&week, "イッシュカン"));
        assert!(!spells(&[word("", "")], ""));
    }

    /// Phones, and why accent was withheld, with `(substring, kana)` readings.
    fn labels(text: &str, readings: &[(&str, &str)]) -> (String, Option<String>) {
        let readings: Vec<_> = readings
            .iter()
            .map(|(of, reading)| {
                let start = text.find(of).unwrap();
                Reading {
                    start,
                    end: start + of.len(),
                    reading: reading.to_string(),
                }
            })
            .collect();
        let labels = super::super::phonemize(text, &readings).unwrap();
        (labels.phonemes.join(" "), labels.accent_withheld)
    }

    #[test]
    fn a_reading_the_dictionary_has_keeps_its_accent() {
        // Alone, 方 has no context; the caller says which word it is.
        assert_eq!(labels("方", &[]).0, "h o o");
        assert_eq!(labels("方", &[("方", "かた")]).0, "k a t a");
        let text = "俺の金はどこだ";
        assert_eq!(labels(text, &[]).0, "o ɾ e n o k i ɴ w a d o k o d a");
        assert_eq!(
            labels(text, &[("俺", "おれ"), ("金", "かね")]),
            ("o ɾ e n o k a n e w a d o k o d a".into(), None)
        );
        // One reading over several of the tokenizer's words finds the whole word.
        assert_eq!(
            labels("ローリーは二十歳だ。", &[("二十歳", "はたち")]),
            ("ɾ o o ɾ i i w a h a t a tɕ i d a".into(), None)
        );
    }

    #[test]
    fn readings_the_front_end_already_says_change_nothing() {
        // 一 is イチ in the dictionary; the counter rule already makes it イッ.
        for (text, readings) in [
            ("学校", vec![("学", "がっ"), ("校", "こう")]),
            ("一回だけ", vec![("一", "いっ"), ("回", "かい")]),
        ] {
            let readings: Vec<_> = readings
                .into_iter()
                .map(|(of, reading)| {
                    let start = text.find(of).unwrap();
                    Reading {
                        start,
                        end: start + of.len(),
                        reading: reading.into(),
                    }
                })
                .collect();
            assert_eq!(
                super::super::phonemize(text, &readings).unwrap(),
                super::super::phonemize(text, &[]).unwrap()
            );
        }
    }

    #[test]
    fn a_reading_the_dictionary_lacks_is_said_without_accent() {
        let forced = Some("japanese_reading_not_in_dictionary".to_string());
        // A colloquial form, a mis-split word (待 + て, read マテ + テ), a
        // name, and a reading that crosses the tokenizer's words (昨 + 日本).
        for (text, readings, phones) in [
            (
                "本当にごめん",
                vec![("本", "ほん"), ("当", "と")],
                "h o ɴ t o n i ɡ o m e ɴ",
            ),
            ("いや待て", vec![("待", "ま")], "i j a m a t e"),
            (
                "宗介だ",
                vec![("宗", "そう"), ("介", "すけ")],
                "s o o s ɯ̥ᵝ k e d a",
            ),
            (
                "昨日本を買った。",
                vec![("昨日", "きのう"), ("本", "ほん"), ("買", "か")],
                "k i n o o h o ɴ o k a tː a",
            ),
        ] {
            assert_eq!(labels(text, &readings), (phones.into(), forced.clone()));
        }
    }

    #[test]
    fn readings_that_cannot_be_honoured_are_refused() {
        let refused =
            |text: &str, readings: &[Reading]| match super::super::phonemize(text, readings) {
                Err(Error::Unlabelable(reason)) => reason,
                other => panic!("{text}: {other:?}"),
            };
        let reading = |start, end, reading: &str| Reading {
            start,
            end,
            reading: reading.into(),
        };
        // The rest of the word (校) has no reading to compare or to say.
        assert_eq!(
            refused("学校", &[reading(0, 3, "がっ")]),
            "japanese_reading_partial:学校"
        );
        // Not on a character boundary, or overlapping.
        assert_eq!(
            refused("学校", &[reading(1, 3, "が")]),
            "japanese_reading_span"
        );
        assert_eq!(
            refused("学校", &[reading(0, 6, "がっこう"), reading(3, 6, "こう")]),
            "japanese_reading_span"
        );
        // Digits are rewritten into numerals, which no longer line up with the text.
        assert_eq!(
            refused("3時に", &[reading(1, 4, "じ")]),
            "japanese_reading_alignment"
        );
        // A decomposed voicing mark would otherwise be dropped: がた said かた.
        assert_eq!(
            refused("方", &[reading(0, 3, "か\u{3099}た")]),
            "japanese_reading_not_kana:か\u{3099}た"
        );
        // The counter rule rewrites the forced ヒト back to イッ; saying so
        // beats returning "ikkai" as if it agreed.
        assert_eq!(
            refused("一回", &[reading(0, 3, "ひと"), reading(3, 6, "かい")]),
            "japanese_reading_unenforceable:ヒト"
        );
        // One reading over 思う方 has no dictionary word, and does not say
        // whether its おう is long. Per kanji there is nothing to guess.
        assert_eq!(
            refused("思う方", &[reading(0, 9, "おもうかた")]),
            "japanese_reading_vowel_length:オモウカタ"
        );
        assert_eq!(
            labels("思う方", &[("思", "おも"), ("方", "かた")]).0,
            "o m o ɯᵝ k a t a"
        );
    }

    #[test]
    fn a_long_stretch_that_disagrees_is_rejected_quickly() {
        // One reading over forty words, wrong only at the end.
        let text = "猫".repeat(40);
        let reading = Reading {
            start: 0,
            end: text.len(),
            reading: format!("{}いぬ", "ねこ".repeat(39)),
        };
        let labels = super::super::phonemize(&text, &[reading]).unwrap();
        assert!(labels.phonemes.join("").ends_with("inɯᵝ"));
    }
}
