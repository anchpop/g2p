//! Shared word-internal consonant length normalization, after typed conversion.
use crate::{Phoneme, Phonemized, parse};

pub(crate) fn merge(labels: &mut Phonemized) {
    let n = labels.phonemes.len();
    let mut keep = vec![true; n];
    for &(start, end) in &labels.word_spans {
        let mut i = start;
        while i + 1 < end {
            let phone = labels.phonemes[i];
            // A tone or pitch bearer is a nucleus (Japanese ɴ is a mora), never
            // half of a geminate.
            let bearer = |j: usize| {
                labels.tone.get(j).is_some_and(Option::is_some)
                    || labels.pitch.get(j).is_some_and(Option::is_some)
            };
            if phone == labels.phonemes[i + 1]
                && !parse::starts_with_vowel(phone.as_str())
                && !bearer(i)
                && !bearer(i + 1)
                && let Ok(long) = format!("{phone}ː").parse::<Phoneme>()
            {
                labels.phonemes[i] = long;
                keep[i + 1] = false;
                i += 2;
            } else {
                i += 1;
            }
        }
    }
    if keep.iter().all(|&k| k) {
        return;
    }
    // A boundary after the first consonant stays after the merged consonant;
    // adjoining syllables therefore remain disjoint and cover all phones.
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0);
    for &k in &keep {
        boundaries.push(boundaries.last().unwrap() + usize::from(k));
    }
    for (start, end) in &mut labels.word_spans {
        *start = boundaries[*start];
        *end = boundaries[*end];
    }
    for syllable in &mut labels.syllables {
        syllable.start = boundaries[syllable.start];
        syllable.end = boundaries[syllable.end];
        syllable.nucleus = boundaries[syllable.nucleus];
    }
    retain(&mut labels.phonemes, &keep);
    retain(&mut labels.stress, &keep);
    retain(&mut labels.tone, &keep);
    retain(&mut labels.pitch, &keep);
}

fn retain<T>(values: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    values.retain(|_| {
        let retained = keep[i];
        i += 1;
        retained
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Pitch, Stress, Syllable};

    #[test]
    fn merges_only_identical_consonants_with_inventory_lengths_within_words() {
        let mut p = Phonemized::from_ipa_tokens(
            "k k a a k kʰ | n | n | ɾ ɾ ʈ ʈ ɳ ɳ ɦ ɦ ʋ ʋ tɕ tɕ tʃ tʃ | ɬ ɬ",
        )
        .unwrap();
        let raw = p.raw.clone();
        merge(&mut p);
        assert_eq!(
            p.phonemes.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
            [
                "kː", "a", "a", "k", "kʰ", "n", "n", "ɾː", "ʈː", "ɳː", "ɦː", "ʋː", "tɕː", "tʃː",
                "ɬ", "ɬ"
            ]
        );
        assert_eq!(p.word_spans, [(0, 5), (5, 6), (6, 7), (7, 14), (14, 16)]);
        assert_eq!(p.raw, raw);
        assert!(p.stress.is_empty() && p.tone.is_empty() && p.pitch.is_empty());
    }

    #[test]
    fn moraic_and_tone_bearing_phones_remain_separate() {
        // Japanese gives one word span per utterance; ɴ ɴ across a pause are two moras.
        let mut p = Phonemized::from_ipa_tokens("ɯᵝ ɴ ɴ").unwrap();
        let original = p.clone();
        merge(&mut p);
        assert_eq!(p, original);
        let mut p = Phonemized::from_ipa_tokens("k k").unwrap();
        p.tone = vec![Some(3), Some(3)];
        let original = p.clone();
        merge(&mut p);
        assert_eq!(p, original);
    }

    #[test]
    fn decorated_vowels_remain_separate() {
        for vowel in ["ã", "ẽ", "ĩ", "õ", "ũ", "ä", "ɛ̃", "aː"] {
            let mut labels = Phonemized::from_ipa_tokens(&format!("{vowel} {vowel}")).unwrap();
            let original = labels.clone();
            merge(&mut labels);
            assert_eq!(labels, original, "{vowel}");
        }
    }

    #[test]
    fn preserves_first_metadata_and_reindexes_syllables() {
        let mut p = Phonemized::from_ipa_tokens("a k k a | n n a").unwrap();
        p.stress = vec![
            Stress::None,
            Stress::Primary,
            Stress::Secondary,
            Stress::None,
            Stress::Primary,
            Stress::Secondary,
            Stress::None,
        ];
        p.tone = vec![Some(2), None, None, Some(3), None, None, Some(4)];
        let pitch = Pitch {
            phrase: 1,
            mora: 1,
            phrase_moras: 2,
            nucleus: 1,
            level: 1,
        };
        p.pitch = vec![Some(pitch.clone()), None, None, None, None, None, None];
        p.syllables = vec![
            Syllable {
                start: 0,
                end: 2,
                nucleus: 0,
                moras: 2,
                stressed: false,
            },
            Syllable {
                start: 2,
                end: 4,
                nucleus: 3,
                moras: 1,
                stressed: false,
            },
            Syllable {
                start: 4,
                end: 7,
                nucleus: 6,
                moras: 1,
                stressed: true,
            },
        ];
        merge(&mut p);
        assert_eq!(
            p.stress,
            [
                Stress::None,
                Stress::Primary,
                Stress::None,
                Stress::Primary,
                Stress::None
            ]
        );
        assert_eq!(p.tone, [Some(2), None, Some(3), None, Some(4)]);
        assert_eq!(p.pitch, [Some(pitch), None, None, None, None]);
        assert_eq!(p.word_spans, [(0, 3), (3, 5)]);
        assert_eq!(
            p.syllables
                .iter()
                .map(|s| (s.start, s.end, s.nucleus))
                .collect::<Vec<_>>(),
            [(0, 2, 0), (2, 3, 2), (3, 5, 4)]
        );
    }
}
