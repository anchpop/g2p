//! Conservative original-input alignment. Failure never changes pronunciation.
use crate::Phonemized;

pub(crate) fn separator(c: char) -> bool {
    c.is_whitespace()
        || ".,!?;:'\"-–—…()[]{}«»‹›‘’“”¿¡·•・．。，、！？；：（）【】「」『』《》〈〉।॥".contains(c)
}

/// Exact monotonic surface alignment, allowing only unspoken separators in gaps.
pub(crate) fn align<'a>(
    text: &str,
    surfaces: impl IntoIterator<Item = &'a str>,
) -> Vec<(usize, usize)> {
    let mut cursor = 0;
    let mut spans = Vec::new();
    for surface in surfaces {
        if surface.is_empty() {
            return Vec::new();
        }
        while !text[cursor..].starts_with(surface) {
            let Some(c) = text[cursor..].chars().next() else {
                return Vec::new();
            };
            if !separator(c) {
                return Vec::new();
            }
            cursor += c.len_utf8();
        }
        spans.push((cursor, cursor + surface.len()));
        cursor += surface.len();
    }
    if !text[cursor..].chars().all(separator) {
        return Vec::new();
    }
    spans
}

/// Validate the whole utterance before committing any merges. Overlapping source
/// ranges (number expansions or a cross-node closure) become one word range.
pub(crate) fn assign(out: &mut Phonemized, text: &str) {
    let mut end = 0;
    let mut previous_start = 0;
    let valid = out.word_text_spans.len() == out.word_spans.len()
        && out.word_text_spans.iter().all(|&(start, stop)| {
            if start >= stop || !text.is_char_boundary(start) || !text.is_char_boundary(stop) {
                return false;
            }
            if start < previous_start {
                return false;
            }
            if start > end && !text[end..start].chars().all(separator) {
                return false;
            }
            previous_start = start;
            end = end.max(stop);
            true
        })
        && text[end..].chars().all(separator);
    if !valid {
        out.word_text_spans.clear();
        return;
    }
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut sources: Vec<(usize, usize)> = Vec::new();
    for (&word, &source) in out.word_spans.iter().zip(&out.word_text_spans) {
        if let Some(last) = sources.last_mut()
            && source.0 < last.1
        {
            last.1 = last.1.max(source.1);
            words.last_mut().unwrap().1 = word.1;
        } else {
            words.push(word);
            sources.push(source);
        }
    }
    out.word_spans = words;
    out.word_text_spans = sources;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_and_overlapping_sources_merge_but_holes_are_withheld() {
        let mut p = Phonemized::from_ipa_tokens("a | b | c").unwrap();
        p.word_text_spans = vec![(0, 3), (0, 3), (4, 6)];
        assign(&mut p, "123 go");
        assert_eq!(p.word_spans, [(0, 2), (2, 3)]);
        assert_eq!(p.word_text_spans, [(0, 3), (4, 6)]);
        let mut overlapping = Phonemized::from_ipa_tokens("a | b").unwrap();
        overlapping.word_text_spans = vec![(0, 3), (2, 6)];
        assign(&mut overlapping, "123 go");
        assert_eq!(overlapping.word_text_spans, [(0, 6)]);
        assert_eq!(overlapping.word_spans, [(0, 2)]);
        for sources in [vec![(0, 1), (4, 6)], vec![(0, 3), (4, 7)]] {
            let mut p = Phonemized::from_ipa_tokens("a | b").unwrap();
            p.word_text_spans = sources;
            assign(&mut p, "123 go");
            assert!(p.word_text_spans.is_empty());
            assert_eq!(p.word_spans, [(0, 1), (1, 2)]);
        }
    }
}
