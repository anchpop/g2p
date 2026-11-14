//! Simplified-script Standard Mandarin grapheme-to-phoneme: g2pM, ported.
//!
//! espeak's `cmn` voice is not the label source for Mandarin. Polyphone
//! disambiguation is contextual (行 xíng/háng, 了 le/liǎo) and tone must
//! come from the reading, not the character. lexide labels Mandarin with
//! [g2pM](https://github.com/kakaobrain/g2pM) (MIT): a CEDICT digest gives
//! each character its readings, and a one-layer BiLSTM (64-d character
//! embeddings, 32-d hidden per direction, two dense layers) picks the reading
//! for the 791 polyphonic characters from sentence context. The weights,
//! vocabulary, and dictionary are embedded (`data/`, ~1.7 MB); the forward
//! pass is a few hundred lines of plain arithmetic.
//!
//! Pinyin then becomes IPA through the `pinyin_to_ipa` package's tables
//! (Duanmu 2007 / Lin 2007 conventions), precomputed for every syllable g2pM
//! can emit and embedded as `data/syllables.tsv`, so pypinyin's syllable
//! splitting does not need porting. The tokenization and tone placement are
//! lexide's `mandarin_labels`, with shared diphthong spellings and surface
//! third-tone/不/一 sandhi: the tone number attaches to the phone that
//! carries the pitch contour (or, for the neutral tone, the first vowel-like
//! phone); pitch letters themselves are not tokens.
//!
//! Text with digits, Latin letters, or characters outside the dictionary is
//! refused ([`Error::Unlabelable`]) rather than labeled with a hole where the
//! audio has speech — g2pM passes such characters through and lexide's
//! pipeline silently dropped them.

use crate::{Error, Reading};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

const EMBED: usize = 64;
const HIDDEN: usize = 32;
const PAD: usize = 0;
const UNK: usize = 1;
const BOS: usize = 2;
const EOS: usize = 3;

static CHARS: &str = include_str!("data/chars.txt");
static CLASSES: &str = include_str!("data/classes.txt");
static CEDICT: &str = include_str!("data/cedict.tsv");
static SYLLABLES: &str = include_str!("data/syllables.tsv");
static WEIGHTS: &[u8] = include_bytes!("data/weights.bin");

struct Model {
    char_index: HashMap<char, usize>,
    classes: Vec<&'static str>,
    /// Character → readings (pinyin with tone digit), first is the default.
    cedict: HashMap<char, Vec<&'static str>>,
    /// Toneless pinyin → IPA variants; each variant is phones with a `0`
    /// placeholder on the tone-bearing phone.
    syllables: HashMap<&'static str, Vec<Vec<&'static str>>>,
    embedding: Vec<f32>,
    fw: Lstm,
    bw: Lstm,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
}

struct Lstm {
    w_ih: Vec<f32>, // [4H, E]
    w_hh: Vec<f32>, // [4H, H]
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(Model::load)
}

impl Model {
    fn load() -> Model {
        // Rows 0..4 are the special tokens (<PAD>, <UNK>, BOS, EOS), which
        // are multi-character strings; every other row is one character.
        let rows: Vec<&str> = CHARS.lines().collect();
        assert_eq!(rows[PAD], "<PAD>");
        assert_eq!(rows[UNK], "<UNK>");
        let char_index = rows
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                let mut it = l.chars();
                let c = it.next()?;
                it.next().is_none().then_some((c, i))
            })
            .collect();
        let n_chars = rows.len();
        let classes: Vec<&str> = CLASSES.lines().collect();
        let cedict = CEDICT
            .lines()
            .map(|l| {
                let (c, prons) = l.split_once('\t').expect("cedict.tsv line");
                let c = c.chars().next().expect("cedict char");
                (c, prons.split(',').collect())
            })
            .collect();
        let syllables = SYLLABLES
            .lines()
            .map(|l| {
                let (syl, variants) = l.split_once('\t').expect("syllables.tsv line");
                (
                    syl,
                    variants
                        .split(" ; ")
                        .map(|v| v.split('|').collect())
                        .collect(),
                )
            })
            .collect();

        let mut off = 0usize;
        let mut take = |n: usize| -> Vec<f32> {
            let bytes = &WEIGHTS[off..off + 4 * n];
            off += 4 * n;
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let n_classes = classes.len();
        let embedding = take(n_chars * EMBED);
        let fw = Lstm {
            w_ih: take(4 * HIDDEN * EMBED),
            w_hh: take(4 * HIDDEN * HIDDEN),
            b_ih: take(4 * HIDDEN),
            b_hh: take(4 * HIDDEN),
        };
        let bw = Lstm {
            w_ih: take(4 * HIDDEN * EMBED),
            w_hh: take(4 * HIDDEN * HIDDEN),
            b_ih: take(4 * HIDDEN),
            b_hh: take(4 * HIDDEN),
        };
        let fc1_w = take(HIDDEN * 2 * HIDDEN);
        let fc1_b = take(HIDDEN);
        let fc2_w = take(n_classes * HIDDEN);
        let fc2_b = take(n_classes);
        assert_eq!(off, WEIGHTS.len(), "weights.bin size mismatch");
        Model {
            char_index,
            classes,
            cedict,
            syllables,
            embedding,
            fw,
            bw,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        }
    }

    fn embed(&self, id: usize) -> &[f32] {
        &self.embedding[id * EMBED..(id + 1) * EMBED]
    }

    /// Hidden states of one direction over `ids` (in the order given).
    fn run_lstm(&self, lstm: &Lstm, ids: &[usize]) -> Vec<[f64; HIDDEN]> {
        let mut h = [0f64; HIDDEN];
        let mut c = [0f64; HIDDEN];
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            let x = self.embed(id);
            // PyTorch gate order: i, f, g, o — each a block of HIDDEN rows.
            let mut gates = [0f64; 4 * HIDDEN];
            for (r, g) in gates.iter_mut().enumerate() {
                let mut acc = lstm.b_ih[r] as f64 + lstm.b_hh[r] as f64;
                let wi = &lstm.w_ih[r * EMBED..(r + 1) * EMBED];
                for (a, b) in wi.iter().zip(x) {
                    acc += *a as f64 * *b as f64;
                }
                let wh = &lstm.w_hh[r * HIDDEN..(r + 1) * HIDDEN];
                for (a, b) in wh.iter().zip(&h) {
                    acc += *a as f64 * b;
                }
                *g = acc;
            }
            let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
            for k in 0..HIDDEN {
                let i = sigmoid(gates[k]);
                let f = sigmoid(gates[HIDDEN + k]);
                let g = gates[2 * HIDDEN + k].tanh();
                let o = sigmoid(gates[3 * HIDDEN + k]);
                c[k] = f * c[k] + i * g;
                h[k] = o * c[k].tanh();
            }
            out.push(h);
        }
        out
    }

    /// Class index of the reading for each position in `poly` (indices into
    /// `ids`, which already carry BOS/EOS).
    fn predict(&self, ids: &[usize], poly: &[usize]) -> Vec<usize> {
        let fw = self.run_lstm(&self.fw, ids);
        let rev: Vec<usize> = ids.iter().rev().copied().collect();
        let mut bw = self.run_lstm(&self.bw, &rev);
        bw.reverse();
        poly.iter()
            .map(|&t| {
                let mut hidden = [0f64; 2 * HIDDEN];
                hidden[..HIDDEN].copy_from_slice(&fw[t]);
                hidden[HIDDEN..].copy_from_slice(&bw[t]);
                let mut l1 = [0f64; HIDDEN];
                for (j, v) in l1.iter_mut().enumerate() {
                    let mut acc = self.fc1_b[j] as f64;
                    for (a, b) in self.fc1_w[j * 2 * HIDDEN..(j + 1) * 2 * HIDDEN]
                        .iter()
                        .zip(&hidden)
                    {
                        acc += *a as f64 * b;
                    }
                    *v = acc.max(0.0);
                }
                let mut best = (f64::NEG_INFINITY, 0usize);
                for k in 0..self.classes.len() {
                    let mut acc = self.fc2_b[k] as f64;
                    for (a, b) in self.fc2_w[k * HIDDEN..(k + 1) * HIDDEN].iter().zip(&l1) {
                        acc += *a as f64 * b;
                    }
                    if acc > best.0 {
                        best = (acc, k);
                    }
                }
                best.1
            })
            .collect()
    }
}

pub use g2p_types::mandarin::Syllable;

/// Pinyin as g2pM spells it: ASCII syllable, `v` for ü, tone digit 1–5.
fn numbered(syllable: &str) -> Option<String> {
    let mut tone = None;
    let mut base = String::new();
    for c in syllable.to_lowercase().replace("u:", "v").nfd() {
        let mark = match c {
            '\u{0304}' | '1' => 1,
            '\u{0301}' | '2' => 2,
            '\u{030c}' | '3' => 3,
            '\u{0300}' | '4' => 4,
            '5' => 5,
            // NFD ü is u + diaeresis.
            '\u{0308}' if base.ends_with('u') => {
                base.pop();
                base.push('v');
                continue;
            }
            'a'..='z' => {
                base.push(c);
                continue;
            }
            _ => return None,
        };
        if tone.replace(mark).is_some() {
            return None;
        }
    }
    (!base.is_empty()).then(|| format!("{base}{}", tone.unwrap_or(5)))
}

/// Each supplied reading as one g2pM syllable per character index of `text`.
fn supplied(text: &str, readings: &[Reading]) -> Result<Vec<Option<String>>, Error> {
    let mut out = vec![None; text.chars().count()];
    for reading in readings {
        let refuse = || Error::Unlabelable(format!("mandarin_reading:{}", reading.reading));
        let stretch = text.get(reading.start..reading.end).ok_or_else(refuse)?;
        let first = text[..reading.start].chars().count();
        let spoken: Vec<_> = stretch
            .chars()
            .enumerate()
            .filter(|(_, c)| c.is_alphanumeric())
            .map(|(i, _)| first + i)
            .collect();
        let syllables: Vec<_> = reading.reading.split_whitespace().collect();
        if spoken.len() != syllables.len() {
            return Err(refuse());
        }
        for (index, syllable) in spoken.into_iter().zip(syllables) {
            if out[index]
                .replace(numbered(syllable).ok_or_else(refuse)?)
                .is_some()
            {
                return Err(refuse());
            }
        }
    }
    Ok(out)
}

/// Which supplied reading each character index of `text` falls in. A reading
/// is the only word boundary this backend knows.
fn words(text: &str, readings: &[Reading]) -> Vec<Option<usize>> {
    let mut out = vec![None; text.chars().count()];
    for (word, reading) in readings.iter().enumerate() {
        let first = text[..reading.start].chars().count();
        let len = text[reading.start..reading.end].chars().count();
        out[first..first + len].fill(Some(word));
    }
    out
}

fn tone_digit(reading: &str) -> Option<u8> {
    reading.chars().last()?.to_digit(10).map(|t| t as u8)
}

/// One [`Syllable`] per Han character in `text`: the supplied reading where
/// there is one, g2pM's otherwise. Punctuation and whitespace are skipped.
/// Digits, Latin letters, and characters with neither a supplied reading nor
/// a g2pM one are refused. Each reading's stretch is treated as one word by
/// the tone sandhi.
pub fn phonemize(text: &str, readings: &[Reading]) -> Result<Vec<Syllable>, Error> {
    let m = model();
    let mut supplied = supplied(text, readings)?;
    let words = words(text, readings);
    let digits: String = text
        .chars()
        .filter(|c| c.is_ascii_digit() || ('０'..='９').contains(c))
        .collect();
    if !digits.is_empty() {
        return Err(Error::Unlabelable(format!("mandarin_digits:{digits}")));
    }
    let latin: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if !latin.is_empty() {
        return Err(Error::Unlabelable(format!(
            "mandarin_latin_script:{}",
            latin.join(",")
        )));
    }
    // g2pM sees the whole string (punctuation included) as context; keep
    // that, so predictions match the reference implementation.
    let chars: Vec<char> = text.chars().collect();
    let mut ids: Vec<usize> = Vec::with_capacity(chars.len() + 2);
    ids.push(BOS);
    let mut poly: Vec<usize> = Vec::new();
    let mut readings: Vec<Option<Cow<str>>> = Vec::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        ids.push(m.char_index.get(&c).copied().unwrap_or(UNK));
        if let Some(known) = supplied[i].take() {
            readings.push(Some(known.into()));
            continue;
        }
        match m.cedict.get(&c) {
            Some(prons) if prons.len() > 1 => {
                poly.push(i + 1);
                readings.push(None);
            }
            Some(prons) => readings.push(Some(prons[0].into())),
            None => {
                // Anything letter-like without a reading — a rare Han
                // character, kana, Hangul, fullwidth or accented Latin — is
                // spoken but would be missing from the labels.
                if c.is_alphanumeric() {
                    return Err(Error::Unlabelable(format!("mandarin_unknown_char:{c}")));
                }
                readings.push(None);
            }
        }
    }
    ids.push(EOS);
    let _ = PAD;
    if !poly.is_empty() {
        let preds = m.predict(&ids, &poly);
        for (&t, cls) in poly.iter().zip(preds) {
            readings[t - 1] = Some(m.classes[cls].into());
        }
    }

    let citation_tones: Vec<_> = readings
        .iter()
        .map(|reading| reading.as_deref().and_then(tone_digit))
        .collect();
    // A neutral syllable still has its character's tone for the 一 and 不
    // rules: 一个 is yí ge, because 个 is gè.
    let lexical_tones: Vec<_> = citation_tones
        .iter()
        .zip(&chars)
        .map(|(&tone, c)| match tone {
            Some(5) => m
                .cedict
                .get(c)
                .and_then(|prons| prons.iter().filter_map(|p| tone_digit(p)).find(|&t| t != 5))
                .or(tone),
            _ => tone,
        })
        .collect();
    let surface_tones = sandhi(&chars, &citation_tones, &lexical_tones, &words);
    let mut out = Vec::new();
    for (i, reading) in readings.into_iter().enumerate() {
        let Some(reading) = reading else { continue };
        let mut pinyin = reading.replace("u:", "v");
        // g2pM tags erhua 儿 two ways: a full "er2" syllable in 这儿, but the
        // bare suffix "r5" in 哪儿. Same sound; normalize to the syllable.
        if pinyin == "r5" {
            pinyin = "er5".to_string();
        }
        let (syl, tone) = match pinyin.char_indices().last() {
            Some((idx, t @ '1'..='5'))
                if !pinyin[..idx].is_empty()
                    && pinyin[..idx].chars().all(|c| c.is_ascii_lowercase()) =>
            {
                (&pinyin[..idx], t as u8 - b'0')
            }
            // Multi-syllable "readings" (unit characters like ㎏) and
            // toneless entries: g2pM has no usable reading.
            _ => {
                return Err(Error::Unlabelable(format!(
                    "mandarin_unknown_char:{}",
                    chars[i]
                )));
            }
        };
        let tone = surface_tones[i].unwrap_or(tone);
        let Some(variants) = m.syllables.get(syl) else {
            return Err(Error::Unlabelable(format!(
                "mandarin_unknown_syllable:{pinyin}"
            )));
        };
        let variant = &variants[0];
        let mut phonemes = Vec::with_capacity(variant.len());
        let mut tones = Vec::with_capacity(variant.len());
        let mut assigned = false;
        for phone in variant {
            let bears = phone.contains('0');
            phonemes.push(phone.replace('0', ""));
            // Tone 5 has no contour, so nothing bears it here; see below.
            tones.push((bears && !assigned && tone != 5).then_some(tone));
            assigned |= bears && tone != 5;
        }
        if !assigned {
            // Neutral tone: attach to the first vowel-like phone (ɚ included
            // for toneless erhua).
            let bearer = phonemes.iter().position(|p| {
                p.chars()
                    .next()
                    .is_some_and(|c| "iyɨɯuɪʊeɤoəɚɛaɑɔɻɹzʐ".contains(c))
            });
            match bearer {
                Some(b) => tones[b] = Some(tone),
                None => {
                    return Err(Error::Unlabelable(format!(
                        "mandarin_no_tone_bearer:{pinyin}"
                    )));
                }
            }
        }
        out.push(Syllable {
            char: chars[i],
            pinyin,
            phonemes,
            tone: tones,
        });
    }
    Ok(out)
}

/// Words whose third tone rises before a neutral syllable that is lexically a
/// third tone: 小姐 xiáojie. Most keep the low third (姐姐 jiějie, 耳朵,
/// 椅子), so this is a word list, not a rule.
const RISES_BEFORE_NEUTRAL: [[char; 2]; 1] = [['小', '姐']];

/// Realized tones, indexed by source character. `lexical` is the tone the 一
/// and 不 rules (and [`RISES_BEFORE_NEUTRAL`]) consult: the citation tone,
/// except that a neutral syllable keeps its character's tone (个 gè). `words` gives the word each character
/// belongs to where known (see [`words`]); characters in no word count as
/// one unknown word, so without readings everything is one word.
///
/// Pauses block sandhi: clause punctuation, whitespace (subtitles mark clause
/// breaks with spaces), ellipses, dashes and the enumeration comma. Other
/// punctuation is transparent.
///
/// Third-tone sandhi is cyclic: inside a word it looks at the next syllable's
/// citation tone, and across a word boundary at the next word's surface tone,
/// so 小老鼠 is xiǎo láoshǔ, not xiáo láo shǔ.
fn sandhi(
    chars: &[char],
    citation: &[Option<u8>],
    lexical: &[Option<u8>],
    words: &[Option<usize>],
) -> Vec<Option<u8>> {
    assert_eq!(chars.len(), citation.len());
    let mut surface = citation.to_vec();
    let digit = |c: char| c.is_ascii_digit() || "〇零一二三四五六七八九".contains(c);
    let numeral = |c: char| digit(c) || "十百千万亿".contains(c);
    let same_word = |a: usize, b: usize| words[a] == words[b];
    // X不X and X一X with the same character on both sides: A-not-A questions
    // (要不要) and verb reduplication (看一看), where 不 and 一 are neutral. The
    // same 不 or 一 before X makes it a repeated word instead: 一个一个 is
    // yí ge yí ge, 不错不错 bú cuò bú cuò. So does an X that ends a longer
    // word: 现在 + 不在 is bú zài.
    let reduplicated = |i: usize| {
        i.checked_sub(1).is_some_and(|p| {
            chars.get(i + 1) == Some(&chars[p])
                && chars[p] != chars[i]
                && citation[p].is_some()
                && p.checked_sub(1).is_none_or(|pp| {
                    chars[pp] != chars[i] && (words[p].is_none() || words[pp] != words[p])
                })
        })
    };
    let mut next: Option<usize> = None;
    for i in (0..chars.len()).rev() {
        if chars[i].is_whitespace() || "。！？，；：.!?,;:…⋯—―～~、".contains(chars[i])
        {
            next = None;
            continue;
        }
        let Some(tone) = citation[i] else { continue };
        let next_lexical = next.and_then(|n| lexical[n]);
        // The 不 and 一 rules are about bù and yī. A supplied reading can be
        // another word written the same way: neutral in 对不起, fǒu for 不.
        surface[i] = Some(match chars[i] {
            '不' if tone == 4 => {
                if reduplicated(i) {
                    5
                } else if next_lexical == Some(4) {
                    2
                } else {
                    4
                }
            }
            '一' if tone == 1 => {
                // Ordinal, counting and compound-final uses keep tone 1: 第一,
                // 十一, 一二, weekdays (星期一 周一 礼拜一), months and dates
                // (一月 一号 一日), and 一 closing a lexical compound (统一 唯一
                // 万一 专一 同一 单一 划一 归一 之一 初一 合一). Except after 第,
                // both characters must be in one word: 不一样 is bù yíyàng,
                // 唯一一个 wéiyī yí ge.
                let counting = i.checked_sub(1).is_some_and(|p| {
                    chars[p] == '第'
                        || (same_word(p, i)
                            && (numeral(chars[p])
                                || "期周拜统唯万专同单划归之初合".contains(chars[p])))
                }) || chars
                    .get(i + 1)
                    .copied()
                    .is_some_and(|n| "月号日".contains(n) || digit(n));
                if reduplicated(i) {
                    5
                } else if counting {
                    1
                } else {
                    match next_lexical {
                        Some(1..=3) => 4,
                        Some(4) => 2,
                        _ => 1,
                    }
                }
            }
            _ if tone == 3 => {
                let next_tone = next.and_then(|n| {
                    if !same_word(i, n) {
                        surface[n]
                    } else if RISES_BEFORE_NEUTRAL.contains(&[chars[i], chars[n]]) {
                        lexical[n]
                    } else {
                        citation[n]
                    }
                });
                if next_tone == Some(3) { 2 } else { 3 }
            }
            _ => tone,
        });
        next = Some(i);
    }
    surface
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phonemize(text: &str) -> Result<Vec<Syllable>, Error> {
        super::phonemize(text, &[])
    }

    fn reading(text: &str, of: &str, reading: &str) -> Reading {
        let start = text.find(of).unwrap();
        Reading {
            start,
            end: start + of.len(),
            reading: reading.into(),
        }
    }

    #[test]
    fn supplied_readings_replace_the_predictor() {
        let pinyin = |text: &str, readings: &[Reading]| {
            super::phonemize(text, readings)
                .unwrap()
                .into_iter()
                .map(|s| s.pinyin)
                .collect::<Vec<_>>()
                .join(" ")
        };
        // Alone, 得 has no context; the caller says which word it is.
        assert_eq!(pinyin("得", &[]), "de2");
        assert_eq!(pinyin("得", &[reading("得", "得", "děi")]), "dei3");
        assert_eq!(pinyin("得", &[reading("得", "得", "de")]), "de5");
        // One syllable per character, numbered or tone-marked, ü either way.
        let text = "我的女儿在这儿";
        assert_eq!(
            pinyin(
                text,
                &[
                    reading(text, "女儿", "nǚ ér"),
                    reading(text, "这儿", "zhe4 r")
                ]
            ),
            "wo3 de5 nv3 er2 zai4 zhe4 er5"
        );
        // A character g2pM does not know is labelable once its reading is given.
        assert!(phonemize("小凪").is_err());
        assert_eq!(
            pinyin("小凪", &[reading("小凪", "凪", "zhǐ")]),
            "xiao3 zhi3"
        );
        for bad in ["nǚ", "nǚ ér x", "n3ü", "女"] {
            assert!(
                matches!(
                    super::phonemize(text, &[reading(text, "女儿", bad)]),
                    Err(Error::Unlabelable(r)) if r.starts_with("mandarin_reading")
                ),
                "{bad}"
            );
        }
    }

    #[test]
    fn supplied_neutral_tones_are_kept_and_sandhi_still_applies() {
        let tones = |text: &str, readings: &[Reading]| {
            super::phonemize(text, readings)
                .unwrap()
                .into_iter()
                .map(|s| s.tone.into_iter().flatten().next().unwrap())
                .collect::<Vec<_>>()
        };
        let text = "对不起";
        assert_eq!(tones(text, &[]), [4, 4, 3]);
        assert_eq!(tones(text, &[reading(text, "不", "bu")]), [4, 5, 3]);
        // Readings are citation tones: 不 before a fourth tone still rises.
        assert_eq!(tones("不对", &[reading("不对", "不", "bù")]), [2, 4]);
        // The 不 rule is bù's, not the character's.
        assert_eq!(tones("不", &[reading("不", "不", "fǒu")]), [3]);
        assert_eq!(tones("你好", &[reading("你好", "你好", "nǐ hǎo")]), [2, 3]);
    }

    #[test]
    fn all_table_variants_use_typed_inventory() {
        for (syllable, variants) in &model().syllables {
            for variant in variants {
                for phone in variant {
                    assert!(
                        phone.replace('0', "").parse::<crate::Phoneme>().is_ok(),
                        "{syllable}: {phone}"
                    );
                }
            }
        }
    }

    /// Sandhi with no neutral syllables and no known words.
    fn sandhi(chars: &[char], citation: &[Option<u8>]) -> Vec<Option<u8>> {
        super::sandhi(chars, citation, citation, &vec![None; chars.len()])
    }

    #[test]
    fn neutral_classifiers_and_word_boundaries_steer_yi_and_bu() {
        let chars = |text: &str| text.chars().collect::<Vec<_>>();
        let t = |tones: &[u8]| tones.iter().map(|&t| Some(t)).collect::<Vec<_>>();
        // 一个: 个 is neutral but lexically gè.
        assert_eq!(
            super::sandhi(&chars("一个"), &t(&[1, 5]), &t(&[1, 4]), &[None, None]),
            t(&[2, 5])
        );
        // 一百 is yìbǎi: only digits make 一 a counting number.
        assert_eq!(sandhi(&chars("一百"), &t(&[1, 3])), t(&[4, 3]));
        // A compound-final 一 needs its partner in the same word.
        let yiyang = (chars("不一样"), t(&[4, 1, 4]));
        assert_eq!(sandhi(&yiyang.0, &yiyang.1), t(&[4, 2, 4]));
        let simple = (chars("简单一些"), t(&[3, 1, 1, 1]));
        assert_eq!(sandhi(&simple.0, &simple.1), t(&[3, 1, 1, 1]));
        assert_eq!(
            super::sandhi(
                &simple.0,
                &simple.1,
                &simple.1,
                &[Some(0), Some(0), Some(1), Some(1)]
            ),
            t(&[3, 1, 4, 1])
        );
        // A-not-A and verb reduplication are neutral; 一个一个 is a quantity.
        assert_eq!(sandhi(&chars("要不要"), &t(&[4, 4, 4])), t(&[4, 5, 4]));
        assert_eq!(sandhi(&chars("看一看"), &t(&[4, 1, 4])), t(&[4, 5, 4]));
        assert_eq!(
            super::sandhi(
                &chars("一个一个"),
                &t(&[1, 5, 1, 5]),
                &t(&[1, 4, 1, 4]),
                &[None; 4]
            ),
            t(&[2, 5, 2, 5])
        );
        // A repeated word is not A-not-A: 不错不错 is bú cuò bú cuò.
        assert_eq!(
            sandhi(&chars("不错不错"), &t(&[4, 4, 4, 4])),
            t(&[2, 4, 2, 4])
        );
        // 第 makes an ordinal even as its own word; other partners must share
        // 一's word, so 唯一 + 一个 is wéiyī yí ge.
        let w = |ws: &[usize]| ws.iter().map(|&w| Some(w)).collect::<Vec<_>>();
        let first = (chars("第一天"), t(&[4, 1, 1]));
        assert_eq!(
            super::sandhi(&first.0, &first.1, &first.1, &w(&[0, 1, 2])),
            t(&[4, 1, 1])
        );
        let only = (chars("唯一一个"), t(&[2, 1, 1, 5]), t(&[2, 1, 1, 4]));
        assert_eq!(
            super::sandhi(&only.0, &only.1, &only.2, &w(&[0, 0, 1, 1])),
            t(&[2, 1, 2, 5])
        );
        // X不X is not A-not-A when X ends a longer word: 现在 + 不在.
        let now = (chars("现在不在"), t(&[4, 4, 4, 4]));
        assert_eq!(
            super::sandhi(&now.0, &now.1, &now.1, &w(&[0, 0, 1, 1])),
            t(&[4, 4, 2, 4])
        );
        // 小姐 rises before its neutral 姐; 姐姐 keeps the low third.
        let miss = (chars("小姐"), t(&[3, 5]), t(&[3, 3]));
        assert_eq!(
            super::sandhi(&miss.0, &miss.1, &miss.2, &[None, None]),
            t(&[2, 5])
        );
        let sister = (chars("姐姐"), t(&[3, 5]), t(&[3, 3]));
        assert_eq!(
            super::sandhi(&sister.0, &sister.1, &sister.2, &[None, None]),
            t(&[3, 5])
        );
        // Third tone is cyclic across words: 小 + 老鼠.
        let mouse = (chars("小老鼠"), t(&[3, 3, 3]));
        assert_eq!(sandhi(&mouse.0, &mouse.1), t(&[2, 2, 3]));
        assert_eq!(
            super::sandhi(&mouse.0, &mouse.1, &mouse.1, &[Some(0), Some(1), Some(1)]),
            t(&[3, 2, 3])
        );
    }

    #[test]
    fn pure_surface_sandhi() {
        type Case<'a> = (&'a str, &'a [Option<u8>], &'a [Option<u8>]);
        let cases: &[Case<'_>] = &[
            ("你好", &[Some(3), Some(3)], &[Some(2), Some(3)]),
            (
                "我很好",
                &[Some(3), Some(3), Some(3)],
                &[Some(2), Some(2), Some(3)],
            ),
            // Subtitles mark a clause break with a space; it is a pause.
            (
                "你 好",
                &[Some(3), None, Some(3)],
                &[Some(3), None, Some(3)],
            ),
            (
                "你……好",
                &[Some(3), None, None, Some(3)],
                &[Some(3), None, None, Some(3)],
            ),
            // Quotation marks are transparent.
            (
                "你“好",
                &[Some(3), None, Some(3)],
                &[Some(2), None, Some(3)],
            ),
            ("不对", &[Some(4), Some(4)], &[Some(2), Some(4)]),
            ("不来", &[Some(4), Some(2)], &[Some(4), Some(2)]),
            ("不", &[Some(4)], &[Some(4)]),
            ("一天", &[Some(1), Some(1)], &[Some(4), Some(1)]),
            ("一年", &[Some(1), Some(2)], &[Some(4), Some(2)]),
            ("一本", &[Some(1), Some(3)], &[Some(4), Some(3)]),
            ("一个", &[Some(1), Some(4)], &[Some(2), Some(4)]),
            ("一", &[Some(1)], &[Some(1)]),
            (
                "第一课",
                &[Some(4), Some(1), Some(4)],
                &[Some(4), Some(1), Some(4)],
            ),
            ("一二", &[Some(1), Some(4)], &[Some(1), Some(4)]),
            (
                "星期一开会",
                &[Some(1), Some(1), Some(1), Some(1), Some(4)],
                &[Some(1), Some(1), Some(1), Some(1), Some(4)],
            ),
            (
                "周一见",
                &[Some(1), Some(1), Some(4)],
                &[Some(1), Some(1), Some(4)],
            ),
            ("一月", &[Some(1), Some(4)], &[Some(1), Some(4)]),
            (
                "统一思想",
                &[Some(3), Some(1), Some(1), Some(3)],
                &[Some(3), Some(1), Some(1), Some(3)],
            ),
            (
                "唯一办法",
                &[Some(2), Some(1), Some(4), Some(3)],
                &[Some(2), Some(1), Some(4), Some(3)],
            ),
            (
                "你好；你好",
                &[Some(3), Some(3), None, Some(3), Some(3)],
                &[Some(2), Some(3), None, Some(2), Some(3)],
            ),
            ("一号", &[Some(1), Some(4)], &[Some(1), Some(4)]),
            (
                "二一天",
                &[Some(4), Some(1), Some(1)],
                &[Some(4), Some(1), Some(1)],
            ),
            // Numeral checks are character-adjacent, not syllable-adjacent.
            (
                "二、一天",
                &[Some(4), None, Some(1), Some(1)],
                &[Some(4), None, Some(4), Some(1)],
            ),
            // 不 and 一 consult citation tone 4 even when the next 不 surfaces as 2.
            (
                "一不对",
                &[Some(1), Some(4), Some(4)],
                &[Some(2), Some(2), Some(4)],
            ),
        ];
        for &(text, citation, expected) in cases {
            assert_eq!(
                sandhi(&text.chars().collect::<Vec<_>>(), citation),
                expected,
                "{text}"
            );
        }
        for mark in "。！？，；：.!?,;: …⋯—―～~、".chars() {
            for (first, tone) in [('你', 3), ('不', 4), ('一', 1)] {
                assert_eq!(
                    sandhi(&[first, mark, '好'], &[Some(tone), None, Some(3)]),
                    [Some(tone), None, Some(3)]
                );
            }
        }
        for numeral in "〇零一二三四五六七八九0123456789".chars() {
            assert_eq!(
                sandhi(&['一', numeral, '天'], &[Some(1), None, Some(4)]),
                [Some(1), None, Some(4)]
            );
        }
        for numeral in "〇零一二三四五六七八九十百千万亿0123456789".chars() {
            assert_eq!(
                sandhi(&[numeral, '一', '天'], &[None, Some(1), Some(1)]),
                [None, Some(1), Some(1)]
            );
        }
    }

    #[test]
    fn integrated_surface_tones_keep_lexical_pinyin() {
        for (text, expected) in [
            ("你好", vec![2, 3]),
            ("我很好", vec![2, 2, 3]),
            ("不对", vec![2, 4]),
            ("一天", vec![4, 1]),
            ("一年", vec![4, 2]),
            ("一本", vec![4, 3]),
            ("一个", vec![2, 4]),
        ] {
            let labels = phonemize(text).unwrap();
            assert_eq!(
                labels
                    .iter()
                    .flat_map(|s| s.tone.iter().flatten().copied())
                    .collect::<Vec<_>>(),
                expected,
                "{text}"
            );
        }
        assert_eq!(phonemize("你好").unwrap()[0].pinyin, "ni3");
    }

    fn labels(text: &str) -> Vec<(String, String)> {
        phonemize(text)
            .unwrap()
            .into_iter()
            .map(|s| (s.pinyin, s.phonemes.join(" ")))
            .collect()
    }

    #[test]
    fn monophones_come_from_the_dictionary() {
        assert_eq!(
            labels("你好"),
            [("ni3".into(), "n i".into()), ("hao3".into(), "x aʊ".into())]
        );
    }

    #[test]
    fn polyphones_are_disambiguated_in_context() {
        // 行 is xíng "walk" in 行走 and háng "row/bank" in 银行.
        let walk = labels("行走")[0].0.clone();
        let bank = labels("银行")[1].0.clone();
        assert_eq!(walk, "xing2");
        assert_eq!(bank, "hang2");
    }

    #[test]
    fn tone_attaches_to_the_contour_phone() {
        let s = &phonemize("中").unwrap()[0];
        assert_eq!(s.phonemes, ["ʈʂ", "ʊ", "ŋ"]);
        assert_eq!(s.tone, [None, Some(1), None]);
        // Neutral tone goes on the first vowel-like phone.
        let le = phonemize("了").unwrap();
        let le = le
            .iter()
            .find(|s| s.pinyin == "le5")
            .expect("了 as le5 in isolation");
        assert_eq!(le.tone, [None, Some(5)]);
    }

    #[test]
    fn erhua_suffix_is_a_syllable() {
        let s = phonemize("哪儿").unwrap();
        assert_eq!(s[1].pinyin, "er5");
        assert_eq!(s[1].phonemes, ["ɚ"]);
        assert_eq!(s[1].tone, [Some(5)]);
    }

    #[test]
    fn refuses_holes() {
        assert!(
            matches!(phonemize("我有2个"), Err(Error::Unlabelable(r)) if r.starts_with("mandarin_digits"))
        );
        assert!(
            matches!(phonemize("我用AOL"), Err(Error::Unlabelable(r)) if r.starts_with("mandarin_latin"))
        );
        // Punctuation is fine.
        assert_eq!(labels("你好，世界！").len(), 4);
    }
}
