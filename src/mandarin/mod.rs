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

use crate::Error;
use std::collections::HashMap;
use std::sync::OnceLock;

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

/// Readings for every Han character in `text` (g2pM), one [`Syllable`] per
/// character. Punctuation and whitespace are skipped. Digits, Latin letters,
/// and characters g2pM has no reading for are refused.
pub fn phonemize(text: &str) -> Result<Vec<Syllable>, Error> {
    let m = model();
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
    let mut readings: Vec<Option<&str>> = Vec::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        ids.push(m.char_index.get(&c).copied().unwrap_or(UNK));
        match m.cedict.get(&c) {
            Some(prons) if prons.len() > 1 => {
                poly.push(i + 1);
                readings.push(None);
            }
            Some(prons) => readings.push(Some(prons[0])),
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
            readings[t - 1] = Some(m.classes[cls]);
        }
    }

    let citation_tones: Vec<_> = readings
        .iter()
        .map(|reading| reading.and_then(|r| r.chars().last()?.to_digit(10).map(|t| t as u8)))
        .collect();
    let surface_tones = sandhi(&chars, &citation_tones);
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

/// Realized tones, indexed by source character. Context always uses citation
/// tones; punctuation other than sentence/clause breaks is transparent.
fn sandhi(chars: &[char], citation: &[Option<u8>]) -> Vec<Option<u8>> {
    assert_eq!(chars.len(), citation.len());
    let mut surface = citation.to_vec();
    let numeral = |c: char| c.is_ascii_digit() || "〇零一二三四五六七八九十百千万亿".contains(c);
    let mut next = None;
    for i in (0..chars.len()).rev() {
        if "。！？，；：.!?,;:".contains(chars[i]) {
            next = None;
            continue;
        }
        let Some(tone) = citation[i] else { continue };
        surface[i] = Some(match chars[i] {
            '不' => {
                if next == Some(4) {
                    2
                } else {
                    4
                }
            }
            '一' => {
                // Ordinal, counting and compound-final uses keep tone 1: 第一,
                // 十一, 一二, weekdays (星期一 周一 礼拜一), months and dates
                // (一月 一号 一日), and 一 closing a lexical compound (统一 唯一
                // 万一 专一 同一 单一 划一 归一 之一 初一 合一 不一 如一).
                let counting = i.checked_sub(1).is_some_and(|p| {
                    "第期周拜统唯万专同单划归之初合不如".contains(chars[p]) || numeral(chars[p])
                }) || chars
                    .get(i + 1)
                    .copied()
                    .is_some_and(|n| "月号日".contains(n) || numeral(n));
                if counting {
                    1
                } else {
                    match next {
                        Some(1..=3) => 4,
                        Some(4) => 2,
                        _ => 1,
                    }
                }
            }
            _ if tone == 3 && next == Some(3) => 2,
            _ => tone,
        });
        next = Some(tone);
    }
    surface
}

#[cfg(test)]
mod tests {
    use super::*;

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
            (
                "你 好",
                &[Some(3), None, Some(3)],
                &[Some(2), None, Some(3)],
            ),
            (
                "你、好",
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
                "二 一天",
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
        for mark in "。！？，；：.!?,;:".chars() {
            for (first, tone) in [('你', 3), ('不', 4), ('一', 1)] {
                assert_eq!(
                    sandhi(&[first, mark, '好'], &[Some(tone), None, Some(3)]),
                    [Some(tone), None, Some(3)]
                );
            }
        }
        for numeral in "〇零一二三四五六七八九十百千万亿0123456789".chars() {
            assert_eq!(
                sandhi(&['一', numeral, '天'], &[Some(1), None, Some(4)]),
                [Some(1), None, Some(4)]
            );
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
