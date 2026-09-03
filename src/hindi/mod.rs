//! Hindi grapheme-to-phoneme: lexide's `schwa-stress-hin` chain, ported.
//!
//! espeak's `hi` voice is not used for Hindi. Its labels emit aspiration as a
//! standalone `ʰ`, which the pronunciation model's vocabulary does not have,
//! and it does not do schwa deletion — Hindi's central orthography-to-sound
//! problem. This chain instead:
//!
//! 1. transliterates Devanagari into Google's `hi_ur` unit scheme, with every
//!    inherent schwa written out (`transliterate.py` from
//!    aryamanarora/schwa-deletion, MIT);
//! 2. decides which schwas survive with that repo's logistic-regression
//!    classifier (phonological features of the five units either side;
//!    94–95% held-out accuracy per its authors);
//! 3. maps units to broad IPA, folding aspiration into its consonant and
//!    nasalization onto its vowel;
//! 4. assigns lexical stress with Roy's (2017) surface syllable-weight rules,
//!    keeping the syllable spans.
//!
//! [`Canon::Legacy`] reproduces lexide's Python output byte for byte (provider
//! schema 5, the labels the 2026-08 corpus was built from). [`Canon::Current`]
//! adds corrections found in a 2026-09-02 audit against Wiktionary and the
//! schwa repo's gold lists; see the [`Canon`] docs.

mod model_data;

use crate::Error;
use crate::parse::Stress;
use model_data::{COEF, FEATURES, INTERCEPT, PHONS};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Which label convention to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Canon {
    /// Byte-identical to lexide's `schwa-stress-hin` (provider schema 5): what
    /// the deployed pronunciation model was trained on. Use this to score
    /// against that model.
    Legacy,
    /// Legacy plus the audited corrections:
    /// * `/ə/` beside `/ɦ/` is `[ɛ]` (शहर, कहना, बहन, जगह) and यह/वह are
    ///   `[jeː]`/`[ʋoː]` — Legacy wrote `ə` in 39% of corpus rows. Applied
    ///   uniformly: it is near-categorical in the native and function words
    ///   that carry most of those tokens, while careful readings of Sanskrit
    ///   compounds (आग्रह, असहयोग) may keep `[ə]`; per-clip realization is
    ///   an acoustic-narrowing question, not a G2P one;
    /// * word-final short ɪ/ʊ are `iː`/`uː` (no length contrast there);
    /// * anusvara before a velar is `ŋ` (संकट), as before other stops it is
    ///   already homorganic — Legacy nasalized the vowel before क/ख only;
    /// * ज्ञ is `[ɡj]` (ज्ञान), not `d͡ʒɲ`;
    /// * a schwa deletion that would create an unpronounceable consonant run
    ///   (दुश्मनों → `ʃmn`) is undone;
    /// * digits and Latin script are an error rather than silently missing
    ///   from the labels while present in the audio.
    Current,
}

/// A syllable span within a word's phoneme list (`[start, end)`), with its
/// mora weight and stress from Roy's rules.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Syllable {
    pub start: usize,
    pub end: usize,
    pub nucleus: usize,
    pub moras: u8,
    pub stressed: bool,
}

/// One Devanagari word's labels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Word {
    pub phonemes: Vec<String>,
    pub stress: Vec<Stress>,
    pub syllables: Vec<Syllable>,
    /// One entry per orthographic schwa: whether the classifier kept it.
    pub schwa_retained: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Transliteration tables (transliterate.py)
// ---------------------------------------------------------------------------

type Unit = &'static str;

const VIRAMA: char = '\u{094D}';
const NUKTA: char = '\u{093C}';
const UNK: &str = "🆒";

fn consonant(c: char) -> Option<Unit> {
    Some(match c {
        'क' => "k",
        'ख' => "kh",
        'ग' => "g",
        'घ' => "gh",
        'ङ' => "ng",
        'च' => "c",
        'छ' => "ch",
        'ज' => "j",
        'झ' => "jh",
        'ञ' => "?",
        'ट' => "tt",
        'ठ' => "tth",
        'ड' => "dd",
        'ढ' => "ddh",
        'ण' => "n",
        'त' => "t",
        'थ' => "th",
        'द' => "d",
        'ध' => "dh",
        'न' => "n",
        'प' => "p",
        'फ' => "ph",
        'ब' => "b",
        'भ' => "bh",
        'म' => "m",
        'य' => "y",
        'र' => "r",
        'ल' => "l",
        'व' => "v",
        'श' => "sh",
        'ष' => "sh",
        'स' => "s",
        'ह' => "h",
        // Precomposed nukta letters (U+0958..U+095E).
        '\u{0958}' => "q",
        '\u{0959}' => "x",
        '\u{095A}' => "Gh",
        '\u{095B}' => "z",
        '\u{095C}' => "rr",
        '\u{095D}' => "rrh",
        '\u{095E}' => "f",
        _ => return None,
    })
}

fn vowel_sign(c: char) -> Option<Unit> {
    Some(match c {
        'अ' => "a",
        'आ' => "aa",
        'इ' => "i",
        'ई' => "ii",
        'उ' => "u",
        'ऊ' => "uu",
        'ए' => "e",
        'ऐ' | 'ऍ' => "E",
        'ओ' => "o",
        'औ' | 'ऑ' => "O",
        'ँ' => "~",
        'ं' => "ng",
        _ => return None,
    })
}

fn matra(c: char) -> Option<Unit> {
    Some(match c {
        'ा' => "aa",
        'ि' => "i",
        'ी' => "ii",
        'ु' => "u",
        'ू' => "uu",
        'े' => "e",
        'ै' | 'ॅ' => "E",
        'ो' => "o",
        'ौ' | 'ॉ' => "O",
        _ => return None,
    })
}

fn nukta(unit: Unit) -> Unit {
    match unit {
        "k" => "q",
        "kh" => "x",
        "g" => "Gh",
        "ph" => "f",
        "j" => "z",
        "jh" => "Zh",
        "dd" => "rr",
        "ddh" => "rrh",
        other => other,
    }
}

/// How anusvara assimilates to the following unit. `None` is the Python
/// `KeyError` (anusvara before a vowel or another mark), which failed the
/// whole utterance there and does here too.
fn nasal_assimilation(next: Unit, canon: Canon) -> Option<Unit> {
    Some(match next {
        // Legacy nasalized the vowel before क/ख but wrote ŋ before ग/घ; the
        // nasal is homorganic before every velar stop (संकट [səŋkəʈ]).
        "k" | "kh" => match canon {
            Canon::Legacy => "~",
            Canon::Current => "ng",
        },
        "g" | "gh" | "ng" | "Gh" => "ng",
        "c" | "ch" | "j" | "n" | "tt" | "tth" | "dd" | "ddh" | "t" | "th" | "d" | "dh" | "sh"
        | "s" => "n",
        "p" | "ph" | "b" | "bh" | "m" => "m",
        // `jh` appears twice in the Python table; the later `~` wins.
        "jh" | "y" | "r" | "l" | "v" | "h" | "q" | "x" | "f" | "z" | "rr" | "rrh" => "~",
        _ => return None,
    })
}

/// Devanagari word → Google-scheme units with every inherent schwa present.
fn transliterate(word: &str, canon: Canon) -> Result<Vec<Unit>, Error> {
    let mut text = word.replace('ऋ', "रि").replace('ृ', "्रि");
    if canon == Canon::Current {
        // ज्ञ is pronounced [ɡj] in Hindi (ज्ञान = gyaan), not [d͡ʒɲ].
        text = text.replace("ज्ञ", "ग्य");
    }
    let mut res: Vec<Unit> = Vec::new();
    for c in text.chars() {
        if c == VIRAMA {
            res.pop();
        } else if c == NUKTA {
            // The nukta modifies the consonant before it; if that consonant
            // still carries its schwa, lift the schwa off and put it back.
            if res.last() == Some(&"a") {
                res.pop();
                if let Some(l) = res.pop() {
                    res.push(nukta(l));
                }
                res.push("a");
            } else if let Some(l) = res.pop() {
                res.push(nukta(l));
            }
        } else if let Some(u) = consonant(c) {
            res.push(u);
            res.push("a");
        } else if let Some(u) = vowel_sign(c) {
            res.push(u);
        } else if let Some(u) = matra(c) {
            res.pop();
            res.push(u);
        }
        // Anything else in the block (danda, digits, avagraha…) is ignored.
    }
    for i in 0..res.len() {
        if res[i] == "ng" {
            res[i] = match res.get(i + 1) {
                None => "~",
                Some(next) => nasal_assimilation(next, canon).ok_or_else(|| {
                    Error::Unlabelable(format!("anusvara before {next:?} in {word:?}"))
                })?,
            };
        }
    }
    Ok(res)
}

// ---------------------------------------------------------------------------
// Loanword फ (lexide `_HINDI_NATIVE_PH_PREFIXES`)
// ---------------------------------------------------------------------------

/// Words whose nukta-less फ is a native aspirated stop /pʰ/. Everything else
/// defaults to /f/: a 2026-08-24 listening audit found the Perso-Arabic and
/// English loans that make up most फ tokens (काफी, फिल्म, सिर्फ, फोन) are
/// categorically [f], while careful orthography's nukta is mostly dropped.
const NATIVE_PH_PREFIXES: &[&str] = &[
    "फिर",
    "फल",
    "फूल",
    "फैल",
    "फेंक",
    "फंस",
    "फँस",
    "फूट",
    "फट",
    "फोड़",
    "फाड़",
    "फाडऩ",
    "फिसल",
    "फुसफुस",
    "फुँफ",
    "फफ",
    "फांसी",
    "फाटक",
    "फीका",
    "फीकी",
    "फीत",
    "फेर",
    "फुफेर",
    "दुफेर",
    "फगवाड़",
    "फुलझड़",
    "फूँक",
    "फुस",
    "फूस",
    "फुहार",
    "फव्वार",
    "फलांग",
    "फलद",
];
const NATIVE_PH_SUBSTRINGS: &[&str] = &[
    "सफल",
    "विफल",
    "क्षेत्रफल",
    "स्फीति",
    "स्फोट",
    "हेरफेर",
    "हेराफेर",
    "फटाफट",
    "दोफहर",
];

fn ph_is_native(word: &str) -> bool {
    NATIVE_PH_PREFIXES.iter().any(|p| word.starts_with(p))
        || NATIVE_PH_SUBSTRINGS.iter().any(|s| word.contains(s))
}

// ---------------------------------------------------------------------------
// Schwa classifier
// ---------------------------------------------------------------------------

/// Unit → indices (into `PHONS`) of the features it carries.
fn feature_index() -> &'static HashMap<&'static str, Vec<usize>> {
    static INDEX: OnceLock<HashMap<&'static str, Vec<usize>>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let phon_pos: HashMap<&str, usize> =
            PHONS.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        FEATURES
            .iter()
            .map(|(unit, feats)| {
                (
                    *unit,
                    feats
                        .iter()
                        .filter_map(|f| phon_pos.get(f).copied())
                        .collect(),
                )
            })
            .collect()
    })
}

/// sklearn `LogisticRegression.predict`: retained iff the decision function
/// is positive. Features are, for each of the ten context positions, one
/// indicator per phonological feature of the unit there (all zero past the
/// word's ends).
fn schwa_retained(units: &[Unit], index: usize) -> bool {
    let index_of = feature_index();
    let unk = &index_of[UNK];
    let mut score = INTERCEPT;
    // Slots 0..5 are units i-5..i-1 (in that order), 5..10 are i+1..i+5;
    // out-of-range slots contribute nothing but keep their column.
    let before = (0..5).map(|k| {
        let p = index as isize - 5 + k as isize;
        (p >= 0).then_some(p as usize)
    });
    let after = (index + 1..index + 6).map(|p| (p < units.len()).then_some(p));
    for (slot, pos) in before.chain(after).enumerate() {
        let Some(pos) = pos else { continue };
        let feats = index_of.get(units[pos]).unwrap_or(unk);
        for &f in feats {
            score += COEF[slot * PHONS.len() + f];
        }
    }
    score > 0.0
}

// ---------------------------------------------------------------------------
// Units → IPA
// ---------------------------------------------------------------------------

fn unit_ipa(unit: Unit) -> Option<&'static str> {
    Some(match unit {
        "a" => "ə",
        "aa" => "aː",
        "i" => "ɪ",
        "ii" => "iː",
        "u" => "ʊ",
        "uu" => "uː",
        "e" => "eː",
        "E" => "ɛː",
        "o" => "oː",
        "O" => "ɔː",
        "k" => "k",
        "kh" => "kʰ",
        "g" => "ɡ",
        "gh" => "ɡʱ",
        "ng" => "ŋ",
        "c" => "t͡ʃ",
        "ch" => "t͡ʃʰ",
        "j" => "d͡ʒ",
        "jh" => "d͡ʒʱ",
        "tt" => "ʈ",
        "tth" => "ʈʰ",
        "dd" => "ɖ",
        "ddh" => "ɖʱ",
        "t" => "t̪",
        "th" => "t̪ʰ",
        "d" => "d̪",
        "dh" => "d̪ʱ",
        "n" => "n",
        "p" => "p",
        "ph" => "pʰ",
        "b" => "b",
        "bh" => "bʱ",
        "m" => "m",
        "y" => "j",
        "r" => "ɾ",
        "l" => "l",
        "v" => "ʋ",
        "sh" => "ʃ",
        "s" => "s",
        "h" => "ɦ",
        "z" => "z",
        "f" => "f",
        "rr" => "ɽ",
        "rrh" => "ɽʱ",
        "q" => "q",
        "x" => "x",
        "Gh" => "ɣ",
        "Zh" => "ʒ",
        "?" => "ɲ",
        _ => return None,
    })
}

const NASAL_TILDE: char = '\u{0303}';

fn is_vowel_token(tok: &str) -> bool {
    tok.chars().next().is_some_and(|c| "əɪiʊueɛaoɔ".contains(c))
}

fn is_vowel_unit(u: Unit) -> bool {
    matches!(
        u,
        "a" | "aa" | "i" | "ii" | "u" | "uu" | "e" | "E" | "o" | "O"
    )
}

// ---------------------------------------------------------------------------
// Current-canon corrections
// ---------------------------------------------------------------------------

/// Undo schwa deletions that leave a consonant run no Hindi syllable
/// structure can carry. A run of three or more consonants between nuclei must
/// split into a coda of at most one consonant plus a legal onset: a single
/// consonant, consonant + `r`/`y`/`l`/`v`, `s` + stop, or `s` + stop +
/// liquid/glide. Runs that come purely from orthographic conjuncts (no
/// deleted schwa inside them) are never touched.
fn restore_illegal_deletions(units: &[Unit], retained: &mut [bool]) {
    fn is_stop(u: Unit) -> bool {
        matches!(
            u,
            "k" | "kh"
                | "g"
                | "gh"
                | "c"
                | "ch"
                | "j"
                | "jh"
                | "tt"
                | "tth"
                | "dd"
                | "ddh"
                | "t"
                | "th"
                | "d"
                | "dh"
                | "p"
                | "ph"
                | "b"
                | "bh"
                | "q"
        )
    }
    fn legal_onset(o: &[Unit]) -> bool {
        match o {
            [] | [_] => true,
            [_, "r" | "y" | "l" | "v"] => true,
            ["s", s] if is_stop(s) => true,
            ["s", s, "r" | "y" | "l" | "v"] if is_stop(s) => true,
            _ => false,
        }
    }
    fn legal_run(run: &[Unit]) -> bool {
        run.len() < 3 || (0..=1).any(|coda| legal_onset(&run[coda..]))
    }
    /// A consonant run of the post-deletion stream with the deleted schwa
    /// sites inside it as `(schwa index, offset within the run)`.
    type Run = (Vec<Unit>, Vec<(usize, usize)>);
    fn runs(units: &[Unit], retained: &[bool]) -> Vec<Run> {
        let mut out = Vec::new();
        let mut run: Vec<Unit> = Vec::new();
        let mut sites: Vec<(usize, usize)> = Vec::new();
        let mut schwa_i = 0usize;
        for &u in units {
            if u == "a" {
                let kept = retained[schwa_i];
                if !kept {
                    sites.push((schwa_i, run.len()));
                }
                schwa_i += 1;
                if !kept {
                    continue;
                }
            }
            if is_vowel_unit(u) || u == "~" {
                out.push((std::mem::take(&mut run), std::mem::take(&mut sites)));
            } else {
                run.push(u);
            }
        }
        out.push((run, sites));
        out
    }
    loop {
        let offender = runs(units, retained)
            .into_iter()
            .find(|(run, sites)| !legal_run(run) && !sites.is_empty());
        let Some((run, sites)) = offender else { return };
        // Restore the site that splits the run into two legal halves, or the
        // first site if none does.
        let (site, _) = sites
            .iter()
            .copied()
            .find(|&(_, off)| legal_run(&run[..off]) && legal_run(&run[off..]))
            .unwrap_or(sites[0]);
        retained[site] = true;
    }
}

/// `/ə/` next to `/ɦ/` surfaces as `[ɛ]` in Standard Hindi when the `ɦ`
/// closes the syllable or sits between two schwas: शहर [ʃɛɦɛr], कहना
/// [kɛɦnaː], बहन [bɛɦɛn], जगह [d͡ʒəɡɛɦ]. It stays `[ə]` before an `ɦ` that
/// is followed by a full vowel: पहाड़ [pəɦaːɽ], कहानी [kəɦaːniː]. Written as
/// `ɛː`, the chain's only open-mid front vowel.
fn raise_schwa_beside_h(tokens: &mut [String]) {
    let is_h = |t: &str| t == "ɦ";
    let is_schwa = |t: &str| t == "ə" || t == "ə̃";
    let n = tokens.len();
    for i in 0..n {
        if !is_schwa(&tokens[i]) {
            continue;
        }
        let before_closing_h = i + 1 < n
            && is_h(&tokens[i + 1])
            && (i + 2 >= n || !is_vowel_token(&tokens[i + 2]) || is_schwa(&tokens[i + 2]));
        let after_h_between_schwas =
            i >= 2 && is_h(&tokens[i - 1]) && tokens[i - 2].starts_with("ɛː");
        if before_closing_h || after_h_between_schwas {
            let nasal = tokens[i].contains(NASAL_TILDE);
            tokens[i] = if nasal {
                "ɛː̃".to_string()
            } else {
                "ɛː".to_string()
            };
        }
    }
}

/// Word-final short /ɪ/ and /ʊ/ have no short/long contrast in Hindi and
/// surface tense, as [i]/[u] (पति [pət̪i], वस्तु [ʋəst̪u]). Both reference
/// lists transcribe them long; the chain wrote the orthographic short vowel.
fn neutralize_final_high_vowels(tokens: &mut [String]) {
    if let Some(last) = tokens.last_mut() {
        match last.as_str() {
            "ɪ" => *last = "iː".to_string(),
            "ʊ" => *last = "uː".to_string(),
            "ɪ̃" => *last = "iː̃".to_string(),
            "ʊ̃" => *last = "uː̃".to_string(),
            _ => {}
        }
    }
}

/// The two pronouns whose spelling and sound have parted ways entirely.
fn special_word(word: &str) -> Option<Vec<&'static str>> {
    match word {
        "यह" => Some(vec!["j", "eː"]),
        "वह" => Some(vec!["ʋ", "oː"]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Stress (Roy 2017 surface rules, lexide `_hindi_surface_stress`)
// ---------------------------------------------------------------------------

const VOWELS: &[&str] = &["ə", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː", "oː", "ɔː", "aː"];

fn oral(tok: &str) -> String {
    tok.replace(NASAL_TILDE, "")
}

fn assign_stress(phones: &[String]) -> (Vec<Stress>, Vec<Syllable>) {
    let nuclei: Vec<usize> = phones
        .iter()
        .enumerate()
        .filter(|(_, p)| VOWELS.contains(&oral(p).as_str()))
        .map(|(i, _)| i)
        .collect();
    let mut stress = vec![Stress::None; phones.len()];
    if nuclei.is_empty() {
        return (stress, Vec::new());
    }
    let mut starts = vec![0usize];
    for w in nuclei.windows(2) {
        let (left, right) = (w[0], w[1]);
        let cluster = &phones[left + 1..right];
        let onset_len = if cluster.len() <= 1 {
            cluster.len()
        } else if cluster.len() == 2 && matches!(cluster[1].as_str(), "j" | "ɾ" | "l" | "ʋ") {
            2
        } else {
            cluster.len() - 1
        };
        starts.push(right - onset_len);
    }
    let mut ends = starts[1..].to_vec();
    ends.push(phones.len());
    let mut syllables = Vec::with_capacity(nuclei.len());
    for (index, ((&start, &end), &nucleus)) in starts.iter().zip(&ends).zip(&nuclei).enumerate() {
        let coda = end.saturating_sub(nucleus + 1);
        let long = oral(&phones[nucleus]).contains('ː');
        let weight = (if long { 2 } else { 1 }) + coda;
        let last = index == nuclei.len() - 1;
        let stressed = weight >= 3
            || (weight == 2 && !last)
            || (weight == 1 && nuclei.len() == 2 && index == 0);
        if stressed {
            stress[nucleus] = Stress::Primary;
        }
        syllables.push(Syllable {
            start,
            end,
            nucleus,
            moras: weight as u8,
            stressed,
        });
    }
    (stress, syllables)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn in_devanagari_block(c: char) -> bool {
    ('\u{0900}'..='\u{097F}').contains(&c)
}

/// Label one Devanagari word.
pub fn word(word: &str, canon: Canon) -> Result<Word, Error> {
    if canon == Canon::Current
        && let Some(tokens) = special_word(word)
    {
        let phonemes: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        let (stress, syllables) = assign_stress(&phonemes);
        return Ok(Word {
            phonemes,
            stress,
            syllables,
            schwa_retained: Vec::new(),
        });
    }
    let mut units = transliterate(word, canon)?;
    if units.contains(&"ph") && !ph_is_native(word) {
        for u in &mut units {
            if *u == "ph" {
                *u = "f";
            }
        }
    }
    let mut retained: Vec<bool> = units
        .iter()
        .enumerate()
        .filter(|(_, u)| **u == "a")
        .map(|(i, _)| schwa_retained(&units, i))
        .collect();
    if canon == Canon::Current {
        restore_illegal_deletions(&units, &mut retained);
    }
    let mut phonemes: Vec<String> = Vec::with_capacity(units.len());
    let mut schwa_i = 0;
    for &u in &units {
        if u == "a" {
            let keep = retained[schwa_i];
            schwa_i += 1;
            if !keep {
                continue;
            }
        }
        if u == "~" {
            // Nasalization attaches to a preceding vowel; a stray mark after
            // a consonant (or a doubled one) carries no contrast and is
            // dropped.
            if let Some(last) = phonemes.last_mut()
                && is_vowel_token(last)
                && !last.contains(NASAL_TILDE)
            {
                last.push(NASAL_TILDE);
            }
        } else if let Some(ipa) = unit_ipa(u) {
            phonemes.push(ipa.to_string());
        }
    }
    if canon == Canon::Current {
        raise_schwa_beside_h(&mut phonemes);
        neutralize_final_high_vowels(&mut phonemes);
    }
    let (stress, syllables) = assign_stress(&phonemes);
    Ok(Word {
        phonemes,
        stress,
        syllables,
        schwa_retained: retained,
    })
}

/// Label every Devanagari word in `text` (maximal runs of the Devanagari
/// block, as lexide splits them). Words that produce no phonemes are dropped.
///
/// Under [`Canon::Current`], digits or Latin letters in the text are an
/// [`Error::Unlabelable`]: they are spoken in the audio but this chain cannot
/// phonemize them, so labels would silently be missing a stretch of speech.
/// Legacy drops them without a word, as lexide's Python did.
pub fn phonemize(text: &str, canon: Canon) -> Result<Vec<Word>, Error> {
    if canon == Canon::Current {
        let digits: String = text
            .chars()
            .filter(|c| c.is_ascii_digit() || ('०'..='९').contains(c))
            .collect();
        if !digits.is_empty() {
            return Err(Error::Unlabelable(format!("hindi_digits:{digits}")));
        }
        let latin: Vec<&str> = text
            .split(|c: char| !c.is_ascii_alphabetic())
            .filter(|w| !w.is_empty())
            .collect();
        if !latin.is_empty() {
            return Err(Error::Unlabelable(format!(
                "hindi_latin_script:{}",
                latin.join(",")
            )));
        }
    }
    let mut words = Vec::new();
    for run in text
        .split(|c: char| !in_devanagari_block(c))
        .filter(|w| !w.is_empty())
    {
        let w = word(run, canon)?;
        if !w.phonemes.is_empty() {
            words.push(w);
        }
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ipa(text: &str, canon: Canon) -> String {
        phonemize(text, canon)
            .unwrap()
            .iter()
            .map(|w| w.phonemes.join(" "))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    #[test]
    fn legacy_matches_the_python_chain_on_known_words() {
        // Values taken from running lexide's `_schwa_stress_hin` directly.
        assert_eq!(ipa("यह", Canon::Legacy), "j ə ɦ");
        assert_eq!(ipa("संकट", Canon::Legacy), "s ə̃ k ə ʈ");
        assert_eq!(ipa("ज्ञान", Canon::Legacy), "d͡ʒ ɲ aː n");
        assert_eq!(ipa("दुश्मनों", Canon::Legacy), "d̪ ʊ ʃ m n oː̃");
        assert_eq!(ipa("ज़्यादा", Canon::Legacy), "z j aː d̪ aː");
        assert_eq!(ipa("सच्ची", Canon::Legacy), "s ə t͡ʃ t͡ʃ iː");
        assert_eq!(ipa("उन्नीस", Canon::Legacy), "ʊ n n iː s");
        assert_eq!(
            ipa("Avant, je n'aimais pas les épinards.", Canon::Legacy),
            ""
        );
    }

    #[test]
    fn loan_ph_is_f_unless_native() {
        assert_eq!(ipa("फ़ोन", Canon::Legacy), "f oː n");
        assert_eq!(ipa("फोन", Canon::Legacy), "f oː n");
        assert_eq!(ipa("फल", Canon::Legacy), "pʰ ə l");
    }

    #[test]
    fn current_raises_schwa_beside_h() {
        assert_eq!(ipa("शहर", Canon::Current), "ʃ ɛː ɦ ɛː ɾ");
        assert_eq!(ipa("कहना", Canon::Current), "k ɛː ɦ n aː");
        assert_eq!(ipa("बहन", Canon::Current), "b ɛː ɦ ɛː n");
        assert_eq!(ipa("जगह", Canon::Current), "d͡ʒ ə ɡ ɛː ɦ");
        // Not before a full vowel.
        assert_eq!(ipa("पहाड़", Canon::Current), "p ə ɦ aː ɽ");
        assert_eq!(ipa("कहानी", Canon::Current), "k ə ɦ aː n iː");
        assert_eq!(ipa("यह वह", Canon::Current), "j eː | ʋ oː");
    }

    #[test]
    fn current_neutralizes_final_high_vowels() {
        assert_eq!(ipa("पति", Canon::Current), "p ə t̪ iː");
        assert_eq!(ipa("वस्तु", Canon::Current), "ʋ ə s t̪ uː");
        assert_eq!(ipa("पति", Canon::Legacy), "p ə t̪ ɪ");
        // Only word-finally.
        assert_eq!(ipa("किताब", Canon::Current), "k ɪ t̪ aː b");
    }

    #[test]
    fn current_fixes_velar_nasal_and_jn() {
        assert_eq!(ipa("संकट", Canon::Current), "s ə ŋ k ə ʈ");
        assert_eq!(ipa("ज्ञान", Canon::Current), "ɡ j aː n");
    }

    #[test]
    fn current_restores_impossible_deletions() {
        assert_eq!(ipa("दुश्मनों", Canon::Current), "d̪ ʊ ʃ m ə n oː̃");
        // Conjunct runs are left alone.
        assert_eq!(ipa("संस्कृत", Canon::Current), "s ə n s k ɾ ɪ t̪");
    }

    #[test]
    fn current_refuses_digits_and_latin() {
        assert!(matches!(
            phonemize("19 वीं शताब्दी", Canon::Current),
            Err(Error::Unlabelable(r)) if r.starts_with("hindi_digits")
        ));
        assert!(matches!(
            phonemize("AOL अपनी", Canon::Current),
            Err(Error::Unlabelable(r)) if r.starts_with("hindi_latin")
        ));
        assert!(phonemize("19 वीं", Canon::Legacy).is_ok());
    }

    #[test]
    fn stress_follows_roy_rules() {
        // हिन्दुस्तान: ɦ ɪ n d̪ ʊ s t̪ aː n — superheavy final syllable stressed.
        let w = &phonemize("हिन्दुस्तान", Canon::Legacy).unwrap()[0];
        let last = w.syllables.last().unwrap();
        assert!(last.stressed && last.moras >= 3, "{w:?}");
        assert_eq!(w.stress.len(), w.phonemes.len());
        assert_eq!(w.syllables[0].start, 0);
        assert_eq!(last.end, w.phonemes.len());
    }
}
