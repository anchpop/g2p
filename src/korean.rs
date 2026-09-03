//! Korean grapheme-to-phoneme: g2pk2, run as a pinned Python process.
//!
//! espeak's `ko` voice matches Wiktionary on 47% of words at the phonemic
//! level: it has no tense (fortis) consonants at all — 달/딸/탈 collapse
//! together — splits affricates and aspirates into letter sequences, and
//! applies none of the sound changes Korean orthography leaves implicit
//! (nasalization, ㄴ-insertion, ㄹㄹ). Korean G2P is a settled problem: the
//! sound changes are codified in the 표준 발음법 and every published system
//! is a rule table over them. g2pk (Kyubyong Park; g2pk2 is its maintained
//! fork, the one Korean TTS stacks and Montreal Forced Aligner use) matches
//! Wiktionary on 95.6% of words, independently of Wiktionary's own module,
//! and is the only one that also gets ㄴ-insertion (꽃잎 [꼰닙]) and the
//! morphologically conditioned tensification (넘다 [넘따], 할 것 [할껏]),
//! because it tags the sentence with mecab-ko first. The remaining ~4% is
//! lexical — Sino-Korean compounds whose ㄹ tensifies the next consonant
//! unpredictably (결점 [결쩜]) — and needs a dictionary, not rules.
//!
//! Not ported: the rule table and the mecab dictionary are large and the
//! wrapper is small. `python/korean/` is a `uv` project pinning `g2pk2`,
//! the prebuilt `mecab-ko` wheel, and `mecab-ko-dic`; the crate embeds it,
//! unpacks it beside the espeak data on first use, and drives
//! `g2p_korean.py` as a JSON-lines server. Needs `uv` on PATH; the first
//! call resolves the environment (network unless uv's cache is warm). The
//! server returns each word's pronunciation as post-sandhi Hangul (값이 →
//! 갑씨); this module maps that to phones.
//!
//! The labels target real, connected speech. Within a clause the sound
//! changes apply across the spaces between 어절 exactly as they do inside a
//! word — 못 만났어 [몬만나써], 부엌 좀 [부억쫌], 할 것 [할껏] — because a
//! speaker does not pause there and that is what the audio contains.
//! Punctuation is where speakers pause, so the text is split into clauses
//! at every punctuation mark and each clause is phonemized on its own:
//! 안녕, 라디오 stays [안녕 라디오] instead of g2pk's whole-string [나디오].
//!
//! Label set (phonemic; what the model is trained to hear): lenis stops and
//! affricate `k t p tɕ`, aspirated `kʰ tʰ pʰ tɕʰ`, tense `k͈ t͈ p͈ tɕ͈ s͈`,
//! `s h m n ŋ`, ㄹ as `ɾ` between vowels and `l` in the coda and geminate
//! (실라 = s i l l a), vowels `a e ʌ o u ɯ i` and glides `j w`. ㅐ/ㅔ (and
//! ㅙ/ㅚ/ㅞ) are one label: the merger is complete in the speech the model
//! is trained and scored on, and a label the audio cannot support is noise.
//! Lenis obstruents are not voiced between sonorants in the label — the
//! allophony is fully predictable and the same phone label covers both.
//! Word-initial 의 is `ɯ i`. Korean has no lexical stress or tone.
//!
//! Digits, Latin, hanja, and bare jamo are refused (`Error::Unlabelable`)
//! rather than passed through g2pk's number and English readings, which are
//! guesses the audio need not contain.

use crate::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

const PYPROJECT: &str = include_str!("../python/korean/pyproject.toml");
const LOCK: &str = include_str!("../python/korean/uv.lock");
const SERVER: &str = include_str!("../python/korean/g2p_korean.py");
const MECAB_SHIM: &str = include_str!("../python/korean/mecab.py");

/// Digest of the embedded Python project (pins included): part of
/// [`crate::identity`].
pub const KOREAN_DIGEST: &str = env!("G2P_KOREAN_DIGEST");

/// One utterance's labels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Labels {
    /// The pronunciation as post-sandhi Hangul, one token per input word,
    /// clauses separated by ` | ` (값이 안 좋아, 라디오 → "갑씨 안 조아 |
    /// 라디오"). Readable; not for scoring.
    pub raw: String,
    pub phonemes: Vec<String>,
    /// All `None`: Korean has no lexical stress.
    pub stress: Vec<crate::Stress>,
    /// `[start, end)` per input word (어절).
    pub word_spans: Vec<(usize, usize)>,
}

struct Server {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

static SERVER_STATE: Mutex<Option<Result<Server, String>>> = Mutex::new(None);

fn project_dir() -> Result<std::path::PathBuf, Error> {
    // Keyed on this project's own digest: the cache root is keyed on the
    // espeak build, and a changed server script or lockfile must not reuse
    // an unpack made from the old one.
    let base = crate::data::cache_root()?.join(format!("python-korean-{KOREAN_DIGEST}"));
    let marker = base.join(".unpacked");
    if !marker.exists() {
        std::fs::create_dir_all(&base)?;
        std::fs::write(base.join("pyproject.toml"), PYPROJECT)?;
        std::fs::write(base.join("uv.lock"), LOCK)?;
        std::fs::write(base.join("g2p_korean.py"), SERVER)?;
        std::fs::write(base.join("mecab.py"), MECAB_SHIM)?;
        std::fs::write(&marker, b"")?;
    }
    Ok(base)
}

fn spawn() -> Result<Server, String> {
    let dir = project_dir().map_err(|e| e.to_string())?;
    let mut child = Command::new("uv")
        .args(["run", "--project"])
        .arg(&dir)
        .args(["--locked", "python", "g2p_korean.py"])
        .current_dir(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            format!(
                "could not start `uv` for the Korean backend ({e}); install uv \
                 (https://docs.astral.sh/uv/) and make sure it is on PATH"
            )
        })?;
    let stdin = child.stdin.take().ok_or("korean server stdin")?;
    let stdout = BufReader::new(child.stdout.take().ok_or("korean server stdout")?);
    Ok(Server {
        _child: child,
        stdin,
        stdout,
    })
}

/// Each word's pronunciation as post-sandhi Hangul, clause by clause, via
/// the server.
pub fn pronunciations(clauses: &[Vec<String>]) -> Result<Vec<Vec<String>>, Error> {
    let mut guard = SERVER_STATE.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(spawn());
    }
    let server = match guard.as_mut().unwrap() {
        Ok(s) => s,
        Err(e) => return Err(Error::Backend(format!("korean: {e}"))),
    };
    let request = serde_json::json!({ "clauses": clauses }).to_string();
    let io_err = |e: std::io::Error| Error::Synth(format!("korean backend I/O: {e}"));
    server.stdin.write_all(request.as_bytes()).map_err(io_err)?;
    server.stdin.write_all(b"\n").map_err(io_err)?;
    server.stdin.flush().map_err(io_err)?;
    let mut line = String::new();
    let n = server.stdout.read_line(&mut line).map_err(io_err)?;
    if n == 0 {
        // The server died; drop it so the next call respawns.
        *guard = None;
        return Err(Error::Synth("korean backend exited".into()));
    }
    let reply: serde_json::Value = serde_json::from_str(&line)
        .map_err(|e| Error::Synth(format!("korean backend reply: {e}")))?;
    if let Some(err) = reply.get("error").and_then(|v| v.as_str()) {
        return Err(Error::Synth(format!("g2pk2: {err}")));
    }
    let prons: Vec<Vec<String>> = reply
        .get("prons")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or_else(|| Error::Synth("korean backend reply lacks prons".into()))?;
    let same_shape =
        prons.len() == clauses.len() && prons.iter().zip(clauses).all(|(p, c)| p.len() == c.len());
    if !same_shape {
        return Err(Error::Synth(format!(
            "korean backend reply shape {:?} does not match the request {:?}",
            prons.iter().map(Vec::len).collect::<Vec<_>>(),
            clauses.iter().map(Vec::len).collect::<Vec<_>>()
        )));
    }
    Ok(prons)
}

// ---------------------------------------------------------------------------
// Text → clauses of words
// ---------------------------------------------------------------------------

fn is_hangul_syllable(c: char) -> bool {
    ('\u{AC00}'..='\u{D7A3}').contains(&c)
}

fn is_jamo(c: char) -> bool {
    ('\u{1100}'..='\u{11FF}').contains(&c)
        || ('\u{3130}'..='\u{318F}').contains(&c)
        || ('\u{A960}'..='\u{A97F}').contains(&c)
        || ('\u{D7B0}'..='\u{D7FF}').contains(&c)
}

/// The Hangul words of `text` grouped into clauses, or why the text is
/// refused. Any character that is neither a Hangul syllable nor whitespace
/// nor a letter or digit — punctuation, symbols, music notes — ends the
/// current clause and is otherwise silent. Every letter or digit that is
/// not a Hangul syllable is refused, since g2pk would read it as a guess.
pub fn clauses(text: &str) -> Result<Vec<Vec<String>>, Error> {
    let mut digits = String::new();
    let mut latin: Vec<String> = Vec::new();
    let mut other = String::new();
    let mut jamo = String::new();
    let mut clauses: Vec<Vec<String>> = Vec::new();
    let mut clause: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut latin_run = String::new();
    let end_word = |word: &mut String, clause: &mut Vec<String>| {
        if !word.is_empty() {
            clause.push(std::mem::take(word));
        }
    };
    for c in text.chars() {
        if !c.is_ascii_alphabetic() && !latin_run.is_empty() {
            latin.push(std::mem::take(&mut latin_run));
        }
        if is_hangul_syllable(c) {
            word.push(c);
        } else if c.is_whitespace() {
            end_word(&mut word, &mut clause);
        } else if c.is_ascii_digit() {
            digits.push(c);
        } else if c.is_ascii_alphabetic() {
            latin_run.push(c);
        } else if is_jamo(c) {
            jamo.push(c);
        } else if c.is_alphanumeric() {
            other.push(c);
        } else {
            end_word(&mut word, &mut clause);
            if !clause.is_empty() {
                clauses.push(std::mem::take(&mut clause));
            }
        }
    }
    if !latin_run.is_empty() {
        latin.push(latin_run);
    }
    end_word(&mut word, &mut clause);
    if !clause.is_empty() {
        clauses.push(clause);
    }
    if !digits.is_empty() {
        return Err(Error::Unlabelable(format!("korean_digits:{digits}")));
    }
    if !latin.is_empty() {
        return Err(Error::Unlabelable(format!(
            "korean_latin_script:{}",
            latin.join(",")
        )));
    }
    if !jamo.is_empty() {
        return Err(Error::Unlabelable(format!("korean_jamo:{jamo}")));
    }
    if !other.is_empty() {
        return Err(Error::Unlabelable(format!(
            "korean_unsupported_char:{other}"
        )));
    }
    Ok(clauses)
}

// ---------------------------------------------------------------------------
// Post-sandhi Hangul → phones
// ---------------------------------------------------------------------------

/// Onset consonant (choseong index) → phone; ㄹ is decided by context.
const ONSETS: [&str; 19] = [
    "k", "k͈", "n", "t", "t͈", "ɾ", "m", "p", "p͈", "s", "s͈", "", "tɕ", "tɕ͈", "tɕʰ", "kʰ", "tʰ", "pʰ",
    "h",
];
/// Vowel (jungseong index) → phones.
const VOWELS: [&[&str]; 21] = [
    &["a"],      // ㅏ
    &["e"],      // ㅐ
    &["j", "a"], // ㅑ
    &["j", "e"], // ㅒ
    &["ʌ"],      // ㅓ
    &["e"],      // ㅔ
    &["j", "ʌ"], // ㅕ
    &["j", "e"], // ㅖ
    &["o"],      // ㅗ
    &["w", "a"], // ㅘ
    &["w", "e"], // ㅙ
    &["w", "e"], // ㅚ
    &["j", "o"], // ㅛ
    &["u"],      // ㅜ
    &["w", "ʌ"], // ㅝ
    &["w", "e"], // ㅞ
    &["w", "i"], // ㅟ
    &["j", "u"], // ㅠ
    &["ɯ"],      // ㅡ
    &["ɯ", "i"], // ㅢ
    &["i"],      // ㅣ
];
/// Coda (jongseong index) → phone. After g2pk's sound changes only the seven
/// neutralized codas ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅇ remain; anything else is a backend
/// bug, surfaced as an error rather than guessed at.
fn coda(index: usize) -> Option<&'static str> {
    Some(match index {
        0 => "",
        1 => "k",
        4 => "n",
        7 => "t",
        8 => "l",
        16 => "m",
        17 => "p",
        21 => "ŋ",
        _ => return None,
    })
}

/// Phones of one word's post-sandhi Hangul. `after_l` says whether the
/// previous word in the clause ended in ㄹ, so a clause like 설날 written as
/// two words still gets the geminate lateral.
fn word_phones(pron: &str, mut after_l: bool) -> Result<(Vec<String>, bool), Error> {
    let mut out: Vec<String> = Vec::new();
    for c in pron.chars() {
        if !is_hangul_syllable(c) {
            return Err(Error::Synth(format!(
                "korean backend returned a non-syllable {c:?} in {pron:?}"
            )));
        }
        let code = c as u32 - 0xAC00;
        let (l, v, t) = (
            (code / 588) as usize,
            ((code % 588) / 28) as usize,
            (code % 28) as usize,
        );
        let onset = ONSETS[l];
        if !onset.is_empty() {
            // ㄹ after coda ㄹ is the geminate lateral, not a tap.
            out.push(if onset == "ɾ" && after_l { "l" } else { onset }.to_string());
        }
        out.extend(VOWELS[v].iter().map(|p| p.to_string()));
        let coda = coda(t).ok_or_else(|| {
            Error::Synth(format!(
                "korean backend left an unneutralized coda in {pron:?}"
            ))
        })?;
        if !coda.is_empty() {
            out.push(coda.to_string());
        }
        after_l = coda == "l";
    }
    Ok((out, after_l))
}

/// Labels from the clauses' pronunciations.
pub fn labels(prons: &[Vec<String>]) -> Result<Labels, Error> {
    let mut labels = Labels {
        raw: prons
            .iter()
            .map(|c| c.join(" "))
            .collect::<Vec<_>>()
            .join(" | "),
        ..Labels::default()
    };
    for clause in prons {
        let mut after_l = false;
        for pron in clause {
            let start = labels.phonemes.len();
            let (phones, ends_in_l) = word_phones(pron, after_l)?;
            after_l = ends_in_l;
            labels.phonemes.extend(phones);
            labels.word_spans.push((start, labels.phonemes.len()));
        }
    }
    labels.stress = vec![crate::Stress::None; labels.phonemes.len()];
    Ok(labels)
}

/// Label one utterance.
pub fn phonemize(text: &str) -> Result<Labels, Error> {
    let clauses = clauses(text)?;
    if clauses.is_empty() {
        return Ok(Labels::default());
    }
    labels(&pronunciations(&clauses)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split(' ').map(str::to_string).collect()
    }

    #[test]
    fn maps_post_sandhi_hangul_to_phones() {
        let l = labels(&[words("갑씨 실라"), words("꼰닙 의사")]).unwrap();
        assert_eq!(
            l.phonemes,
            [
                "k", "a", "p", "s͈", "i", "s", "i", "l", "l", "a", "k͈", "o", "n", "n", "i", "p",
                "ɯ", "i", "s", "a"
            ]
        );
        assert_eq!(l.word_spans, [(0, 5), (5, 10), (10, 16), (16, 20)]);
        assert!(l.stress.iter().all(|s| *s == crate::Stress::None));
        assert_eq!(l.raw, "갑씨 실라 | 꼰닙 의사");
    }

    #[test]
    fn tap_between_vowels_lateral_in_coda_and_across_words() {
        let phones = |s: &str| word_phones(s, false).unwrap().0;
        assert_eq!(phones("라디오"), ["ɾ", "a", "t", "i", "o"]);
        assert_eq!(phones("물"), ["m", "u", "l"]);
        assert_eq!(phones("설랄"), ["s", "ʌ", "l", "l", "a", "l"]);
        // 잘 래다 (from 잘 내다 in connected speech): the ㄹ that starts the
        // second word is the geminate's second half.
        let l = labels(&[words("잘 래다")]).unwrap();
        assert_eq!(l.phonemes, ["tɕ", "a", "l", "l", "e", "t", "a"]);
    }

    #[test]
    fn rejects_unneutralized_codas() {
        assert!(matches!(word_phones("값", false), Err(Error::Synth(_))));
    }

    #[test]
    fn splits_clauses_at_punctuation() {
        assert_eq!(
            clauses("“안녕, 라디오!” 뭐라고요?").unwrap(),
            [words("안녕"), words("라디오"), words("뭐라고요")]
        );
        assert_eq!(
            clauses("못 만났어... 부엌 좀 봐").unwrap(),
            [words("못 만났어"), words("부엌 좀 봐")]
        );
        assert!(clauses("...").unwrap().is_empty());
    }

    #[test]
    fn refuses_digits_latin_jamo_hanja() {
        for (text, reason) in [
            ("3개 주세요", "korean_digits:3"),
            ("mp3 파일", "korean_digits:3"),
            (
                "그 사람 좀 old school이야",
                "korean_latin_script:old,school",
            ),
            ("ㄴ 것 같다", "korean_jamo:ㄴ"),
            ("韓國 사람", "korean_unsupported_char:韓國"),
        ] {
            match clauses(text) {
                Err(Error::Unlabelable(r)) => assert_eq!(r, reason, "{text}"),
                other => panic!("{text}: {other:?}"),
            }
        }
    }

    #[test]
    #[ignore = "needs uv on PATH and network on first run"]
    fn runs_the_python_backend() {
        // Connected speech inside the clause (할 것 → 할 껏, 못 만났어 →
        // 몬 만나써); the comma blocks it (라디오 keeps its ㄹ).
        let l = phonemize("할 것을 못 만났어, 라디오!").unwrap();
        assert_eq!(l.raw, "할 꺼슬 몬 만나써 | 라디오");
    }
}
