//! Thai grapheme-to-phoneme: vachana-thai, run as a pinned Python process.
//!
//! espeak's `th` voice is not a G2P: it spells consonants letter by letter,
//! ignores the vowels Thai orthography leaves unwritten, and writes digits
//! for tones (0 of 3,000 Wiktionary words match). lexide labels Thai with
//! vachana-thai — the TLTK rule/lexicon front end over pythainlp's dictionary
//! word segmenter — which matches Wiktionary on 88% of words segmentally and
//! 87% on tone.
//!
//! This one is not ported. Its lexicon covers 98.6% of corpus tokens, but the
//! rest go through TLTK's probabilistic chart parser with trigram statistics,
//! and refusing those would drop 8% of Thai sentences. Instead the crate
//! embeds a `uv` project pinning `vachana-g2p` and `pythainlp`
//! (`python/thai/`, with its lockfile), unpacks it beside the espeak data on
//! first use, and drives `g2p_thai.py` as a JSON-lines server. Needs `uv` on
//! PATH; the first call resolves the environment (network). The label stage
//! — tokenizing vachana's IPA into the model's phone inventory, tone per
//! syllable, stress on each word's final syllable — is lexide's `thai_labels`,
//! ported here.

use crate::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

const PYPROJECT: &str = include_str!("../python/thai/pyproject.toml");
const LOCK: &str = include_str!("../python/thai/uv.lock");
const SERVER: &str = include_str!("../python/thai/g2p_thai.py");

/// Digest of the embedded Python project (pins included): part of
/// [`crate::identity`].
pub const THAI_DIGEST: &str = env!("G2P_THAI_DIGEST");

/// One utterance's labels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Labels {
    /// vachana's IPA string as returned.
    pub raw: String,
    pub phonemes: Vec<String>,
    pub stress: Vec<crate::Stress>,
    /// Tone class per phoneme: 1 mid, 2 low, 3 falling, 4 high, 5 rising on
    /// vowels (the tone-bearing phone of each syllable), `None` on consonants.
    pub tone: Vec<Option<u8>>,
    /// `[start, end)` per word (vachana's space-separated tokens).
    pub word_spans: Vec<(usize, usize)>,
}

struct Server {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

static SERVER_STATE: Mutex<Option<Result<Server, String>>> = Mutex::new(None);

fn project_dir() -> Result<std::path::PathBuf, Error> {
    let base = crate::data::cache_root()?.join("python-thai");
    let marker = base.join(".unpacked");
    if !marker.exists() {
        std::fs::create_dir_all(&base)?;
        std::fs::write(base.join("pyproject.toml"), PYPROJECT)?;
        std::fs::write(base.join("uv.lock"), LOCK)?;
        std::fs::write(base.join("g2p_thai.py"), SERVER)?;
        std::fs::write(&marker, b"")?;
    }
    Ok(base)
}

fn spawn() -> Result<Server, String> {
    let dir = project_dir().map_err(|e| e.to_string())?;
    let mut child = Command::new("uv")
        .args(["run", "--project"])
        .arg(&dir)
        .args(["--locked", "python", "g2p_thai.py"])
        .current_dir(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            format!(
                "could not start `uv` for the Thai backend ({e}); install uv \
                 (https://docs.astral.sh/uv/) and make sure it is on PATH"
            )
        })?;
    let stdin = child.stdin.take().ok_or("thai server stdin")?;
    let stdout = BufReader::new(child.stdout.take().ok_or("thai server stdout")?);
    Ok(Server {
        _child: child,
        stdin,
        stdout,
    })
}

/// vachana's IPA for `text`, via the server.
pub fn raw_ipa(text: &str) -> Result<String, Error> {
    let mut guard = SERVER_STATE.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(spawn());
    }
    let server = match guard.as_mut().unwrap() {
        Ok(s) => s,
        Err(e) => return Err(Error::Backend(format!("thai: {e}"))),
    };
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let request = serde_json::json!({ "text": flat }).to_string();
    let io_err = |e: std::io::Error| Error::Synth(format!("thai backend I/O: {e}"));
    server.stdin.write_all(request.as_bytes()).map_err(io_err)?;
    server.stdin.write_all(b"\n").map_err(io_err)?;
    server.stdin.flush().map_err(io_err)?;
    let mut line = String::new();
    let n = server.stdout.read_line(&mut line).map_err(io_err)?;
    if n == 0 {
        // The server died; drop it so the next call respawns.
        *guard = None;
        return Err(Error::Synth("thai backend exited".into()));
    }
    let reply: serde_json::Value = serde_json::from_str(&line)
        .map_err(|e| Error::Synth(format!("thai backend reply: {e}")))?;
    if let Some(err) = reply.get("error").and_then(|v| v.as_str()) {
        return Err(Error::Synth(format!("vachana: {err}")));
    }
    reply
        .get("ipa")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| Error::Synth("thai backend reply lacks ipa".into()))
}

// ---------------------------------------------------------------------------
// Label stage (lexide `thai_labels`)
// ---------------------------------------------------------------------------

/// The model's Thai phone inventory, longest first so multi-character phones
/// win the match.
const THAI_PHONES: &[&str] = &[
    "tɕʰ", "iə", "ɯə", "uə", "aː", "iː", "uː", "ɯː", "eː", "ɛː", "oː", "ɔː", "əː", "tɕ", "pʰ",
    "tʰ", "kʰ", "a", "i", "u", "ɯ", "e", "ɛ", "o", "ɔ", "ə", "b", "p", "m", "f", "d", "t", "n",
    "s", "r", "l", "k", "ŋ", "w", "j", "h", "ʔ",
];
const VOWEL_HEADS: &str = "aiuɯeɛoɔə";

fn tone_of(mark: char) -> Option<u8> {
    match mark {
        '\u{0300}' => Some(2), // grave: low
        '\u{0302}' => Some(3), // circumflex: falling
        '\u{0301}' => Some(4), // acute: high
        '\u{030C}' => Some(5), // caron: rising
        _ => None,
    }
}

fn is_word_break(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_digit()
        || c.is_ascii_punctuation()
        || matches!(c, '、' | '，' | '。')
}

/// Tokenize vachana's IPA into phones, tone per phone, stress on each word's
/// final vowel, and word spans.
pub fn parse(raw: &str) -> Result<Labels, Error> {
    use unicode_normalization::UnicodeNormalization;
    let text: Vec<char> = raw.nfd().collect();
    let mut labels = Labels {
        raw: raw.to_string(),
        ..Labels::default()
    };
    let mut word_start = 0usize;
    let mut word_vowels: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    let finish_word =
        |labels: &mut Labels, word_vowels: &mut Vec<usize>, word_start: &mut usize| {
            if let Some(&last) = word_vowels.last() {
                // Standard Thai lexical words carry primary stress on their final
                // syllable; vachana keeps its tokenizer's word boundaries as
                // spaces.
                labels.stress[last] = crate::Stress::Primary;
            }
            if labels.phonemes.len() > *word_start {
                labels.word_spans.push((*word_start, labels.phonemes.len()));
            }
            *word_start = labels.phonemes.len();
            word_vowels.clear();
        };
    while pos < text.len() {
        let ch = text[pos];
        if is_word_break(ch) {
            finish_word(&mut labels, &mut word_vowels, &mut word_start);
            pos += 1;
            continue;
        }
        let mut phone: Option<&str> = None;
        let mut embedded_tone: Option<u8> = None;
        // vachana writes the tone mark right after the first vowel character
        // (îː, ìə), i.e. inside a multi-character phone.
        for cand in THAI_PHONES {
            let cc: Vec<char> = cand.chars().collect();
            if !VOWEL_HEADS.contains(cc[0]) {
                continue;
            }
            if text[pos..].starts_with(&cc) {
                phone = Some(cand);
                break;
            }
            let mark_pos = pos + 1;
            if text[pos] == cc[0]
                && mark_pos < text.len()
                && tone_of(text[mark_pos]).is_some()
                && text[mark_pos + 1..].starts_with(&cc[1..])
            {
                phone = Some(cand);
                embedded_tone = tone_of(text[mark_pos]);
                break;
            }
        }
        if phone.is_none() {
            phone = THAI_PHONES
                .iter()
                .copied()
                .find(|p| text[pos..].starts_with(&p.chars().collect::<Vec<_>>()));
        }
        let Some(phone) = phone else {
            let tail: String = text[pos..].iter().take(20).collect();
            return Err(Error::Unlabelable(format!("thai_unparsed_ipa:{tail}")));
        };
        pos += phone.chars().count() + usize::from(embedded_tone.is_some());
        let is_vowel = VOWEL_HEADS.contains(phone.chars().next().unwrap());
        let mut tone = embedded_tone.or(if is_vowel { Some(1) } else { None });
        while pos < text.len() && tone_of(text[pos]).is_some() {
            tone = tone_of(text[pos]);
            pos += 1;
        }
        labels.phonemes.push(phone.to_string());
        labels.tone.push(tone);
        labels.stress.push(crate::Stress::None);
        if is_vowel {
            word_vowels.push(labels.phonemes.len() - 1);
        }
    }
    finish_word(&mut labels, &mut word_vowels, &mut word_start);
    Ok(labels)
}

/// Label one utterance. Text with Latin letters is refused: vachana has no
/// English or name path and mostly copies such spans through, and a
/// one-letter abbreviation may or may not have a lexicon entry, so no
/// mixed-script sentence is labeled.
pub fn phonemize(text: &str) -> Result<Labels, Error> {
    let latin: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if !latin.is_empty() {
        return Err(Error::Unlabelable(format!(
            "thai_mixed_latin_script:{}",
            latin.join(",")
        )));
    }
    parse(&raw_ipa(text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vachana_ipa_like_lexide() {
        let l = parse("sàwàtdiː kʰráp, pʰǒm tɕʰɯ̂ː").unwrap();
        assert_eq!(
            l.phonemes,
            [
                "s", "a", "w", "a", "t", "d", "iː", "kʰ", "r", "a", "p", "pʰ", "o", "m", "tɕʰ",
                "ɯː"
            ]
        );
        // Tones ride on vowels: low, low, mid, high, rising, falling.
        let tones: Vec<u8> = l.tone.iter().flatten().copied().collect();
        assert_eq!(tones, [2, 2, 1, 4, 5, 3]);
        // Final vowel of each word is stressed.
        let stressed: Vec<usize> = l
            .stress
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == crate::Stress::Primary)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(stressed, [6, 9, 12, 15]);
        assert_eq!(l.word_spans, [(0, 7), (7, 11), (11, 14), (14, 16)]);
    }

    #[test]
    fn refuses_mixed_script() {
        assert!(matches!(
            phonemize("ผมใช้ AOL"),
            Err(Error::Unlabelable(r)) if r.starts_with("thai_mixed_latin")
        ));
    }

    #[test]
    #[ignore = "needs uv on PATH and network on first run"]
    fn runs_the_python_backend() {
        let l = phonemize("สวัสดีครับ").unwrap();
        assert_eq!(l.phonemes[..4], ["s", "a", "w", "a"]);
    }
}
