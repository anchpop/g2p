//! Grapheme-to-phoneme for yap and lexide, via the maintainer's espeak-ng
//! fork (github.com/anchpop/espeak-ng, branch `french-phrase-stress-liaison`).
//!
//! The fork is a git submodule, built and linked statically by `build.rs`,
//! and its compiled phoneme data is embedded in the binary. There is no
//! binary to install, no data path to configure, and no way to run against
//! mainline espeak by accident — the failure mode that produced a corpus of
//! wrong Hindi labels and a backend running without the French patches.
//!
//! Output is byte-identical to the CLI's `espeak-ng -q --ipa -x --stdin`, the
//! invocation both projects used before, because it takes the same path
//! through the engine: a full (silent) synthesis with the phoneme trace
//! enabled. The library's simpler `espeak_TextToPhonemes` entry point skips
//! the pitch/length passes and disagrees with the CLI on tone languages and
//! some stress, so it is not used.
//!
//! Every result carries the raw IPA string espeak printed (word boundaries
//! and stress marks intact, for humans and LLMs) and its tokenization into
//! the versioned label inventory (see [`parse`]). Models must pin the g2p
//! revision used for training. Callers that score audio use the tokenized
//! form, never raw IPA.
//!
//! Not every language is an espeak language. [`label_source`] is the one
//! table of where each language's labels come from, and [`phonemize_lang`]
//! dispatches on it: espeak for most, the built-in [`hindi`], [`mandarin`],
//! and [`japanese`] chains for Hindi, Simplified Mandarin, and Japanese
//! (espeak's `hi`/`cmn`/`ja` voices are never used), and for Thai and
//! Korean the vachana and g2pk backends run as embedded, pinned Python
//! projects ([`thai`], [`korean`]).

mod attribution;
mod data;
#[cfg(test)]
mod espeak_tests;
mod ffi;
mod geminate;
pub mod hindi;
#[cfg(feature = "japanese")]
pub mod japanese;
mod voices;
#[cfg(not(feature = "japanese"))]
pub mod japanese {
    pub use g2p_types::japanese::*;
}
pub mod korean;
pub mod mandarin;
pub mod parse;
pub mod thai;

pub use g2p_types::{LabelSource, Language, Phoneme, Phonemized, Pitch, Reading, UnknownPhoneme};
pub use hindi::Syllable;

pub use parse::{Parsed, Stress};

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::sync::{Mutex, MutexGuard};

/// Digest of every espeak-ng source file that can affect phoneme output.
pub const ESPEAK_DIGEST: &str = env!("G2P_ESPEAK_DIGEST");
/// Commit of the espeak-ng submodule this crate was built from, when the
/// build could read it (`unknown` otherwise). Informational; use
/// [`identity`] for cache keys.
pub const ESPEAK_COMMIT: &str = env!("G2P_ESPEAK_COMMIT");

/// Label-compatibility identifier: crate version plus the espeak and pinned
/// Python backend digests. Rust label changes require an intentional crate
/// version bump; the source-digest regression test forces that review.
/// Label-preserving API/comment changes may refresh the test baseline without
/// changing this identifier.
///
/// Not a complete build fingerprint: toolchain, platform, environment, enabled
/// features (including Japanese availability), and downstream dependency
/// resolution are not encoded. This crate's Cargo.lock is not enforced by
/// downstream library consumers. Cache keys must also include the request
/// (language and text).
pub fn identity() -> String {
    format!(
        "g2p/{} espeak-ng/{ESPEAK_DIGEST} thai/{} korean/{}",
        env!("CARGO_PKG_VERSION"),
        thai::THAI_DIGEST,
        korean::KOREAN_DIGEST
    )
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    UnknownPhoneme(#[from] UnknownPhoneme),
    #[error("espeak-ng failed to initialize: {0}")]
    Init(String),
    #[error("could not unpack embedded espeak-ng data: {0}")]
    Data(#[from] std::io::Error),
    /// With public typed input, this means our private variety table names
    /// a voice the embedded engine does not know: an internal invariant
    /// violation, not bad caller input. Private raw-engine tests also
    /// exercise this diagnostic deliberately.
    #[error("espeak-ng has no voice {0:?}")]
    UnknownVoice(String),
    #[error("espeak-ng synthesis failed: {0}")]
    Synth(String),
    #[error("text contains a NUL byte")]
    NulByte,
    /// The backend refuses to label this text rather than emit labels that
    /// silently omit part of what is spoken (e.g. Hindi text with digits).
    /// The string is a stable `reason:detail` code.
    #[error("cannot label this text: {0}")]
    Unlabelable(String),
    #[error("no G2P backend for language {0:?}")]
    UnsupportedLanguage(String),
    /// An out-of-process backend (Thai's Python project) could not be
    /// started or died; the message says what to install.
    #[error("G2P backend unavailable: {0}")]
    Backend(String),
}

/// Phonemize `text` with an espeak voice (e.g. `fr-fr`, `en-us`, `pt-br`,
/// `cmn`). The voice name resolves exactly as the CLI's `-v` does: by voice
/// name first, then as a language. Embedded newlines are treated as spaces —
/// the text is one utterance.
///
/// Thread-safe (espeak-ng has global state; calls serialize on a lock).
fn phonemize_espeak(text: &str, voice: &str) -> Result<Phonemized, Error> {
    let (trace, language) = phonemize_espeak_traces(text, voice)?;
    let (parsed, offsets) = parse::parse_framed_with_offsets(&trace.framed, &language);
    let boundaries: Vec<_> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain([text.len()])
        .collect();
    let bytes = |(start, len): (i32, i32)| {
        if start < 0 || len <= 0 {
            return None;
        }
        Some((
            *boundaries.get(start as usize)?,
            *boundaries.get((start + len) as usize)?,
        ))
    };
    // The (word, letters) source of the engine phone whose rendering
    // contains `offset`.
    let source = |offset: usize| {
        let index = trace.sources.partition_point(|s| s.offset <= offset);
        let s = trace.sources.get(index.checked_sub(1)?)?;
        Some(Source {
            word: bytes(s.word)?,
            letters: bytes(s.letters)?,
        })
    };
    let sources = offsets
        .into_iter()
        .map(|(first, last)| {
            let (first, last) = (source(first)?, source(last)?);
            Some(Source {
                word: union(first.word, last.word),
                letters: union(first.letters, last.letters),
            })
        })
        .collect::<Vec<_>>();
    espeak_phonemized(
        trace.raw,
        parsed,
        matches!(language.as_str(), "yue" | "vi"),
        &sources,
    )
}

fn union(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
    (a.0.min(b.0), a.1.max(b.1))
}

/// A phoneme's original-input byte ranges: its source word, and the letters
/// of that word it was spoken from.
#[derive(Clone, Copy)]
struct Source {
    word: (usize, usize),
    letters: (usize, usize),
}

/// `sources` is parallel to `parsed.phonemes`. A word's text span is the union
/// of its phonemes' source words, unknown if any of them is.
fn espeak_phonemized(
    raw: String,
    parsed: Parsed,
    has_tones: bool,
    sources: &[Option<Source>],
) -> Result<Phonemized, Error> {
    let mut out = Phonemized {
        raw,
        ..Phonemized::default()
    };
    for (start, end) in parsed.word_spans {
        let word_start = out.phonemes.len();
        let mut word_source = Some((usize::MAX, 0));
        for (i, &source) in (start..end).zip(&sources[start..end]) {
            let phone = &parsed.phonemes[i];
            if has_tones && matches!(phone.as_str(), "1" | "2" | "3" | "4" | "5" | "6" | "7") {
                // eSpeak's WritePhMnemonic appends tone_ph directly to its
                // bearing phone, before any coda. Keep that exact alignment.
                if out.phonemes.len() == word_start || out.tone.last().unwrap().is_some() {
                    return Err(Error::Unlabelable(format!(
                        "espeak_unattached_tone:{phone}"
                    )));
                }
                *out.tone.last_mut().unwrap() = Some(phone.as_bytes()[0] - b'0');
            } else {
                out.phonemes.push(phone.parse()?);
                out.stress.push(parsed.stress[i]);
                if has_tones {
                    out.tone.push(None);
                }
                out.phoneme_text_spans
                    .push(source.map(|s| s.letters).unwrap_or_default());
                word_source = word_source
                    .zip(source)
                    .map(|(word, phone)| union(word, phone.word));
            }
        }
        if out.phonemes.len() > word_start {
            out.word_spans.push((word_start, out.phonemes.len()));
            out.word_text_spans.push(word_source.unwrap_or_default());
        }
    }
    Ok(out)
}

/// Just espeak's IPA string for `text` (clauses joined with single spaces).
#[cfg(test)]
fn phonemize_espeak_raw(text: &str, voice: &str) -> Result<String, Error> {
    Ok(phonemize_espeak_traces(text, voice)?.0.raw)
}

fn phonemize_espeak_traces(text: &str, voice: &str) -> Result<(Clause, String), Error> {
    let mut guard = engine()?;
    let engine = guard.as_mut().expect("engine() initializes the engine");
    engine.select_voice(voice)?;
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let clauses = engine.synth(&flat)?;
    let mut trace = Clause::default();
    for clause in clauses {
        if !trace.framed.is_empty() {
            trace.framed.push(' ');
        }
        let base = trace.framed.len();
        trace
            .sources
            .extend(clause.sources.into_iter().map(|source| PhoneSource {
                offset: base + source.offset,
                ..source
            }));
        trace.framed.push_str(&clause.framed);
        trace.raw.push_str(&clause.raw);
        trace.raw.push(' ');
    }
    trace.raw = trace.raw.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok((trace, engine.language.clone()))
}

/// Label source for a language code (ISO 639-3, `zho-hans` for Simplified
/// Mandarin). `None` for languages no consumer labels.
pub fn label_source(lang: &str) -> Option<LabelSource> {
    use LabelSource::*;
    Some(match lang {
        "hin" => Hindi,
        "zho-hans" => Mandarin,
        "jpn" => Japanese,
        "tha" => Thai,
        "kor" => Korean,
        _ if voices::ESPEAK_VOICES
            .iter()
            .any(|(language, _)| language.code() == lang) =>
        {
            Espeak
        }
        _ => return None,
    })
}

/// Phonemize a language-only code using its established pronunciation default.
/// Use [`phonemize`] to select an explicit regional pronunciation.
pub fn phonemize_lang(lang: &str, text: &str) -> Result<Phonemized, Error> {
    let language =
        Language::from_code(lang).ok_or_else(|| Error::UnsupportedLanguage(lang.to_string()))?;
    phonemize(language, text)
}

/// Phonemize with an explicit language–variety choice.
/// The shared enum only represents supported combinations; no separate
/// variety or backend selection is needed.
pub fn phonemize(language: Language, text: &str) -> Result<Phonemized, Error> {
    phonemize_with(language, text, &Hints::default())
}

/// What the caller already knows about the text, for [`phonemize_with`].
/// A language that cannot use a kind of hint refuses it rather than ignore it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hints<'a> {
    /// Pronunciations to use rather than predict: the result agrees with
    /// every reading or is an error. Japanese and Simplified Mandarin only;
    /// elsewhere the spelling already determines the pronunciation.
    pub readings: &'a [Reading],
    /// The text's words, as `[start, end)` byte ranges in order, not
    /// overlapping; text between them must be unspoken (whitespace,
    /// punctuation). A range may be punctuation alone, which says nothing.
    /// Each word is phonemized on its own, instead of the backend segmenting
    /// the text itself, and its phonemes are spoken from it. Thai only, whose
    /// unspaced script otherwise goes through a dictionary segmenter that
    /// joins words (ได้มา) and confuses lookalikes (มา + นะ as มานะ).
    pub words: &'a [(usize, usize)],
}

/// [`phonemize`], using what [`Hints`] say about the text.
pub fn phonemize_with(language: Language, text: &str, hints: &Hints) -> Result<Phonemized, Error> {
    let Hints { readings, words } = *hints;
    let lang = language.code();
    let source = label_source(lang).ok_or_else(|| Error::UnsupportedLanguage(lang.to_string()))?;
    // English corpus records can contain Korean instructional speech. Do not
    // let eSpeak silently route that speech through its Korean voice.
    if lang == "eng"
        && text.chars().any(|c| {
            matches!(c,
                '\u{1100}'..='\u{11ff}' | '\u{3130}'..='\u{318f}' |
                '\u{a960}'..='\u{a97f}' | '\u{ac00}'..='\u{d7af}' |
                '\u{d7b0}'..='\u{d7ff}' | '\u{ffa0}'..='\u{ffdc}'
            )
        })
    {
        return Err(Error::Unlabelable(
            "english_hangul:request contains Korean text".into(),
        ));
    }

    if lang == "ara" {
        let digits: String = text
            .chars()
            .filter(|c| c.is_ascii_digit() || ('\u{0660}'..='\u{0669}').contains(c))
            .collect();
        if !digits.is_empty() {
            return Err(Error::Unlabelable(format!("arabic_digits:{digits}")));
        }
    }

    if !readings.is_empty() && !matches!(source, LabelSource::Mandarin | LabelSource::Japanese) {
        return Err(Error::Unlabelable(format!("readings_unsupported:{lang}")));
    }
    if !words.is_empty() {
        if !matches!(source, LabelSource::Thai) {
            return Err(Error::Unlabelable(format!("words_unsupported:{lang}")));
        }
        check_words(text, words)?;
    }

    let mut result = match source {
        LabelSource::Espeak => phonemize_espeak(text, engine_voice(language)),
        LabelSource::Hindi => hindi_phonemized(hindi::phonemize(text)?),
        LabelSource::Mandarin => mandarin_phonemized(mandarin::phonemize(text, readings)?, text),
        #[cfg(feature = "japanese")]
        LabelSource::Japanese => japanese_phonemized(japanese::phonemize(text, readings)?),
        #[cfg(not(feature = "japanese"))]
        LabelSource::Japanese => Err(Error::UnsupportedLanguage(lang.to_string())),
        LabelSource::Thai => thai_phonemized(thai::phonemize(text, words)?),
        LabelSource::Korean => korean_phonemized(korean::phonemize(text)?),
    }?;
    geminate::merge(&mut result);
    attribution::assign(&mut result, text);
    Ok(result)
}

/// [`Hints::words`] must tile the spoken text: in order, on character
/// boundaries, with only unspoken text between them.
fn check_words(text: &str, words: &[(usize, usize)]) -> Result<(), Error> {
    let mut end = 0;
    for &(start, stop) in words {
        let gap = text.get(end..start);
        if start >= stop
            || text.get(start..stop).is_none()
            || !gap.is_some_and(|gap| gap.chars().all(attribution::separator))
        {
            return Err(Error::Unlabelable(format!("words_invalid:{start}..{stop}")));
        }
        end = stop;
    }
    if !text[end..].chars().all(attribution::separator) {
        return Err(Error::Unlabelable(format!("words_invalid:{end}..")));
    }
    Ok(())
}

fn engine_voice(language: Language) -> &'static str {
    voices::ESPEAK_VOICES
        .iter()
        .find(|(candidate, _)| *candidate == language)
        .map(|(_, voice)| *voice)
        .expect("eSpeak language has a private voice mapping")
}

/// Korean labels in the common shape: no stress, no tone, the post-sandhi
/// Hangul as `raw`.
fn korean_phonemized(labels: korean::Labels) -> Result<Phonemized, Error> {
    Ok(Phonemized {
        raw: labels.raw,
        phonemes: labels
            .phonemes
            .iter()
            .map(|p| p.parse())
            .collect::<Result<_, _>>()?,
        stress: labels.stress,
        word_spans: labels.word_spans,
        word_text_spans: labels.word_text_spans,
        phoneme_text_spans: labels.phoneme_text_spans,
        ..Phonemized::default()
    })
}

/// Thai labels in the common shape.
fn thai_phonemized(labels: thai::Labels) -> Result<Phonemized, Error> {
    let mut out = Phonemized {
        raw: labels.raw,
        phonemes: labels
            .phonemes
            .iter()
            .map(|p| p.parse())
            .collect::<Result<_, _>>()?,
        stress: labels.stress,
        word_spans: labels.word_spans,
        word_text_spans: labels.word_text_spans,
        tone: labels.tone,
        ..Phonemized::default()
    };
    // Each phoneme was spoken from its segmented word.
    out.phoneme_text_spans = word_level_phoneme_spans(&out);
    Ok(out)
}

/// Japanese NJD-node word boundaries, pitch factors, and native phone trace.
#[cfg(feature = "japanese")]
fn japanese_phonemized(labels: japanese::Labels) -> Result<Phonemized, Error> {
    let n = labels.phonemes.len();
    Ok(Phonemized {
        raw: labels.native_phones.join(" "),
        stress: vec![Stress::None; n],
        word_spans: if labels.word_spans.is_empty() && n != 0 {
            vec![(0, n)]
        } else {
            labels.word_spans
        },
        word_text_spans: labels.word_text_spans,
        phoneme_text_spans: labels.phoneme_text_spans,
        phonemes: labels
            .phonemes
            .iter()
            .map(|p| p.parse())
            .collect::<Result<_, _>>()?,
        pitch: labels.pitch,
        accent_withheld: labels.accent_withheld,
        ..Phonemized::default()
    })
}

/// Flatten per-syllable Mandarin labels into the common shape: one word span
/// per syllable, no stress, tone on the bearing phone, and pinyin as `raw`.
fn mandarin_phonemized(
    syllables: Vec<mandarin::Syllable>,
    text: &str,
) -> Result<Phonemized, Error> {
    let surfaces: Vec<_> = syllables.iter().map(|s| s.char.to_string()).collect();
    let mut out = Phonemized {
        word_text_spans: attribution::align(text, surfaces.iter().map(String::as_str)),
        ..Phonemized::default()
    };
    let mut pinyin = Vec::with_capacity(syllables.len());
    for s in syllables {
        let start = out.phonemes.len();
        out.stress
            .extend(std::iter::repeat_n(Stress::None, s.phonemes.len()));
        out.tone.extend(s.tone);
        out.phonemes.extend(
            s.phonemes
                .iter()
                .map(|p| p.parse())
                .collect::<Result<Vec<Phoneme>, _>>()?,
        );
        out.word_spans.push((start, out.phonemes.len()));
        pinyin.push(s.pinyin);
    }
    out.raw = pinyin.join(" ");
    // Each phoneme was spoken from its syllable's character.
    out.phoneme_text_spans = word_level_phoneme_spans(&out);
    Ok(out)
}

/// Flatten per-word Hindi labels into the common shape.
/// Every phoneme gets its word's text span, for backends whose words are the
/// finest source unit they know.
fn word_level_phoneme_spans(out: &Phonemized) -> Vec<(usize, usize)> {
    out.word_text_spans
        .iter()
        .zip(&out.word_spans)
        .flat_map(|(&text, &(start, end))| std::iter::repeat_n(text, end - start))
        .collect()
}

fn hindi_phonemized(words: Vec<hindi::Word>) -> Result<Phonemized, Error> {
    let mut out = Phonemized::default();
    let mut raw_words = Vec::with_capacity(words.len());
    for w in words {
        out.word_text_spans.push(w.text_span.unwrap_or_default());
        let start = out.phonemes.len();
        out.syllables
            .extend(w.syllables.into_iter().map(|s| Syllable {
                start: s.start + start,
                end: s.end + start,
                nucleus: s.nucleus + start,
                ..s
            }));
        raw_words.push(w.phonemes.concat());
        out.phonemes.extend(
            w.phonemes
                .iter()
                .map(|p| p.parse())
                .collect::<Result<Vec<Phoneme>, _>>()?,
        );
        out.stress.extend(w.stress);
        out.word_spans.push((start, out.phonemes.len()));
    }
    // Each phoneme was spoken from its word.
    out.phoneme_text_spans = word_level_phoneme_spans(&out);
    out.raw = raw_words.join(" ");
    Ok(out)
}

struct Engine {
    voice: Option<String>,
    language: String,
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

#[derive(Default)]
struct Clause {
    raw: String,
    framed: String,
    sources: Vec<PhoneSource>,
}

/// Where one rendered engine phone came from, as (absolute source character
/// index, length) pairs: its word, and the letters of that word it was spoken
/// from (the whole word when the engine does not know them).
#[derive(Clone, Copy)]
struct PhoneSource {
    /// Rendered byte offset of the phone.
    offset: usize,
    word: (i32, i32),
    letters: (i32, i32),
}

thread_local! {
    static CLAUSES: RefCell<Vec<Clause>> = const { RefCell::new(Vec::new()) };
    static PHONE_SOURCES: RefCell<Vec<PhoneSource>> = const { RefCell::new(Vec::new()) };
}

unsafe extern "C" fn collect_phone_source(
    offset: i32,
    start: i32,
    len: i32,
    letters_start: i32,
    letters_len: i32,
) {
    PHONE_SOURCES.with(|s| {
        s.borrow_mut().push(PhoneSource {
            offset: offset as usize,
            word: (start, len),
            letters: (letters_start, letters_len),
        })
    });
}

unsafe extern "C" fn discard_audio(
    _wav: *mut std::os::raw::c_short,
    _n: std::os::raw::c_int,
    _events: *mut ffi::espeak_EVENT,
) -> std::os::raw::c_int {
    0
}

unsafe extern "C" fn collect_clause(s: *const std::os::raw::c_char) -> std::os::raw::c_int {
    if !s.is_null() {
        // SAFETY: espeak passes a NUL-terminated string it owns for the
        // duration of the call; we copy it out immediately.
        let line = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
        // SAFETY: synchronous callback runs under ENGINE's lock, after
        // CalcPitches/CalcLengths and before Generate mutates the phone list.
        // The internal renderer reuses its static buffer, so raw MUST be
        // copied first, and the separated result copied before returning.
        // It only formats the current list; there is no second synthesis.
        let framed = unsafe {
            CStr::from_ptr(ffi::GetTranslatedPhonemeStringWithSource(
                ffi::ESPEAK_PHONEMES_IPA | ((parse::PHONE_SEPARATOR as i32) << 8),
                collect_phone_source,
            ))
        }
        .to_string_lossy()
        .into_owned();
        let sources = PHONE_SOURCES.with(|s| std::mem::take(&mut *s.borrow_mut()));
        CLAUSES.with(|c| {
            c.borrow_mut().push(Clause {
                raw: line,
                framed,
                sources,
            })
        });
    }
    0
}

fn engine() -> Result<MutexGuard<'static, Option<Engine>>, Error> {
    let mut guard = ENGINE.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(Engine::init()?);
    }
    Ok(guard)
}

impl Engine {
    fn init() -> Result<Engine, Error> {
        let dir = data::ensure_unpacked()?;
        let c_dir = CString::new(dir.to_string_lossy().as_bytes()).map_err(|_| Error::NulByte)?;
        // SAFETY: plain C calls with valid NUL-terminated arguments; this
        // runs once, under the engine lock.
        unsafe {
            ffi::espeak_ng_InitializePath(c_dir.as_ptr());
            let mut ctx: ffi::espeak_ng_ERROR_CONTEXT = std::ptr::null_mut();
            let st = ffi::espeak_ng_Initialize(&mut ctx);
            if st != ffi::ENS_OK {
                return Err(Error::Init(format!(
                    "{} (data dir {})",
                    ffi::status_message(st),
                    dir.display()
                )));
            }
            let st = ffi::espeak_ng_InitializeOutput(
                ffi::ENOUTPUT_MODE_SYNCHRONOUS,
                0,
                std::ptr::null(),
            );
            if st != ffi::ENS_OK {
                return Err(Error::Init(ffi::status_message(st)));
            }
            ffi::espeak_SetSynthCallback(Some(discard_audio));
            // The trace mode is what makes the engine render IPA; the stream
            // it also prints to is irrelevant (we take the string from the
            // callback), so point it at /dev/null.
            let devnull = libc::fopen(c"/dev/null".as_ptr(), c"w".as_ptr());
            if devnull.is_null() {
                return Err(Error::Init("could not open /dev/null".into()));
            }
            ffi::espeak_SetPhonemeTrace(
                ffi::ESPEAK_PHONEMES_IPA | ffi::ESPEAK_PHONEMES_SHOW,
                devnull,
            );
            ffi::espeak_SetPhonemeCallback(Some(collect_clause));
        }
        Ok(Engine {
            voice: None,
            language: String::new(),
        })
    }

    fn select_voice(&mut self, voice: &str) -> Result<(), Error> {
        if self.voice.as_deref() == Some(voice) {
            return Ok(());
        }
        let c_voice = CString::new(voice).map_err(|_| Error::NulByte)?;
        // Same resolution order as the CLI: a voice name, else a language.
        // SAFETY: valid NUL-terminated string; struct fully initialized.
        let ok = unsafe {
            ffi::espeak_ng_SetVoiceByName(c_voice.as_ptr()) == ffi::ENS_OK || {
                let mut sel = ffi::espeak_VOICE {
                    name: std::ptr::null(),
                    languages: c_voice.as_ptr(),
                    identifier: std::ptr::null(),
                    gender: 0,
                    age: 0,
                    variant: 0,
                    xx1: 0,
                    score: 0,
                    spare: std::ptr::null_mut(),
                };
                ffi::espeak_ng_SetVoiceByProperties(&mut sel) == ffi::ENS_OK
            }
        };
        if !ok {
            // Leave `self.voice` unset: whatever espeak has loaded now is
            // not what the caller asked for.
            self.voice = None;
            return Err(Error::UnknownVoice(voice.to_string()));
        }
        // Use the resolved voice's primary language, not the caller's alias
        // ("English (America)", "en-us+f3", etc.). `languages` is a list of
        // priority-byte + NUL-terminated language entries; we need the first.
        // SAFETY: a successful voice selection supplies the engine-owned
        // voice; pointers remain valid while this lock is held.
        self.language = unsafe {
            let selected = ffi::espeak_GetCurrentVoice();
            CStr::from_ptr((*selected).languages.add(1))
        }
        .to_string_lossy()
        .into_owned();
        self.voice = Some(voice.to_string());
        Ok(())
    }

    fn synth(&mut self, text: &str) -> Result<Vec<Clause>, Error> {
        let c_text = CString::new(text).map_err(|_| Error::NulByte)?;
        CLAUSES.with(|c| c.borrow_mut().clear());
        // SAFETY: `size` includes the terminating NUL as the API requires;
        // callbacks were registered in `init`.
        let st = unsafe {
            ffi::espeak_Synth(
                c_text.as_ptr().cast(),
                c_text.as_bytes_with_nul().len(),
                0,
                ffi::POS_CHARACTER,
                0,
                ffi::ESPEAK_CHARS_AUTO | ffi::ESPEAK_PHONEMES | ffi::ESPEAK_ENDPAUSE,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if st != ffi::ENS_OK {
            return Err(Error::Synth(ffi::status_message(st)));
        }
        Ok(CLAUSES.with(|c| std::mem::take(&mut *c.borrow_mut())))
    }
}
