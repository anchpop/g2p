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

mod data;
#[cfg(test)]
mod espeak_tests;
mod ffi;
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

pub use g2p_types::{LabelSource, Phonemized, Pitch, Variety};
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
/// (language, text, and variety).
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
    /// The language has no labels for this non-default variety.
    #[error("variety {variety:?} is not applicable to language {lang:?}")]
    VarietyNotApplicable { lang: String, variety: Variety },
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
fn phonemize(text: &str, voice: &str) -> Result<Phonemized, Error> {
    let (raw, framed, language) = phonemize_traces(text, voice)?;
    let Parsed {
        phonemes,
        stress,
        word_spans,
    } = parse::parse_framed(&framed, &language);
    Ok(Phonemized {
        raw,
        phonemes,
        stress,
        word_spans,
        syllables: Vec::new(),
        tone: Vec::new(),
        pitch: Vec::new(),
        accent_withheld: None,
    })
}

/// Just espeak's IPA string for `text` (clauses joined with single spaces).
#[cfg(test)]
fn phonemize_raw(text: &str, voice: &str) -> Result<String, Error> {
    Ok(phonemize_traces(text, voice)?.0)
}

fn phonemize_traces(text: &str, voice: &str) -> Result<(String, String, String), Error> {
    let mut guard = engine()?;
    let engine = guard.as_mut().expect("engine() initializes the engine");
    engine.select_voice(voice)?;
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let clauses = engine.synth(&flat)?;
    let join = |lines: Vec<String>| {
        lines
            .iter()
            .flat_map(|c| c.split_whitespace())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let (raw, framed): (Vec<_>, Vec<_>) = clauses.into_iter().unzip();
    Ok((join(raw), join(framed), engine.language.clone()))
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
            .any(|(language, _, _)| *language == lang) =>
        {
            Espeak
        }
        _ => return None,
    })
}

/// Borrowed input to the phonemizer. Use [`Self::new`] for established defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct PhonemizeRequest<'a> {
    pub lang: &'a str,
    pub text: &'a str,
    pub variety: Variety,
}

impl<'a> PhonemizeRequest<'a> {
    pub fn new(lang: &'a str, text: &'a str) -> Self {
        Self {
            lang,
            text,
            variety: Variety::Default,
        }
    }

    /// Select the language's pronunciation variety.
    pub fn variety(mut self, variety: Variety) -> Self {
        self.variety = variety;
        self
    }
}

/// Phonemize `text` as language `lang` (see [`label_source`]), with the
/// trained Hindi labels and the language's default variety/backend.
pub fn phonemize_lang(lang: &str, text: &str) -> Result<Phonemized, Error> {
    phonemize_language(PhonemizeRequest::new(lang, text))
}

/// Phonemize a language with typed pronunciation selection.
/// Spanish defaults to European labels; Portuguese defaults to Brazilian.
/// Unsupported varieties return [`Error::VarietyNotApplicable`], and unknown
/// language codes always return [`Error::UnsupportedLanguage`].
pub fn phonemize_language(request: PhonemizeRequest<'_>) -> Result<Phonemized, Error> {
    let PhonemizeRequest {
        lang,
        text,
        variety,
    } = request;
    let source = label_source(lang).ok_or_else(|| Error::UnsupportedLanguage(lang.to_string()))?;
    match source {
        LabelSource::Espeak => phonemize(text, variety_voice(lang, variety)?),
        _ if variety != Variety::Default => Err(Error::VarietyNotApplicable {
            lang: lang.to_string(),
            variety,
        }),
        LabelSource::Hindi => Ok(hindi_phonemized(hindi::phonemize(text)?)),
        LabelSource::Mandarin => Ok(mandarin_phonemized(mandarin::phonemize(text)?)),
        #[cfg(feature = "japanese")]
        LabelSource::Japanese => Ok(japanese_phonemized(japanese::phonemize(text)?)),
        #[cfg(not(feature = "japanese"))]
        LabelSource::Japanese => Err(Error::UnsupportedLanguage(lang.to_string())),
        LabelSource::Thai => Ok(thai_phonemized(thai::phonemize(text)?)),
        LabelSource::Korean => Ok(korean_phonemized(korean::phonemize(text)?)),
    }
}

fn variety_voice(lang: &str, variety: Variety) -> Result<&'static str, Error> {
    voices::ESPEAK_VOICES
        .iter()
        .find(|(language, candidate, _)| *language == lang && *candidate == variety)
        .map(|(_, _, voice)| *voice)
        .ok_or_else(|| Error::VarietyNotApplicable {
            lang: lang.to_string(),
            variety,
        })
}

/// Korean labels in the common shape: no stress, no tone, the post-sandhi
/// Hangul as `raw`.
fn korean_phonemized(labels: korean::Labels) -> Phonemized {
    Phonemized {
        raw: labels.raw,
        phonemes: labels.phonemes,
        stress: labels.stress,
        word_spans: labels.word_spans,
        ..Phonemized::default()
    }
}

/// Thai labels in the common shape.
fn thai_phonemized(labels: thai::Labels) -> Phonemized {
    Phonemized {
        raw: labels.raw,
        phonemes: labels.phonemes,
        stress: labels.stress,
        word_spans: labels.word_spans,
        tone: labels.tone,
        ..Phonemized::default()
    }
}

/// Japanese labels in the common shape: one word span for the utterance
/// (OpenJTalk's word boundaries are not part of the label), no stress, the
/// pitch factor, and OpenJTalk's phone string as `raw`.
#[cfg(feature = "japanese")]
fn japanese_phonemized(labels: japanese::Labels) -> Phonemized {
    let n = labels.phonemes.len();
    Phonemized {
        raw: labels.native_phones.join(" "),
        stress: vec![Stress::None; n],
        word_spans: if n == 0 { Vec::new() } else { vec![(0, n)] },
        phonemes: labels.phonemes,
        pitch: labels.pitch,
        accent_withheld: labels.accent_withheld,
        ..Phonemized::default()
    }
}

/// Flatten per-syllable Mandarin labels into the common shape: one word span
/// per syllable, no stress, tone on the bearing phone, and pinyin as `raw`.
fn mandarin_phonemized(syllables: Vec<mandarin::Syllable>) -> Phonemized {
    let mut out = Phonemized::default();
    let mut pinyin = Vec::with_capacity(syllables.len());
    for s in syllables {
        let start = out.phonemes.len();
        out.stress
            .extend(std::iter::repeat_n(Stress::None, s.phonemes.len()));
        out.tone.extend(s.tone);
        out.phonemes.extend(s.phonemes);
        out.word_spans.push((start, out.phonemes.len()));
        pinyin.push(s.pinyin);
    }
    out.raw = pinyin.join(" ");
    out
}

/// Flatten per-word Hindi labels into the common shape.
fn hindi_phonemized(words: Vec<hindi::Word>) -> Phonemized {
    let mut out = Phonemized::default();
    let mut raw_words = Vec::with_capacity(words.len());
    for w in words {
        let start = out.phonemes.len();
        out.syllables
            .extend(w.syllables.into_iter().map(|s| Syllable {
                start: s.start + start,
                end: s.end + start,
                nucleus: s.nucleus + start,
                ..s
            }));
        raw_words.push(w.phonemes.concat());
        out.phonemes.extend(w.phonemes);
        out.stress.extend(w.stress);
        out.word_spans.push((start, out.phonemes.len()));
    }
    out.raw = raw_words.join(" ");
    out
}

struct Engine {
    voice: Option<String>,
    language: String,
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

thread_local! {
    /// Raw and phone-separated renderings of the SAME post-pitch/length clause.
    static CLAUSES: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
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
            CStr::from_ptr(ffi::GetTranslatedPhonemeString(
                ffi::ESPEAK_PHONEMES_IPA | ((parse::PHONE_SEPARATOR as i32) << 8),
            ))
        }
        .to_string_lossy()
        .into_owned();
        CLAUSES.with(|c| c.borrow_mut().push((line, framed)));
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

    fn synth(&mut self, text: &str) -> Result<Vec<(String, String)>, Error> {
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
