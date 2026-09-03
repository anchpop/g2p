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
//! the phoneme sequence the lexide pronunciation model was trained on (see
//! [`parse`]). Callers that score audio must use the tokenized form; the two
//! must never be mixed.
//!
//! Not every language is an espeak language. [`label_source`] is the one
//! table of where each language's labels come from, and [`phonemize_lang`]
//! dispatches on it: espeak for most, the built-in [`hindi`] chain for Hindi
//! (espeak's `hi` voice is never used), and a refusal for languages whose
//! labels come from Python backends this crate does not run.

mod data;
mod ffi;
pub mod hindi;
pub mod parse;

pub use hindi::{Canon as HindiCanon, Syllable};
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

/// Identifies *which* phonemizer produced an output: crate version plus the
/// espeak source digest. Stamp cached or persisted phoneme data with this so
/// output from a different build can never pose as current.
pub fn identity() -> String {
    format!(
        "g2p/{} espeak-ng/{ESPEAK_DIGEST}",
        env!("CARGO_PKG_VERSION")
    )
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("espeak-ng failed to initialize: {0}")]
    Init(String),
    #[error("could not unpack embedded espeak-ng data: {0}")]
    Data(#[from] std::io::Error),
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
}

/// Phonemization of one utterance.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Phonemized {
    /// espeak's own IPA output — stress marks and word boundaries intact,
    /// clauses joined with single spaces. Readable; not for scoring.
    pub raw: String,
    /// Model-label tokenization of `raw` (see [`parse`]).
    pub phonemes: Vec<String>,
    /// Parallel to `phonemes`.
    pub stress: Vec<Stress>,
    /// `[start, end)` ranges into `phonemes`, one per word espeak emitted.
    pub word_spans: Vec<(usize, usize)>,
    /// Syllable spans (absolute indices into `phonemes`) for backends that
    /// compute them — Hindi. Empty for espeak languages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub syllables: Vec<Syllable>,
}

/// Phonemize `text` with an espeak voice (e.g. `fr-fr`, `en-us`, `pt-br`,
/// `cmn`). The voice name resolves exactly as the CLI's `-v` does: by voice
/// name first, then as a language. Embedded newlines are treated as spaces —
/// the text is one utterance.
///
/// Thread-safe (espeak-ng has global state; calls serialize on a lock).
pub fn phonemize(text: &str, voice: &str) -> Result<Phonemized, Error> {
    let raw = phonemize_raw(text, voice)?;
    let Parsed {
        phonemes,
        stress,
        word_spans,
    } = parse::parse(&raw);
    Ok(Phonemized {
        raw,
        phonemes,
        stress,
        word_spans,
        syllables: Vec::new(),
    })
}

/// Just espeak's IPA string for `text` (clauses joined with single spaces).
pub fn phonemize_raw(text: &str, voice: &str) -> Result<String, Error> {
    let mut guard = engine()?;
    let engine = guard.as_mut().expect("engine() initializes the engine");
    engine.select_voice(voice)?;
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let clauses = engine.synth(&flat)?;
    Ok(clauses
        .iter()
        .flat_map(|c| c.split_whitespace())
        .collect::<Vec<_>>()
        .join(" "))
}

/// Where a language's phoneme labels come from. One table for both yap and
/// lexide: which G2P a language may use is a correctness constraint, not a
/// preference — targets from a different source than the model's training
/// labels disagree about the phoneme inventory, and nothing downstream can
/// tell (Hindi scored against espeak `hi` measured as the worst language by
/// a wide margin before this was understood).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSource {
    /// Our espeak-ng fork, with this voice.
    Espeak(&'static str),
    /// The ported `schwa-stress-hin` chain ([`hindi`]).
    Hindi,
    /// A Python backend lexide runs outside this crate, by provider name.
    /// Not callable from here yet.
    ExternalBackend(&'static str),
}

/// Label source for a language code (ISO 639-3, `zho-hans` for Simplified
/// Mandarin). `None` for languages no consumer labels.
pub fn label_source(lang: &str) -> Option<LabelSource> {
    use LabelSource::*;
    Some(match lang {
        "hin" => Hindi,
        "jpn" => ExternalBackend("pyopenjtalk"),
        "zho-hans" => ExternalBackend("g2pm-ipa"),
        "tha" => ExternalBackend("vachana-thai"),
        // Model languages labeled from espeak. `pt-br`, not `pt`: European
        // Portuguese targets against Brazilian audio measured 41% median
        // phoneme distance where `pt-br` measured 31%.
        "eng" => Espeak("en-us"),
        "deu" => Espeak("de"),
        "fra" => Espeak("fr-fr"),
        "ita" => Espeak("it"),
        "por" => Espeak("pt-br"),
        "spa" => Espeak("es"),
        "rus" => Espeak("ru"),
        // Korean uses espeak because nothing better is wired up, not because
        // espeak has been checked against a Korean corpus. Validate before
        // trusting.
        "kor" => Espeak("ko"),
        // Pimsleur-era languages in lexide's corpus, espeak-labeled and not
        // through a backend audit.
        "sqi" => Espeak("sq"),
        "ara" => Espeak("ar"),
        "hye" => Espeak("hy"),
        "yue" => Espeak("yue"),
        "hrv" => Espeak("hr"),
        "ces" => Espeak("cs"),
        "dan" => Espeak("da"),
        "fas" => Espeak("fa"),
        "nld" => Espeak("nl"),
        "fin" => Espeak("fi"),
        "hat" => Espeak("ht"),
        "heb" => Espeak("he"),
        "hun" => Espeak("hu"),
        "isl" => Espeak("is"),
        "ind" => Espeak("id"),
        "gle" => Espeak("ga"),
        "ell" => Espeak("el"),
        "nor" => Espeak("nb"),
        "pol" => Espeak("pl"),
        "pan" => Espeak("pa"),
        "ron" => Espeak("ro"),
        "swa" => Espeak("sw"),
        "swe" => Espeak("sv"),
        "tur" => Espeak("tr"),
        "ukr" => Espeak("uk"),
        "urd" => Espeak("ur"),
        "vie" => Espeak("vi"),
        _ => return None,
    })
}

/// Phonemize `text` as language `lang` (see [`label_source`]), with the
/// current Hindi canon. Espeak languages take the espeak path; Hindi takes
/// the ported chain; languages with an external backend or no source are
/// [`Error::UnsupportedLanguage`].
pub fn phonemize_lang(lang: &str, text: &str) -> Result<Phonemized, Error> {
    phonemize_lang_with(lang, text, HindiCanon::Current)
}

/// [`phonemize_lang`] with an explicit Hindi label canon (irrelevant for
/// other languages).
pub fn phonemize_lang_with(lang: &str, text: &str, canon: HindiCanon) -> Result<Phonemized, Error> {
    match label_source(lang) {
        Some(LabelSource::Espeak(voice)) => phonemize(text, voice),
        Some(LabelSource::Hindi) => Ok(hindi_phonemized(hindi::phonemize(text, canon)?)),
        Some(LabelSource::ExternalBackend(_)) | None => {
            Err(Error::UnsupportedLanguage(lang.to_string()))
        }
    }
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
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

thread_local! {
    /// Clause strings delivered by the phoneme callback during one `synth`.
    static CLAUSES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
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
        CLAUSES.with(|c| c.borrow_mut().push(line));
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
        Ok(Engine { voice: None })
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
        self.voice = Some(voice.to_string());
        Ok(())
    }

    fn synth(&mut self, text: &str) -> Result<Vec<String>, Error> {
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
