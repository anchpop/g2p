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

mod data;
mod ffi;
pub mod parse;

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
