//! `g2p` — command-line front end for non-Rust consumers (lexide's Python
//! pipeline).
//!
//! ```text
//! g2p identity                 print the build identity (for cache keys)
//! g2p --lang <code> [--variety <name>] [--] <text...>
//!                              phonemize by language and variety
//! g2p serve                    JSON lines: one request per line on stdin,
//!                              one response per line on stdout, flushed
//!                              after each — keep one process running and
//!                              stream requests through it.
//! ```
//!
//! Request: `{"text": "यह शहर", "lang": "hin"}`. Hindi always uses
//! the deployed model's labels; there is no wire-level label version.
//! Select a variety with `{"text": "cinco", "lang": "spa",
//! "variety": "latin_american"}` (`default` when omitted).
//! Legacy Python wire requests may still send `voice`: it is converted to a
//! language/variety and overrides both wire fields. Unmapped voices fail;
//! no arbitrary engine voice is exposed by the Rust API or positional CLI.
//! Response: `{"raw": "ɔ̃ nˈɛ", "phonemes": ["ɔ̃","n","ɛ"], "stress": [0,0,1],
//!            "word_spans": [[0,1],[1,3]]}` plus `"syllables": [...]` when
//!            the backend computes them, or `{"error": "...",
//!            "unlabelable": "reason:detail"}` (the second key only when the
//!            backend refused the text rather than failed).
//!
//! Each request is phonemized as exactly one utterance, so the clause-vs-line
//! framing ambiguity of `espeak-ng --stdin` (a comma splits a line in two, a
//! line without terminal punctuation merges into the next) cannot occur.

#[path = "../voices.rs"]
mod voices;

use std::io::{BufRead, Write};
use std::os::fd::FromRawFd;

#[derive(serde::Deserialize)]
struct Request {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    variety: g2p::Variety,
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum Response {
    Ok {
        raw: String,
        phonemes: Vec<String>,
        stress: Vec<u8>,
        word_spans: Vec<(usize, usize)>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        syllables: Vec<g2p::Syllable>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tone: Vec<Option<u8>>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pitch: Vec<Option<g2p::Pitch>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        accent_withheld: Option<String>,
    },
    Err {
        error: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        unlabelable: Option<String>,
    },
}

impl From<Result<g2p::Phonemized, g2p::Error>> for Response {
    fn from(r: Result<g2p::Phonemized, g2p::Error>) -> Self {
        match r {
            Ok(p) => Response::Ok {
                raw: p.raw,
                phonemes: p.phonemes,
                stress: p.stress.iter().map(|s| s.code()).collect(),
                word_spans: p.word_spans,
                syllables: p.syllables,
                tone: p.tone,
                pitch: p.pitch,
                accent_withheld: p.accent_withheld,
            },
            Err(e) => Response::Err {
                unlabelable: match &e {
                    g2p::Error::Unlabelable(reason) => Some(reason.clone()),
                    _ => None,
                },
                error: e.to_string(),
            },
        }
    }
}

/// Temporary adapter for Python's legacy wire requests. Explicit varieties
/// take precedence over equivalent default entries when inverting the table.
fn from_espeak_voice(voice: &str) -> Result<(&'static str, g2p::Variety), String> {
    voices::ESPEAK_VOICES.iter()
        .filter(|(_, _, name)| *name == voice)
        .max_by_key(|(_, variety, _)| *variety != g2p::Variety::Default)
        .map(|(lang, variety, _)| (*lang, *variety))
        .ok_or_else(|| format!(
            "no variety maps to voice {voice}; only voices present in the training corpus can be replayed"
        ))
}

fn handle(req: Request) -> Response {
    let (lang, variety) = if let Some(voice) = req.voice.as_deref() {
        // Historical voice is authoritative even when the wire also supplies
        // a contradictory language or variety. There is no raw-engine fallback.
        match from_espeak_voice(voice) {
            Ok(selection) => selection,
            Err(error) => {
                return Response::Err {
                    error,
                    unlabelable: None,
                };
            }
        }
    } else if let Some(lang) = req.lang.as_deref() {
        (lang, req.variety)
    } else {
        return Response::Err {
            error: "request needs at least one of `voice` or `lang`".into(),
            unlabelable: None,
        };
    };
    let request = g2p::PhonemizeRequest::new(lang, &req.text).variety(variety);
    Response::from(g2p::phonemize_language(request))
}

const USAGE: &str =
    "usage: g2p identity | g2p serve | g2p --lang <code> [--variety <name>] [--] <text...>";

fn cli_request(args: &[String]) -> Result<Request, String> {
    if args.first().map(String::as_str) != Some("--lang") || args.len() < 3 {
        return Err(USAGE.into());
    }
    let mut text_start = 2;
    let mut variety = g2p::Variety::Default;
    if args[text_start] == "--variety" {
        let value = args.get(text_start + 1).ok_or(USAGE)?;
        variety = serde_json::from_value(serde_json::Value::String(value.clone()))
            .map_err(|e| format!("invalid variety: {e}"))?;
        text_start += 2;
    }
    if args.get(text_start).map(String::as_str) == Some("--") {
        text_start += 1;
    } else if args
        .get(text_start)
        .is_some_and(|arg| arg.starts_with("--"))
    {
        return Err(USAGE.into());
    }
    if text_start >= args.len() {
        return Err(USAGE.into());
    }
    Ok(Request {
        text: args[text_start..].join(" "),
        lang: Some(args[1].clone()),
        variety,
        voice: None,
    })
}

/// Our JSON goes to the original stdout; the process's fd 1 is then pointed
/// at stderr so anything espeak prints with `printf` (it reports an invalid
/// phoneme code that way) cannot land in the middle of a JSON line.
fn take_stdout() -> std::fs::File {
    // SAFETY: dup/dup2 on the standard descriptors; the returned fd is owned
    // by the File and nothing else.
    unsafe {
        let fd = libc::dup(1);
        assert!(fd >= 0, "dup(1) failed");
        libc::dup2(2, 1);
        std::fs::File::from_raw_fd(fd)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = std::io::BufWriter::new(take_stdout());
    let write = |out: &mut std::io::BufWriter<std::fs::File>, r: &Response| {
        serde_json::to_writer(&mut *out, r).unwrap();
        out.write_all(b"\n").unwrap();
    };
    match args.first().map(String::as_str) {
        Some("identity") => writeln!(out, "{}", g2p::identity()).unwrap(),
        Some("serve") => {
            for line in std::io::stdin().lock().lines() {
                let line = line.expect("read stdin");
                if line.trim().is_empty() {
                    continue;
                }
                let response = match serde_json::from_str::<Request>(&line) {
                    Ok(req) => handle(req),
                    Err(e) => Response::Err {
                        error: format!("bad request: {e}"),
                        unlabelable: None,
                    },
                };
                write(&mut out, &response);
                out.flush().unwrap();
            }
        }
        _ => match cli_request(&args) {
            Ok(request) => write(&mut out, &handle(request)),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(2);
            }
        },
    }
    out.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn response(request: Value) -> Value {
        serde_json::to_value(handle(serde_json::from_value(request).unwrap())).unwrap()
    }

    #[test]
    fn language_with_voice_matches_voice_only() {
        assert_eq!(
            response(json!({"text": "zapato", "lang": "spa", "voice": "es-419"})),
            response(json!({"text": "zapato", "voice": "es-419"}))
        );
        assert_eq!(
            response(json!({"text": "zapato", "lang": "spa"})),
            response(json!({"text": "zapato", "voice": "es"}))
        );
    }

    #[test]
    fn unmapped_legacy_voice_is_a_caller_error_not_a_refusal() {
        assert_eq!(
            response(json!({"text": "यह शहर", "lang": "hin", "voice": "hi"})),
            json!({"error": "no variety maps to voice hi; only voices present in the training corpus can be replayed"})
        );
    }

    #[test]
    fn hindi_always_emits_the_trained_current_labels() {
        let result = response(json!({"text": "यह शहर", "lang": "hin"}));
        assert_eq!(
            result["phonemes"],
            json!(["j", "eː", "ʃ", "ɛː", "ɦ", "ɛː", "ɾ"])
        );
        // Removed options have no effect, like other unknown JSON fields.
        assert_eq!(
            result,
            response(json!({"text": "यह शहर", "lang": "hin", "hindi_labels": "legacy"}))
        );
    }

    #[test]
    fn missing_language_and_voice_requires_at_least_one() {
        assert_eq!(
            response(json!({"text": "hello"})),
            json!({"error": "request needs at least one of `voice` or `lang`"})
        );
    }

    #[test]
    fn variety_defaults_and_spanish_options() {
        let default = response(json!({"text": "cinco", "lang": "spa"}));
        for variety in ["default", "european"] {
            assert_eq!(
                response(json!({"text": "cinco", "lang": "spa", "variety": variety})),
                default
            );
        }
        let latin = response(json!({"text": "cinco", "lang": "spa", "variety": "latin_american"}));
        assert_eq!(latin, response(json!({"text": "cinco", "voice": "es-419"})));
        assert_eq!(latin["phonemes"][0], "s");
        assert_eq!(default["phonemes"][0], "θ");
    }

    #[test]
    fn raw_voice_precedes_variety_in_language_and_voice_only_requests() {
        for lang in [None, Some("spa"), Some("fra"), Some("hin"), Some("xx-nope")] {
            let mut request = json!({"text": "cinco", "voice": "es-419", "variety": "european"});
            if let Some(lang) = lang {
                request["lang"] = json!(lang);
            }
            assert_eq!(
                response(request),
                response(json!({"text": "cinco", "voice": "es-419"}))
            );
        }
        assert_eq!(
            response(json!({"text": "", "lang": "hin", "voice": "hi", "variety": "european"})),
            json!({"error": "no variety maps to voice hi; only voices present in the training corpus can be replayed"})
        );
    }

    #[test]
    fn unsupported_variety_is_a_caller_error() {
        assert_eq!(
            response(json!({"text": "", "lang": "tha", "variety": "latin_american"})),
            json!({"error": "variety LatinAmerican is not applicable to language \"tha\""})
        );
        assert_eq!(
            response(json!({"text": "", "variety": "latin_american"})),
            json!({"error": "request needs at least one of `voice` or `lang`"})
        );
    }

    #[test]
    fn invalid_variety_is_rejected_at_protocol_boundary() {
        for value in [
            json!("LatinAmerican"),
            json!("unknown"),
            json!(null),
            json!(1),
        ] {
            // Raw voice precedence does not bypass schema validation.
            let request = json!({"text": "cinco", "voice": "es-419", "variety": value});
            assert!(serde_json::from_value::<Request>(request).is_err());
        }
    }

    #[test]
    fn legacy_inverse_prefers_explicit_varieties_and_roundtrips_every_mapping() {
        use g2p::Variety;
        for (voice, expected) in [
            ("es-419", ("spa", Variety::LatinAmerican)),
            ("es", ("spa", Variety::European)),
            ("pt", ("por", Variety::European)),
            ("pt-br", ("por", Variety::Brazilian)),
            ("en-us", ("eng", Variety::Default)),
        ] {
            assert_eq!(from_espeak_voice(voice).unwrap(), expected);
        }
        for &(lang, _, voice) in voices::ESPEAK_VOICES {
            let (mapped_lang, variety) = from_espeak_voice(voice).unwrap();
            assert_eq!(mapped_lang, lang);
            assert!(voices::ESPEAK_VOICES.contains(&(mapped_lang, variety, voice)));
        }
        for voice in [
            "hi",
            "cmn",
            "en-gb",
            "en-us+f3",
            "English (America)",
            "xx-nope",
        ] {
            assert_eq!(
                from_espeak_voice(voice).unwrap_err(),
                format!(
                    "no variety maps to voice {voice}; only voices present in the training corpus can be replayed"
                )
            );
        }
    }

    #[test]
    fn portuguese_wire_varieties_and_legacy_voices_agree() {
        for (variety, voice) in [
            ("default", "pt-br"),
            ("brazilian", "pt-br"),
            ("european", "pt"),
        ] {
            let typed = response(json!({"text": "dia noite", "lang": "por", "variety": variety}));
            assert!(typed.get("error").is_none());
            assert_eq!(
                typed,
                response(json!({"text": "dia noite", "voice": voice}))
            );
            // The voice wins over both contradictory fields, not just variety.
            assert_eq!(
                typed,
                response(
                    json!({"text": "dia noite", "voice": voice, "lang": "spa", "variety": "latin_american"})
                )
            );
        }
    }

    #[test]
    fn refusal_keeps_unlabelable_reason() {
        let result = Response::from(Err(g2p::Error::Unlabelable("reason:detail".into())));
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            json!({"error": "cannot label this text: reason:detail", "unlabelable": "reason:detail"})
        );
    }
}
