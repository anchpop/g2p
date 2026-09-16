//! `g2p` — command-line front end for non-Rust consumers (lexide's Python
//! pipeline).
//!
//! ```text
//! g2p identity                 print the build identity (for cache keys)
//! g2p <voice> <text...>        phonemize one utterance with an espeak voice
//! g2p --lang <code> <text...>  phonemize by language (see label_source)
//! g2p serve                    JSON lines: one request per line on stdin,
//!                              one response per line on stdout, flushed
//!                              after each — keep one process running and
//!                              stream requests through it.
//! ```
//!
//! Request:  `{"text": "on est", "voice": "fr-fr"}` or
//!           `{"text": "यह शहर", "lang": "hin", "hindi_labels": "legacy"}`
//!           (`hindi_labels` is optional and only affects Hindi; default `current`).
//!           `voice` may accompany `lang` as a dialect/voice override, e.g.
//!           `{"text": "zapato", "lang": "spa", "voice": "es-419"}`.
//! Response: `{"raw": "ɔ̃ nˈɛ", "phonemes": ["ɔ̃","n","ɛ"], "stress": [0,0,1],
//!            "word_spans": [[0,1],[1,3]]}` plus `"syllables": [...]` when
//!            the backend computes them, or `{"error": "...",
//!            "unlabelable": "reason:detail"}` (the second key only when the
//!            backend refused the text rather than failed).
//!
//! Each request is phonemized as exactly one utterance, so the clause-vs-line
//! framing ambiguity of `espeak-ng --stdin` (a comma splits a line in two, a
//! line without terminal punctuation merges into the next) cannot occur.

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
    hindi_labels: Option<g2p::HindiLabels>,
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

fn handle(req: Request) -> Response {
    match (req.voice, req.lang) {
        (Some(voice), None) => Response::from(g2p::phonemize(&req.text, &voice)),
        (voice, Some(lang)) => Response::from(g2p::phonemize_language(
            &lang,
            voice.as_deref(),
            &req.text,
            req.hindi_labels.unwrap_or(g2p::HindiLabels::Current),
        )),
        _ => Response::Err {
            error: "request needs exactly one of `voice` or `lang`".into(),
            unlabelable: None,
        },
    }
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
    fn backend_voice_override_is_a_caller_error_not_a_refusal() {
        assert_eq!(
            response(json!({"text": "यह शहर", "lang": "hin", "voice": "hi"})),
            json!({"error": "voice \"hi\" is not applicable to language \"hin\""})
        );
    }

    #[test]
    fn language_labels_are_preserved() {
        let expected = Response::from(g2p::phonemize_lang_with(
            "hin",
            "यह शहर",
            g2p::HindiLabels::Legacy,
        ));
        assert_eq!(
            response(json!({"text": "यह शहर", "lang": "hin", "hindi_labels": "legacy"})),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[test]
    fn missing_language_and_voice_keeps_existing_error() {
        assert_eq!(
            response(json!({"text": "hello"})),
            json!({"error": "request needs exactly one of `voice` or `lang`"})
        );
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
        Some("--lang") if args.len() >= 3 => {
            let r = Response::from(g2p::phonemize_lang(&args[1], &args[2..].join(" ")));
            write(&mut out, &r);
        }
        Some(voice) if args.len() >= 2 && !voice.starts_with('-') => {
            let r = Response::from(g2p::phonemize(&args[1..].join(" "), voice));
            write(&mut out, &r);
        }
        _ => {
            eprintln!(
                "usage: g2p identity | g2p serve | g2p <voice> <text...> | g2p --lang <code> <text...>"
            );
            std::process::exit(2);
        }
    }
    out.flush().unwrap();
}
