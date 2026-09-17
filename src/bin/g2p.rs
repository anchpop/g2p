//! `g2p` — command-line front end for non-Rust consumers (lexide's Python
//! pipeline).
//!
//! ```text
//! g2p identity                 print the build identity (for cache keys)
//! g2p --lang <language> [--] <text...>
//!                              phonemize by combined language
//! g2p serve                    JSON lines: one request per line on stdin,
//!                              one response per line on stdout, flushed
//!                              after each — keep one process running and
//!                              stream requests through it.
//! ```
//!
//! Request: `{"text": "यह शहर", "lang": "hin"}`. Hindi always uses
//! the deployed model's labels; there is no wire-level label version.
//! Requests contain only `text` and a combined `lang`, for example
//! `{"text": "cinco", "lang": "spa-419"}`. Spanish and Portuguese require
//! an explicit region: spa-ES, spa-419, por-BR, or por-PT.
//!
//! Each request is phonemized as exactly one utterance, so the clause-vs-line
//! framing ambiguity of `espeak-ng --stdin` (a comma splits a line in two, a
//! line without terminal punctuation merges into the next) cannot occur.

use std::io::{BufRead, Write};
use std::os::fd::FromRawFd;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    text: String,
    lang: g2p::Language,
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
    g2p::phonemize(req.lang, &req.text).into()
}

const USAGE: &str = "usage: g2p identity | g2p serve | g2p --lang <language> [--] <text...>";

fn cli_request(args: &[String]) -> Result<Request, String> {
    if args.first().map(String::as_str) != Some("--lang") || args.len() < 3 {
        return Err(USAGE.into());
    }
    let mut text_start = 2;
    let lang = serde_json::from_value(serde_json::Value::String(args[1].clone()))
        .map_err(|e| format!("invalid language: {e}"))?;
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
        lang,
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
    use serde_json::json;

    #[test]
    fn wire_accepts_only_text_and_a_combined_language() {
        let req: Request =
            serde_json::from_value(json!({"lang": "spa-419", "text": "cinco"})).unwrap();
        assert_eq!(
            serde_json::to_value(handle(req)).unwrap()["phonemes"][0],
            "s"
        );
        for field in ["voice", "variety", "canon", "hindi_labels"] {
            let mut req = json!({"lang": "eng", "text": "hello"});
            req[field] = json!("obsolete");
            assert!(serde_json::from_value::<Request>(req).is_err());
        }
        for lang in ["spa", "por", "spa-BR", "unknown"] {
            assert!(serde_json::from_value::<Request>(json!({"lang": lang, "text": ""})).is_err());
        }
    }

    #[test]
    fn refusal_keeps_unlabelable_reason() {
        let req: Request = serde_json::from_value(json!({"lang": "hin", "text": "19"})).unwrap();
        let result = serde_json::to_value(handle(req)).unwrap();
        assert!(
            result["unlabelable"]
                .as_str()
                .unwrap()
                .starts_with("hindi_digits:")
        );
    }
}
