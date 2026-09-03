//! `g2p` — command-line front end for non-Rust consumers (lexide's Python
//! pipeline).
//!
//! ```text
//! g2p identity                 print the build identity (for cache keys)
//! g2p <voice> <text...>        phonemize one utterance, print JSON
//! g2p serve                    JSON lines: one request per line on stdin,
//!                              one response per line on stdout, flushed
//!                              after each — keep one process running and
//!                              stream requests through it.
//! ```
//!
//! Request:  `{"text": "on est", "voice": "fr-fr"}`
//! Response: `{"raw": "ɔ̃ nˈɛ", "phonemes": ["ɔ̃","n","ɛ"], "stress": [0,0,1],
//!            "word_spans": [[0,1],[1,3]]}` or `{"error": "..."}`.
//!
//! Each request is phonemized as exactly one utterance, so the clause-vs-line
//! framing ambiguity of `espeak-ng --stdin` (a comma splits a line in two, a
//! line without terminal punctuation merges into the next) cannot occur.

use std::io::{BufRead, Write};
use std::os::fd::FromRawFd;

#[derive(serde::Deserialize)]
struct Request {
    text: String,
    voice: String,
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum Response {
    Ok {
        raw: String,
        phonemes: Vec<String>,
        stress: Vec<u8>,
        word_spans: Vec<(usize, usize)>,
    },
    Err {
        error: String,
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
            },
            Err(e) => Response::Err {
                error: e.to_string(),
            },
        }
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
    match args.first().map(String::as_str) {
        Some("identity") => writeln!(out, "{}", g2p::identity()).unwrap(),
        Some("serve") => {
            for line in std::io::stdin().lock().lines() {
                let line = line.expect("read stdin");
                if line.trim().is_empty() {
                    continue;
                }
                let response = match serde_json::from_str::<Request>(&line) {
                    Ok(req) => Response::from(g2p::phonemize(&req.text, &req.voice)),
                    Err(e) => Response::Err {
                        error: format!("bad request: {e}"),
                    },
                };
                serde_json::to_writer(&mut out, &response).unwrap();
                out.write_all(b"\n").unwrap();
                out.flush().unwrap();
            }
        }
        Some(voice) if args.len() >= 2 => {
            let text = args[1..].join(" ");
            let response = Response::from(g2p::phonemize(&text, voice));
            serde_json::to_writer(&mut out, &response).unwrap();
            out.write_all(b"\n").unwrap();
        }
        _ => {
            eprintln!("usage: g2p identity | g2p serve | g2p <voice> <text...>");
            std::process::exit(2);
        }
    }
    out.flush().unwrap();
}
