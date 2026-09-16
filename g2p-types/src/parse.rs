//! Parsed pronunciation structure; parsing implementations live in g2p.

use serde::{Deserialize, Serialize};

/// Lexical stress of a token, as espeak marked it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stress {
    None,
    Primary,
    Secondary,
}

impl Stress {
    /// The integer code lexide's corpus files use (0/1/2).
    pub fn code(self) -> u8 {
        match self {
            Stress::None => 0,
            Stress::Primary => 1,
            Stress::Secondary => 2,
        }
    }
}

/// A parsed utterance. `phonemes` and `stress` are parallel; each
/// `word_spans` entry is a half-open `[start, end)` index range into them
/// for one word espeak emitted (empty words are dropped).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Parsed {
    pub phonemes: Vec<String>,
    pub stress: Vec<Stress>,
    pub word_spans: Vec<(usize, usize)>,
}
