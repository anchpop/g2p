# g2p

Grapheme-to-phoneme for [yap](https://github.com/yaptown/yap) and
[lexide](https://github.com/anchpop/lexide), built on the maintainer's
[espeak-ng fork](https://github.com/anchpop/espeak-ng) (branch
`french-phrase-stress-liaison`).

The fork is a git submodule. `build.rs` compiles it with CMake, links it
statically, and embeds its compiled phoneme data in the binary. Consumers get
one thing to depend on and nothing to install or configure: no `ESPEAK_NG_BIN`,
no `ESPEAK_NG_DATA_PATH`, no way to run against mainline espeak by mistake.

## Output

`phonemize(text, voice)` returns:

- `raw` — espeak's IPA exactly as `espeak-ng -q --ipa -x` prints it (stress
  marks, word boundaries), clauses joined with spaces. For humans and LLMs.
- `phonemes` / `stress` / `word_spans` — the tokenization the lexide
  pronunciation model was trained on: stress and boundaries removed,
  continuation diacritics folded onto the previous token, `ʲ` folded onto a
  preceding consonant, language-switch markers stripped, each half of a
  diphthong its own token. Anything that scores audio against the model
  must use this form. See `src/parse.rs`.

Output is byte-identical to the CLI because it runs the same code path (a
silent synthesis with the phoneme trace on), not the `espeak_TextToPhonemes`
shortcut, which skips the pitch/length passes and differs on tone languages.

`identity()` returns a string keyed on a digest of every fork source file that
affects output. Stamp persisted phoneme data with it.

## Languages

`label_source(lang)` is the one table, shared by yap and lexide, of where each
language's phoneme labels come from. Which G2P a language may use is a
correctness constraint, not a preference: targets from a different source
than the model's training labels disagree about the phoneme inventory, and
nothing downstream can tell.

| languages | source |
|---|---|
| eng deu fra ita por spa rus (+ kor, unvalidated, and lexide's Pimsleur-era languages) | the espeak fork, one voice each |
| hin | the built-in Hindi chain (below) |
| zho-hans | the built-in Mandarin chain (below) |
| jpn | OpenJTalk via `jpreprocess` (below) |
| tha | a Python backend lexide runs outside this crate; `phonemize_lang` refuses it |

### Japanese

espeak's `ja` voice is not used. `src/japanese` runs OpenJTalk's text front
end through [jpreprocess](https://github.com/jpreprocess/jpreprocess), a
Rust rewrite with the NAIST dictionary bundled (downloaded at build time and
embedded; the binary grows by ~85 MB), then applies lexide's label stage:
OpenJTalk phones to IPA, the sokuon closure as length on the following
obstruent, and a Tokyo pitch level per mora from each accent phrase's nucleus
and mora count. Accent is withheld (phones kept) for fragments whose first
content word is a particle, auxiliary, or suffix, or when the parse is
self-inconsistent. Against pyopenjtalk on the lexide corpus, 99.2% of
sentences label identically; the rest differ in readings of Latin
abbreviations and digit-plus-counter words (ケイ vs ケー, 年 vs とし) and in
accent-phrase chaining.

### Mandarin

espeak's `cmn` voice is not used. `src/mandarin` is a port of g2pM
(kakaobrain, MIT): a CEDICT digest gives each character its readings and a
small BiLSTM picks the reading for the 791 polyphonic characters from
sentence context; weights and dictionary are embedded (~1.7 MB). Pinyin
becomes IPA through the `pinyin_to_ipa` package's tables, precomputed for
every syllable g2pM can emit. Labels carry a tone number on each syllable's
tone-bearing phone. Output matches lexide's Python chain on every one of the
18,357 corpus sentences both can label; text with digits, Latin letters, or
characters outside the dictionary is refused rather than labeled with a hole
(the Python chain silently dropped such characters, 2,694 corpus rows).

### Hindi

espeak's `hi` voice is not used. `src/hindi` is a port of lexide's
`schwa-stress-hin` chain: Devanagari → phone units, the ACL 2020
logistic-regression schwa-deletion classifier (aryamanarora/schwa-deletion,
MIT; weights embedded), a unit → IPA map, and Roy's (2017) surface
syllable-weight stress rules with syllable spans.

Two label conventions, chosen with `HindiCanon`:

- `Legacy` is byte-identical to the Python chain on the entire lexide corpus
  (13,086 sentences: phonemes, stress, syllables). It is what the deployed
  pronunciation model was trained on, so it is what yap scores against.
- `Current` adds the corrections from a 2026-09-02 audit against Wiktionary
  and the schwa repo's gold lists: `/ə/` beside `/ɦ/` is `[ɛ]` (शहर, कहना,
  बहन; यह/वह are `[jeː]`/`[ʋoː]`), anusvara before velars is `ŋ`, ज्ञ is
  `[ɡj]`, word-final short ɪ/ʊ are `iː`/`uː`, and a schwa deletion that would
  leave an unpronounceable consonant run (दुश्मनों → `ʃmn`) is undone. Text
  with digits or Latin letters is refused (`Error::Unlabelable`) rather than
  labeled with a hole where the audio has speech.

## Rust

```toml
g2p = { git = "https://github.com/anchpop/g2p", rev = "..." }
```

```rust
let p = g2p::phonemize("on est", "fr-fr")?;          // by espeak voice
assert_eq!(p.phonemes, ["ɔ̃", "n", "ɛ"]);
let h = g2p::phonemize_lang("hin", "यह शहर")?;        // by language, current canon
let l = g2p::phonemize_lang_with("hin", "यह शहर", g2p::HindiCanon::Legacy)?;
```

Voices are espeak voice names (`fr-fr`, `en-us`, `pt-br`, `cmn`, `ru`, …),
resolved the way the CLI's `-v` resolves them. Calls are thread-safe
(serialized on a lock; espeak has global state).

## Command line

```
cargo install --git https://github.com/anchpop/g2p --locked
g2p fr-fr "on est"           # one utterance → JSON
g2p --lang hin "यह शहर"      # by language
g2p identity                 # build identity
g2p serve                    # JSON lines on stdin/stdout, one utterance per line
```

`serve` is how lexide's Python uses it: keep one process running and stream
`{"text": ..., "voice": ...}` or `{"text": ..., "lang": ..., "canon": ...}`
requests through it. Each line is exactly one utterance, so the
clause-versus-line framing ambiguity of `espeak-ng --stdin` cannot occur.
Responses carry `syllables` when the backend computes them, and a refusal
comes back as `{"error": ..., "unlabelable": "reason:detail"}`.

## Building

Needs `cmake` and a C compiler. Clone with `--recurse-submodules` (cargo does
this for git dependencies). First build compiles espeak-ng and its
dictionaries, roughly a minute.

To move to a new fork commit: `git -C espeak-ng checkout <rev>`, commit the
submodule pointer, and bump the crate version. Consumers pin by `rev`, so yap
(which must match the deployed pronunciation model's labels) and lexide
(which may be relabeling for the next model) can point at different builds.

## License

espeak-ng is GPL-3.0-or-later and is linked statically, so this crate is too.
