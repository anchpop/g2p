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

## Rust

```toml
g2p = { git = "https://github.com/anchpop/g2p", rev = "..." }
```

```rust
let p = g2p::phonemize("on est", "fr-fr")?;
assert_eq!(p.phonemes, ["ɔ̃", "n", "ɛ"]);
```

Voices are espeak voice names (`fr-fr`, `en-us`, `pt-br`, `cmn`, `ru`, …),
resolved the way the CLI's `-v` resolves them. Calls are thread-safe
(serialized on a lock; espeak has global state).

## Command line

```
cargo install --git https://github.com/anchpop/g2p --locked
g2p fr-fr "on est"      # one utterance → JSON
g2p identity            # build identity
g2p serve               # JSON lines on stdin/stdout, one utterance per line
```

`serve` is how lexide's Python uses it: keep one process running and stream
`{"text": ..., "voice": ...}` requests through it. Each line is exactly one
utterance, so the clause-versus-line framing ambiguity of `espeak-ng --stdin`
cannot occur.

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
