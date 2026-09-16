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

`phonemize_language(PhonemizeRequest::new(lang, text))` returns:

- `raw` — for espeak-backed languages, IPA exactly as `espeak-ng -q --ipa -x` prints it
  (stress marks, word boundaries), clauses joined with spaces. For humans and LLMs.
- `phonemes` / `stress` / `word_spans` — versioned pronunciation-model
  labels: stress and boundaries removed, continuation diacritics folded onto
  the previous token, `ʲ` folded onto a preceding consonant, language-switch
  markers stripped, and the units below merged. See `src/parse.rs`.

**Raw-voice and Hindi-version API knobs are removed.** Hindi retains the
trained Current labels as its default; all existing default labels and the
0.4.0 identity are unchanged.

**0.4 introduced the espeak label inventory used by the deployed checkpoint.**
Keep models pinned to the g2p revision used to train them; changing inventories
requires coordinated retraining/relabeling.

| espeak voice | single-token vowel units |
|---|---|
| English (`en-us`, `en-gb`) | `eɪ aɪ ɔɪ aʊ oʊ əʊ`; rhotic `ɑːɹ ɔːɹ ɪɹ ɛɹ ʊɹ` when emitted |
| German (`de`) | `aɪ aʊ ɔʏ ɔø` (Häuser emits `ɔø`) |
| Brazilian Portuguese (`pt-br`) | oral `aʊ eɪ oʊ aɪ`; nasal `ɐ̃ʊ̃ ɐ̃ɪ̃ õɪ̃ ũɪ̃` and mãe's literal `ɐ̃j` |
| Czech (`cs`) | `eɪ oʊ aʊ` |

This inventory describes engine behavior; British English and the additional
voice/alias cases in private engine tests do not expand the public language/
variety set.

Across espeak languages, affricates `tʃ dʒ ts dz tɕ dʑ tʂ dʐ ʈʂ ɖʐ pf bv tθ dð kx ɡɣ`
merge **only inside an actual engine phoneme**, preserving decorations such as
Russian `tʃʲ` and Italian `dzː`. Explicit IPA ties are preserved inside the unit
(e.g. Latvian `t͡s`, Belarusian `d͡zʲ`, Pashto `t͡ʃ`), not emitted as stray labels.
Adjacent `t` + `s` in English cats stays split.
Vowel merges are also phone-boundary-aware, with two narrow same-word coda
exceptions: English vowel + `ɹ` (more/ear are separate engine phones) and
Portuguese `ɐ̃` + `j` (mãe). These exceptions never cross stress/language/word
boundaries or consume a glide/r before another vowel (mirror/hero). The trace
does not provide full syllabification; this coda rule is conservative, not a
syllable parser. British nonrhotic car/air/tour keep their emitted vowels;
we do not invent a rhotic. Nasal labels retain espeak's decomposed `õ`/`ũ`
spellings and mãe's `j`; merging does not normalize or remap IPA. Voice aliases
use the resolved language, and language-switch markers select the appropriate
merge inventory (return markers restore the original regional voice).

The engine callback captures a parallel phone-separated rendering of the
**same** post-pitch/length phoneme list, without a second synthesis or changing
public `raw`. Plain `parse::parse(raw)` retains legacy character segmentation:
raw IPA cannot distinguish an affricate from two neighboring phones. Use
`phonemize_language`/`phonemize_lang` for current labels.

Stress, tone and length handling are unchanged (adjacent vowels still share
stress, even across engine-phone separators). The Japanese, Mandarin, Korean
and Thai backend chains are unchanged. Hindi selection is described below;
neither private Hindi algorithm is changed. Source fixes remove the Persian
q1 artifact (قهوه `q1ˈahveː` → `qˈahveː`) and Russian mnemonic `^` (царь
`tsˈɑrɪ^` → `tsˈɑrɪ`); these corrections also appear in `raw`.

Output is byte-identical to the CLI because it runs the same code path (a
silent synthesis with the phoneme trace on), not the `espeak_TextToPhonemes`
shortcut, which skips the pitch/length passes and differs on tone languages.

`identity()` is a label-compatibility identifier: crate version plus source
digests for the espeak fork and pinned Thai/Korean Python projects. Rust label
changes require an intentional crate version bump. The checked-in source-digest
regression test (`tests/label_compatibility.rs`) forces review of edits to the
engine/shared source trees (including Mandarin model data), manifests, lockfile,
and build script. Whole-file bytes are intentional: even comments trigger review.
For label-preserving changes, verify output and refresh the reviewed digest; for
label changes, bump the crate version and refresh both version/digest baselines.
The test digest is not included in the runtime identity, so API-only refactors
do not invalidate compatible deployed labels.

This is not a complete build fingerprint. Toolchain, platform, environment,
enabled features (including Japanese availability), and downstream dependency
resolution are not encoded. In particular, a downstream library consumer uses
its own lockfile rather than this repository's Cargo.lock. Stamp persisted
labels with the identity and retain the request choices (language, text,
variety); identity alone is not a cache key.

## Languages

`label_source(lang)` is the one table, shared by yap and lexide, of where each
language's phoneme labels come from. Which G2P a language may use is a
correctness constraint, not a preference: targets from a different source
than the model's training labels disagree about the phoneme inventory, and
nothing downstream can tell.

| languages | source |
|---|---|
| eng deu fra ita por spa rus (+ lexide's Pimsleur-era languages) | the espeak fork, variety selected by g2p |
| hin | the built-in Hindi chain (below) |
| zho-hans | the built-in Mandarin chain (below) |
| jpn | OpenJTalk via `jpreprocess` (below) |
| tha | vachana-thai, as an embedded pinned Python project (below) |
| kor | g2pk2 + mecab-ko, as an embedded pinned Python project (below) |

### Pronunciation varieties

Use a language plus `Variety`; backend voice names are private to g2p.
`PhonemizeRequest` holds `lang`, `text`, and `variety`.
`new(lang, text)` chooses `Variety::Default` and the trained model labels; the
`.variety(...)` builder selects another supported reading.

| language | supported varieties |
|---|---|
| Spanish | `Default`/`European`: distinción; `LatinAmerican`: seseo |
| Portuguese | `Default`/`Brazilian`: Brazilian; `European`: European |
| all other supported languages | `Default` only |

Unsupported varieties return `Error::VarietyNotApplicable`; unknown language
codes return `Error::UnsupportedLanguage`. Portuguese training rows include
both Brazilian and European readings, so g2p supports both. Yap's Brazilian
course excludes European clips and does not accept European learner readings;
that is product policy, not a limitation of this engine.

### Korean

espeak's `ko` voice matches Wiktionary on 47% of words: it has no tense
consonants at all (달/딸/탈 collapse), splits affricates and aspirates into
letters, and applies none of the implicit sound changes. Korean G2P is a
settled rule table (the 표준 발음법); [g2pk](https://github.com/Kyubyong/g2pK)
is the standard implementation — `g2pk2` is its maintained fork, used by the
Korean TTS stacks and Montreal Forced Aligner — and matches Wiktionary on
95.6% of words, the only candidate that also gets ㄴ-insertion (꽃잎 [꼰닙])
and morphological tensification (넘다 [넘따], 할 것 [할껏]) because it tags
with mecab-ko first. The remaining ~4% is lexical Sino-Korean tensification
(결점 [결쩜]) that needs a dictionary. `python/korean/` is a `uv` project
pinning `g2pk2`, the prebuilt `mecab-ko` wheel, and `mecab-ko-dic`; the crate
embeds it, unpacks it beside the espeak data, and drives it as a JSON-lines
server that returns each word's pronunciation as post-sandhi Hangul; the
phone mapping runs in Rust (`src/korean.rs`, which documents the label set).
The labels target connected speech: within a clause the sound changes apply
across the spaces between words (못 만났어 [몬만나써], 할 것 [할껏]), as they
do in real audio; punctuation, where speakers pause, splits the text into
clauses that are phonemized separately (안녕, 라디오 keeps its ㄹ). **Needs
`uv` on PATH**; the first call resolves the environment. Digits, Latin,
hanja, and bare jamo are refused. The pins are part of `identity()`.

### Thai

espeak's `th` voice is not a G2P at all (0 of 3,000 Wiktionary words match;
it spells consonants letter by letter). vachana-thai — TLTK's rule/lexicon
front end over pythainlp's dictionary segmenter — matches Wiktionary on 88%
of words segmentally and 87% on tone. It is not ported: its lexicon covers
98.6% of corpus tokens, but the remainder goes through a probabilistic chart
parser with trigram statistics, and refusing those sentences would drop 8% of
the corpus. Instead `python/thai/` is a `uv` project pinning `vachana-g2p`
and `pythainlp` with a lockfile; the crate embeds it, unpacks it beside the
espeak data, and drives it as a JSON-lines server. lexide's label stage
(phone inventory, tone per syllable, stress on each word's final syllable)
runs in Rust. **Needs `uv` on PATH**; the first call resolves the
environment. Mixed Thai/Latin text is refused. The pins are part of
`identity()`.

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

Hindi always emits the deployed model's **Current** labels. There is no public
label-version selector: unified dispatch and `hindi::phonemize(text)` /
`hindi::word(text)` all use the same private model-label constant. The deployed
checkpoint's training sidecar (`phoneme_backend_g2p-hin.jsonl`) emits `ज्ञ` as
`ɡ j`, whereas the historical Legacy chain emits `d͡ʒ ɲ`.

This retains the former Current default, including refusal of digits and Latin
script rather than leaving holes where audio contains speech. Removing the
selector does not change default labels or the 0.4.0 identity. Consumers that
explicitly selected the historical Legacy inventory must use the trained
Current labels instead; downstream scoring/training deployment is coordinated
by their owners rather than exposing an alternate inventory here.

The detailed Legacy/Current distinction remains documented in the private
engine enum. Both implementations retain their unit tests, and the complete
former Legacy JSON output remains an internal regression fixture. The Current
corrections include eligible schwa raising beside `ɦ`, final `ɪ`/`ʊ` lengthening,
velar anusvara, `ज्ञ`, undoing impossible schwa deletions, and digits/Latin refusal.
A future model-label constant change must trip the source-review guard,
deliberately change the crate version/identity, and rederive downstream data
alongside the model.

## Rust

```toml
g2p = { git = "https://github.com/anchpop/g2p", rev = "..." }
```

Consumers that only store or transport labels can depend on `g2p-types` from
the same repository. It has only serde as a normal dependency: no native
engine, build script, dictionary download, or Python backend. Types remain
re-exported from the same g2p root and backend modules; Japanese label types
are available even without the `japanese` engine feature. The Hindi result
structures remain shared, but label selection is private to the engine.
`LabelSource` is a `Copy` enum identifying the backend; `Espeak` is a unit
variant and exposes no engine voice string.

```rust
let p = g2p::phonemize_lang("fra", "on est")?;
assert_eq!(p.phonemes, ["ɔ̃", "n", "ɛ"]);
let h = g2p::phonemize_lang("hin", "यह शहर")?;        // trained Hindi labels
let s = g2p::phonemize_language(
    g2p::PhonemizeRequest::new("spa", "cinco").variety(g2p::Variety::LatinAmerican),
)?;
assert_eq!(s.phonemes[0], "s");
```

Calls are thread-safe (serialized on a lock; espeak has global state).
The Rust API only accepts language and typed variety selection, not raw engine
voice strings. `Error::UnknownVoice` is an internal table/engine invariant
diagnostic, not a caller-input error.

## Command line

```
cargo install --git https://github.com/anchpop/g2p --locked
g2p --lang fra "on est"      # one utterance → JSON
g2p --lang hin "यह शहर"      # default Hindi labels
g2p --lang spa --variety latin_american "cinco"
g2p --lang por --variety european "dia noite"
g2p identity                 # label-compatibility identity
g2p serve                    # JSON lines on stdin/stdout, one utterance per line
```

The CLI accepts `--lang <code> [--variety <name>] [--] <text...>`; `--` ends
option parsing when text starts with `--`. Positional engine voice names are
not supported.

For `serve`, new clients send `{"text": ..., "lang": ...}` with optional
`"variety"`: `"default"` (also when omitted), `"latin_american"`, `"european"`,
or `"brazilian"`. For example:
`{"text": "cinco", "lang": "spa", "variety": "latin_american"}`.

Until the legacy Python transport is removed, JSON requests may still carry
`"voice"`. The adapter converts corpus voice names to a language and variety:
`es-419` → Spanish/LatinAmerican, `es` → Spanish/European, `pt-br` →
Portuguese/Brazilian, `pt` → Portuguese/European; other mapped corpus voices
select their language's default. This selection wins over both supplied
`lang` and `variety`, even if contradictory. Unmapped names fail with
`no variety maps to voice X; only voices present in the training corpus can be replayed`.
There is no raw-engine fallback or public Rust voice adapter. Unknown variety
names and null remain schema errors even with a legacy voice. A request without
either `lang` or legacy `voice` is an error, not an implicit language.

Each line is exactly one utterance, so the clause-versus-line framing ambiguity
of `espeak-ng --stdin` cannot occur. Responses carry `syllables` when the backend
computes them, and a refusal comes back as
`{"error": ..., "unlabelable": "reason:detail"}`.

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
