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

`phonemize(language, text)` returns:

- `raw` — for espeak-backed languages, IPA exactly as `espeak-ng -q --ipa -x` prints it
  (stress marks, word boundaries), clauses joined with spaces. For humans and LLMs.
- `phonemes` / `stress` / `word_spans` — versioned pronunciation-model
  labels: stress and boundaries removed, continuation diacritics folded onto
  the previous token, `ʲ` folded onto a preceding consonant, language-switch
  markers stripped, and the units below merged. See `src/parse.rs`.
- `word_text_spans` — original-input UTF-8 byte ranges parallel to `word_spans`,
  ordered and nonoverlapping. Slice the original text, not `raw`. Consecutive
  phone words with overlapping source ranges become one word span. If any part of
  an utterance cannot be attributed confidently, this entire field is empty;
  pronunciation and prosody are still returned. Constructors without source text
  leave it empty, and old serialized results deserialize with an empty field.
- `phoneme_text_spans` — the same, per phoneme: the text each phoneme was
  spoken from, within its word's span, so a word can be split between the
  pieces of text inside it. espeak: the letters of the rule match, prefix,
  suffix or spelled letter that produced it (c'est → `s`<c> `ɛ`<est>,
  Hausaufgabe → Haus|auf|gabe), the whole word for a dictionary entry, and
  each word of a dictionary phrase that spells a `||` (word boundary) or
  `||+` (source break only, the phrase stays one phonological word) per word
  ("in the", "en plus", "не было"; phrases without one still own all their
  words). Japanese: the
  kana of its mora, or the kanji its mora was read from (by the supplied
  readings; a lone kanji run without one takes the moras left over), else
  the NJD node. Korean: the syllable it is pronounced in (값이 [갑씨] gives
  `s͈` to 이). Mandarin: the character. Hindi and Thai: the word. A long
  consonant joined from two sources (Japanese っ, Italian geminates) spans
  both. Empty whenever `word_text_spans` is.

Attribution follows the backend's actual units: Hindi source runs, Mandarin
characters, Japanese NJD nodes, Korean eojeol, and Thai segmented surfaces.
Japanese closures spanning nodes merge their source ranges; long vowels and
kana normalization retain the original byte ranges. Thai repetition uses the
backend's own pre-expansion tokens, so base plus ๆ owns every repeated phone,
even when final segmentation splits or joins the copies. Exact backend
normalization is supported; dropped or unalignable surfaces still fall back.
eSpeak carries independent, wide character ranges on every phone through its
list passes, including inserted phones. Dictionary phrases own all consumed
words; each rendered word owns the union of its phones' ranges. Synthetic
pronunciation boundaries, clitics, hyphens, and number expansions therefore do
not require guessed offsets or impose the old packed-position limits.

A 3,000-sentence-per-language attribution sweep covers the twelve course
languages. Remaining empty attribution can be genuine backend omissions (for
example an unread Japanese kanji or a Thai token with empty pronunciation),
not merely unusual spelling or a mismatch between text and phone word counts.

`phonemes` contains the shared `g2p_types::Phoneme` enum, not arbitrary strings.
`Phoneme::as_str()`/`Display` give its exact IPA spelling; serde stores those same
strings. Tokens are matched after NFC normalization, and the inventory is stored
in NFC. Parsing is fallible: CTC controls, unsupported labels and notation
fragments never become phonemes. Checkpoint IDs do not define enum values.
The inventory includes supported G2P tokens, acoustic-training contrasts and
observed WikiPron segmental tokens. A known phone may still be absent from a
particular model vocabulary; lexide reports that separately.

For tokenized dictionary IPA, use
`Phonemized::from_ipa_tokens("ˈt͡ʃ oʊ | ts a")?`. Optional `|` separates words.
Stress and syllable/liaison notation remain in `raw`; unavailable prosodic
annotations stay empty. Segment boundaries are preserved, and an unknown token
returns `UnknownPhoneme`. This does not guess segmentation of continuous IPA.
Use `Phonemized::from_words` for already-typed words. Both constructors live in
`g2p-types`, so consumers need no native engine.

**Raw-voice and Hindi-version API knobs are removed.** Hindi uses the
pronunciation rules described below.

**0.5 makes g2p the owner of pronunciation normalization.** Consumers use the
returned phones directly. English exact `ɐ`/`ᵻ` become `ə`; French loanword
vowels use the French inventory (including removal of listed vowel lengths);
Italian exact `ɪ`/`ʊ` become `i`/`u`, including code-switched words. Modified
variants remain distinct. Greek `ε` is corrected to IPA `ɛ`, and leaked
punctuation/syllable separators and their attached modifiers are omitted.
These rules previously lived in lexide's training preprocessing. Model
vocabulary validation and recording-specific acoustic corrections remain with
the consumer. `hʲ` is preserved: one malformed corpus record is not a reason
to delete a phone globally. English requests containing Hangul are explicitly
refused as `english_hangul`, preventing an implicit switch to eSpeak Korean
for bilingual instructional records. Build identity records provenance; consumers can
key expensive downstream results by the expected phones themselves.

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
`phonemize`/`phonemize_lang` for current labels.

Adjacent vowels still share stress, even across engine-phone separators.
Cantonese and Vietnamese tone numbers are carried in `tone` on the engine’s
tone-bearing phone, rather than in `phonemes`; `raw` keeps the original trace.
Tone codes retain eSpeak’s numbering (1–7), including its contextual tone 7. Length is preserved except for
the French vowel replacements described above. All backends then share a
word-internal geminate pass: identical adjacent consonants become a single
long token when the inventory supports it (`k k` → `kː`). Different modifiers
never merge (`k kʰ` stays split), nor do vowels or phones across word boundaries.
The first consonant's stress/tone/pitch is retained and word/syllable indices
are remapped; diagnostic `raw` is not rewritten. Japanese long vowels remain
separate moras. Arabic requests containing ASCII or Arabic-Indic digits are
refused with `arabic_digits` rather than silently mislabeling spoken numbers.
Latin American Spanish `ll` is `ʝ` in every position (the fork's `es_rules`
had intervocalic `ll` as a long `j`, so ella/calle did not share a token with
word-initial llegar or with y). Source fixes remove the Persian
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

Use `phonemize(Language, text)`. `Language` is defined in the lightweight
`g2p-types` crate and re-exported by g2p and lexide. It selects the language and
pronunciation variety together, for example `SpanishEuro`, `SpanishLatinAmerica`,
`PortugueseBrazil`, or `PortugueseEuro`. Backend voice names stay inside g2p.

`Language::code()` groups regional variants under the same language code for
text analysis. `Language::from_code()` resolves language-only input using the
established defaults: European Spanish and Brazilian Portuguese.

The language-only `phonemize_lang` helper remains for yap and resolves its code
to the same typed implementation. The Rust API has no separate variety selector.

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

pythainlp's dictionary segmenter joins words its dictionary lists as one
(ได้มา, ที่จะ, ดีแล้ว), takes lookalikes (มา + นะ as มานะ "perseverance") and
sometimes splits wrongly (แล้ว จะ as แล้ | วจะ). A caller that knows the words
passes them as `Hints { words, .. }` (byte ranges of the text): vachana then
converts each word on its own (same text preparation, its dictionary, else
TLTK), each word's phonemes are spoken from it, and stress falls on each
given word's final syllable. An unknown word that TLTK transcribes only in
part (ดีพโรส as ดี alone) is read as the segmenter would split it, so no
syllable is dropped. Pass lexical words, not multi-word chunks: those miss
the dictionary and go to TLTK, which is slow and less accurate. Without
words the result is unchanged.

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
every syllable g2pM can emit. Diphthongs use the shared `aɪ aʊ eɪ oʊ`
tokens rather than Mandarin-only non-syllabic spellings. Labels carry a
**surface** tone number on each syllable's tone-bearing phone, consistent
with Japanese's realized H/L targets: in a consecutive third-tone run all
but the last become tone 2; 不 becomes 2 before citation tone 4 (otherwise 4);
一 becomes 4 before citation tones 1–3 and 2 before 4. 一 stays 1 finally,
beside a numeral character, after 第/weekday/compound heads (星期一 统一 唯一), or before 月/号/日. Context uses citation tones across
word boundaries and soft punctuation, but never across `。！？，；：.!?,;:`.
This is deterministic sandhi, not prosodic inference. Pinyin and `raw` retain
lexical tones. Text with digits, Latin letters, or characters outside the
dictionary is refused rather than labeled with a hole.

### Supplied readings (Japanese and Mandarin)

Both languages guess the pronunciation of ambiguous text from context, and a
caller often knows better: it shows the learner a reading, or the text is one
word with no context at all (方: かた or ほう?). `phonemize_with` takes
`Hints { readings, .. }`: `Reading { start, end, reading }` items over byte
ranges of the text, and returns labels that agree with every one of them, or
an error. Text
outside the readings is predicted as before, and with no readings the result
is exactly `phonemize`'s.

* **Mandarin**: pinyin in citation tones, one syllable per character,
  tone-marked or numbered; no tone is the neutral tone and `r` is erhua. The
  reading replaces g2pM's choice for that character; sandhi still applies on
  top, and a supplied neutral tone stays neutral. A syllable outside the
  table (`zhei4`, `hng`) is refused.
* **Japanese**: kana for the range. Give each kanji (or indivisible run, as
  ruby would group it) its own reading: a reading is never split across the
  tokenizer's words, so finer readings keep more of the dictionary. Words
  that already say the readings are left alone. A word that does not is
  replaced by the dictionary's entry with that reading (accent and sound
  changes intact); when the dictionary has none (names, colloquial forms,
  mis-tokenized text), a word that says exactly the supplied kana is used and
  `accent_withheld` is `japanese_reading_not_in_dictionary`. Refused: a word
  only partly covered by readings (`japanese_reading_partial`); text the
  front end rewrites so that it no longer lines up with the input, which in
  practice means digits (`japanese_reading_alignment`); a reading that a
  later rule overrides even when forced, such as 一 as ひと in 一組, which
  the number rules rewrite (`japanese_reading_unenforceable`); a reading that
  is not plain kana (`japanese_reading_not_kana`); and a reading the
  dictionary lacks that takes in okurigana and has an おう or えい, because
  it does not say whether that is one long vowel or two
  (`japanese_reading_vowel_length`).

On Yap's packs (2026-10-25, every displayed reading passed in): Mandarin
agrees on all 1,484,059 characters, 17 of 191,185 sentences refused for an
unknown syllable; Japanese leaves 94.5% of 239,205 sentences unchanged, fixes
4.5% from the dictionary, forces 0.7%, and newly refuses 72.

### Hindi

espeak's `hi` voice is not used. `src/hindi` is a port of lexide's
`schwa-stress-hin` chain: Devanagari → phone units, the ACL 2020
logistic-regression schwa-deletion classifier (aryamanarora/schwa-deletion,
MIT; weights embedded), a unit → IPA map, and Roy's (2017) surface
syllable-weight stress rules with syllable spans.

Hindi uses one set of pronunciation rules across unified dispatch,
`hindi::phonemize(text)` and `hindi::word(text)`: eligible schwa raising beside
`ɦ`, final `ɪ`/`ʊ` lengthening, velar anusvara, `ज्ञ` as `ɡ j`, and restoration
of impossible schwa deletions. Affricates use the shared untied tokens
`tʃ tʃʰ dʒ dʒʱ`; unified dispatch also merges word-internal geminates.
Digits and Latin script are explicitly refused
rather than leaving holes in the labels. There is no label-version selector.

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
let p = g2p::phonemize(g2p::Language::French, "on est")?;
assert_eq!(p.phonemes, ["ɔ̃", "n", "ɛ"]);
let h = g2p::phonemize(g2p::Language::Hindi, "यह शहर")?;        // trained Hindi labels
let s = g2p::phonemize(g2p::Language::SpanishLatinAmerica, "cinco")?;
assert_eq!(s.phonemes[0], "s");
```

Calls are thread-safe (serialized on a lock; espeak has global state).
The Rust API selects pronunciation with the combined language enum. `Error::UnknownVoice` is an internal table/engine invariant
diagnostic, not a caller-input error.

g2p is a Rust library; callers use `g2p::phonemize(Language, text)` directly.

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
